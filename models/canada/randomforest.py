import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
import joblib

PATHOGEN_FILE = r"/content/CanadaPathogenIncSevData.xlsx"
CLIMATE_FILE  = r"/content/CanadaMonthyClimateData.csv"
PROVINCE_CODES = {"AB", "MB", "SK"}

SUSCEPTIBLE_MONTHS = [5, 6, 7, 8]   # May–Aug
LABEL_THRESHOLD = 0.01              # <=0.01 -> 0, >0.01 -> 1

TEST_FRACTION_YEARS = 0.20

N_ESTIMATORS = 1400
RANDOM_STATE = 42
N_JOBS = -1
CLASS_WEIGHT = "balanced_subsample"
def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def pathogenimpact_to_class(x, thr=0.01):
    if pd.isna(x):
        return np.nan
    return 0 if x <= thr else 1

def time_split_by_year(df, year_col="Year", test_fraction_years=0.2):
    years = sorted(df[year_col].dropna().unique().tolist())
    if len(years) < 5:
        raise ValueError("Not enough years for a time split.")
    split_index = max(1, int(round(len(years) * (1 - test_fraction_years))))
    split_year = years[split_index - 1]
    train = df[df[year_col] <= split_year].copy()
    test  = df[df[year_col] >  split_year].copy()
    return train, test, split_year

def hierarchical_impute_monthly(climate_df, numeric_cols):
    out = climate_df.copy()
    # 1) Region x Month mean
    for col in numeric_cols:
        out[col] = out[col].fillna(out.groupby(["Region", "Month"])[col].transform("mean"))
    # 2) Month mean across all regions
    for col in numeric_cols:
        out[col] = out[col].fillna(out.groupby("Month")[col].transform("mean"))
    # 3) Global mean
    for col in numeric_cols:
        out[col] = out[col].fillna(out[col].mean())
    return out

def make_seasonal_features(climate_df, months):
    keep_cols = [
        "temperature_2m_max",
        "vapour_pressure_deficit_max",
        "soil_moisture_0_to_7cm_mean",
        "precipitation_sum",
        "relative_humidity_2m_mean",
    ]
    keep_cols = [c for c in keep_cols if c in climate_df.columns]

    climate_df = ensure_numeric(climate_df, keep_cols)
    climate_df = hierarchical_impute_monthly(climate_df, keep_cols)

    climate_df = climate_df[climate_df["Month"].isin(months)].copy()
    suffix = f"_m{months[0]}-{months[-1]}"

    feats = (
        climate_df
        .groupby(["Region", "Year"], as_index=False)
        .agg({
            "temperature_2m_max": "max",
            "vapour_pressure_deficit_max": "max",
            "soil_moisture_0_to_7cm_mean": "mean",
            "precipitation_sum": "sum",
            "relative_humidity_2m_mean": "mean",
        })
        .rename(columns={
            "temperature_2m_max": f"temp_max{suffix}",
            "vapour_pressure_deficit_max": f"vpd_max{suffix}",
            "soil_moisture_0_to_7cm_mean": f"soil_moist_mean{suffix}",
            "precipitation_sum": f"precip_sum{suffix}",
            "relative_humidity_2m_mean": f"humidity_mean{suffix}",
        })
    )
    return feats
def plot_roc_like_example(y_true, y_score, model_name="Random Forest", save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2, label="No Skill")
    ax.plot(fpr, tpr, marker="o", linewidth=2, label=f"{model_name} (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate for Pathogen Outbreak")
    ax.set_ylabel("True Positive Rate for Pathogen Outbreak")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left")
    ax.grid(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

def plot_confusion_matrix_row_percent(y_true, y_pred, labels=(0, 1), title="Confusion Matrix (Row %)", save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted Pathogen Outbreak",
        ylabel="True Pathogen Outbreak",
        title=title
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                f"{cm_pct[i, j]:.1f}%",
                ha="center", va="center",
                color="white" if cm_pct[i, j] > 50 else "black",
                fontsize=11
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print("\nRaw confusion matrix counts:")
    print(cm)

def make_pretty_feature_name(col_name: str) -> str:
    base = col_name.split("_m")[0]
    mapping = {
        "temp": "Maximum Temperature",
        "vpd": "Vapour Pressure Deficit",
        "soil": "Soil Moisture",
        "precip_sum": "Total Precipitation",
        "humidity": "Relative Humidity"
    }
    return mapping.get(base, base).strip()

def plot_feature_importance(model, feature_cols, top_n=20, save_path="feature_importance_top20.png"):
    if not hasattr(model, "feature_importances_"):
        raise ValueError("This model does not have feature_importances_ (not a tree model).")

    fi = pd.DataFrame({
        "feature_code": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    fi["feature_name"] = fi["feature_code"].apply(make_pretty_feature_name)
    top = fi.head(top_n).copy()

    plt.figure(figsize=(10, 6))
    plt.barh(top["feature_name"][::-1], top["importance"][::-1])
    plt.xlabel("Feature Importance (Canada)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    fi.to_csv("feature_importances.csv", index=False)
    print(f"Saved: {save_path}, feature_importances.csv")
    return fi
def main():
    print("Loading files...")
    pathogen_df = pd.read_excel(PATHOGEN_FILE)
    climate_df  = pd.read_csv(CLIMATE_FILE)
    if "Region" not in pathogen_df.columns:
        raise ValueError("Pathogen file must contain a 'Region' column.")
    before = len(pathogen_df)
    pathogen_df = pathogen_df[~pathogen_df["Region"].isin(PROVINCE_CODES)].copy()
    after = len(pathogen_df)
    print(f"Dropped province-average rows: {before - after} removed, {after} remaining")
    pathogen_df["Year"] = pd.to_numeric(pathogen_df["Year"], errors="coerce").astype("Int64")
    pathogen_df = ensure_numeric(pathogen_df, ["PathogenImpact"])
    pathogen_df["Label"] = pathogen_df["PathogenImpact"].apply(lambda x: pathogenimpact_to_class(x, LABEL_THRESHOLD))
    pathogen_df = pathogen_df.dropna(subset=["Label"]).copy()
    train_p, test_p, split_year = time_split_by_year(pathogen_df, "Year", TEST_FRACTION_YEARS)
    print(f"Time split: Train <= {split_year}, Test > {split_year}")
    print(f"Train pathogen rows: {len(train_p)} | Test pathogen rows: {len(test_p)}")

    months = SUSCEPTIBLE_MONTHS
    feats = make_seasonal_features(climate_df, months)
    feature_cols = [c for c in feats.columns if c not in ["Region", "Year"]]

    train_df = train_p.merge(feats, on=["Region", "Year"], how="left").dropna(subset=feature_cols)
    test_df  = test_p.merge(feats, on=["Region", "Year"], how="left").dropna(subset=feature_cols)

    print(f"Merged rows -> Train: {len(train_df)} | Test: {len(test_df)}")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        max_features=0.6,
        class_weight=CLASS_WEIGHT,
        min_samples_leaf=10,
        max_depth=14
    )

    X_train = train_df[feature_cols].values
    y_train = train_df["Label"].astype(int).values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["Label"].astype(int).values

    print("Training Random Forest...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"\nAccuracy         : {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=2))
    labels = tuple(np.sort(np.unique(y_test)).tolist())
    plot_confusion_matrix_row_percent(
        y_test, y_pred, labels=labels,
        title="Confusion Matrix (Canada)",
        save_path="confusion_matrix_row_percent.png"
    )
    print("Saved: confusion_matrix_row_percent.png")

    if len(labels) == 2:
        pos_class = 1 if 1 in model.classes_ else model.classes_[-1]
        pos_idx = list(model.classes_).index(pos_class)
        y_score = model.predict_proba(X_test)[:, pos_idx]

        plot_roc_like_example(
            y_true=y_test,
            y_score=y_score,
            model_name="Random Forest",
            save_path="roc_like_example.png"
        )
        print("Saved: roc_like_example.png")
    _ = plot_feature_importance(
        model,
        feature_cols,
        top_n=min(20, len(feature_cols)),
        save_path="feature_importance_top20.png"
    )

    out_name = "rf_canada_pathogenimpact_binary_may_aug.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "feature_names_pretty": [make_pretty_feature_name(c) for c in feature_cols],
            "months_used": months,
            "split_year": int(split_year),
            "label_rule": f"PathogenImpact <= {LABEL_THRESHOLD} -> 0, > {LABEL_THRESHOLD} -> 1",
        },
        out_name
    )

if __name__ == "__main__":
    main()


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import joblib

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
PATHOGEN_FILE = r"/USPathogenIncSevData.xlsx"
CLIMATE_FILE  = r"/USMonthyClimateData.csv"

#AUTO_SELECT_MONTHS = False
SPRING_SUSCEPTIBLE_MONTHS = [5,6,7,8]
WINTER_SUSCEPTIBLE_MONTHS = [9,10]

SPRING_WHEAT_STATES = {"ND", "SD", "MN", "MT"}
TEST_FRACTION_YEARS = 0.20

N_ESTIMATORS = 1200
RANDOM_STATE = 42
N_JOBS = -1
CLASS_WEIGHT= "balanced_subsample" #{0: 1.0, 1: 0.9, 2: 1.1} 
#class_weight="balanced_subsample",
#max_features="sqrt",
#min_samples_leaf=5,     
#max_depth=20
def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def incidence_to_class(x):
    if pd.isna(x):
        return np.nan
    if x == 0:
        return 0
    return 1

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
    for col in numeric_cols:
        out[col] = out[col].fillna(out.groupby(["Region", "Month"])[col].transform("mean"))
    for col in numeric_cols:
        out[col] = out[col].fillna(out.groupby("Month")[col].transform("mean"))
    for col in numeric_cols:
        out[col] = out[col].fillna(out[col].mean())
    return out

def _aggregate_features(df):
    """
    Aggregate per (Region, Year) for a subset of months that has
    already been filtered before calling this.
    """
    feats = (
        df
        .groupby(["Region", "Year"], as_index=False)
        .agg({
            "temperature_2m_max": "max",
            "vapour_pressure_deficit_max": "max",
            "soil_moisture_0_to_7cm_mean": "mean",
            "precipitation_sum": "sum",
            "relative_humidity_2m_mean": "mean",
        })
    )
  
    feats = feats.rename(columns={
        "temperature_2m_max": "temp_max_grow",
        "vapour_pressure_deficit_max": "vpd_max_grow",
        "soil_moisture_0_to_7cm_mean": "soil_moist_mean_grow",
        "precipitation_sum": "precip_sum_grow",
        "relative_humidity_2m_mean": "humidity_mean_grow",
    })

    return feats
def make_seasonal_features_by_wheat_type(climate_df):
    """
    Build seasonal features using different month windows for
    spring-wheat vs winter-wheat states.
    """
    keep_cols = [
        "temperature_2m_max",
        "vapour_pressure_deficit_max",
        "soil_moisture_0_to_7cm_mean",
        "precipitation_sum",
        "relative_humidity_2m_mean",
    ]

    keep_cols = [c for c in keep_cols if c in climate_df.columns]

    df = climate_df.copy()
    df = ensure_numeric(df, keep_cols)
    df = hierarchical_impute_monthly(df, keep_cols)
    df["Region"] = df["Region"].astype(str)
    spring_mask = df["Region"].isin(SPRING_WHEAT_STATES)

    spring_df = df[spring_mask & df["Month"].isin(SPRING_SUSCEPTIBLE_MONTHS)].copy()
    winter_df = df[~spring_mask & df["Month"].isin(WINTER_SUSCEPTIBLE_MONTHS)].copy()

    spring_feats = _aggregate_features(spring_df)
    winter_feats = _aggregate_features(winter_df)

    feats = pd.concat([spring_feats, winter_feats], ignore_index=True)

    return feats
def main():
    print("Loading files...")
    pathogen_df = pd.read_excel(PATHOGEN_FILE)
    climate_df  = pd.read_csv(CLIMATE_FILE)

    pathogen_df["Year"] = pd.to_numeric(pathogen_df["Year"], errors="coerce").astype("Int64")
    pathogen_df = ensure_numeric(pathogen_df, ["Incidence"])
    pathogen_df["IncidenceClass"] = pathogen_df["Incidence"].apply(incidence_to_class)
    pathogen_df = pathogen_df.dropna(subset=["IncidenceClass"]).copy()
    train_p, test_p, split_year = time_split_by_year(pathogen_df, "Year", TEST_FRACTION_YEARS)
    print(f"Time split: Train <= {split_year}, Test > {split_year}")
    print(f"Train pathogen rows: {len(train_p)} | Test pathogen rows: {len(test_p)}")

    print(f"Spring wheat states: {sorted(SPRING_WHEAT_STATES)}")
    print(f"Spring-wheat susceptible months: {SPRING_SUSCEPTIBLE_MONTHS}")
    print(f"Winter-wheat susceptible months: {WINTER_SUSCEPTIBLE_MONTHS}")

    feats = make_seasonal_features_by_wheat_type(climate_df)
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
        max_depth=18
        
    )

    X_train = train_df[feature_cols].values
    y_train = train_df["IncidenceClass"].astype(int).values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["IncidenceClass"].astype(int).values

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)


    print(f"Accuracy         : {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")

    print("\nConfusion Matrix (rows=true, cols=pred) [0,1]:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=[0, 1], digits=2))
    
    labels = [0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100


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
        title=("Confusion Matrix (United States)")
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                f"{cm_pct[i, j]:.1f}%",
                ha="center", va="center",
                color="white" if cm_pct[i, j] > 50 else "black",
                fontsize=9
            )

   
    plt.tight_layout()
    plt.show()

    out_name = "rf_US_incidence_bins.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "spring_states":sorted(list(SPRING_WHEAT_STATES)),
            "spring_months": SPRING_SUSCEPTIBLE_MONTHS,
            "winter_months": WINTER_SUSCEPTIBLE_MONTHS,
            "split_year": int(split_year),
            "bins": "(0)->0, (1,2)->1 ",
        },
        out_name
    )
    print(f"\nSaved model to: {out_name}")
if __name__ == "__main__":
    main()


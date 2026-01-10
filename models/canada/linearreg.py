import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

PATHOGEN_FILE = "/content/CanadaPathogenIncSevData.xlsx"
CLIMATE_FILE  = "/content/CanadaMonthyClimateData.csv"

OUTCOME = "PathogenImpact"

p = pd.read_excel(PATHOGEN_FILE)
c = pd.read_csv(CLIMATE_FILE)

p["Region"] = p["Region"].astype(str).str.upper().str.strip()
c["Region"] = c["Region"].astype(str).str.upper().str.strip()

c = c[c["Month"].isin([5, 6, 7, 8])].copy()

predictors = [col for col in c.columns if col not in ["Region", "Year", "Month"]]
c_agg = c.groupby(["Region", "Year"], as_index=False)[predictors].mean()
df = p.merge(c_agg, on=["Region", "Year"], how="inner").dropna(subset=[OUTCOME] + predictors)

scaler = StandardScaler()
X_std = scaler.fit_transform(df[predictors])

X_std = pd.DataFrame(X_std, columns=predictors, index=df.index)

y = df[OUTCOME]
X_const = sm.add_constant(X_std)

model = sm.OLS(y, X_const).fit()

stats_table = pd.DataFrame({
    "Output": ["N (Region–year rows)", "R²", "Adjusted R²"],
    "Value": [int(model.nobs), float(model.rsquared), float(model.rsquared_adj)]
})

print("\nModel Summary Stats:")
print(stats_table.to_string(index=False))
coef_table = pd.DataFrame({
    "Climate variable": predictors,
    "β (coefficient)": [model.params.get(v, float("nan")) for v in predictors],
    "p-value": [model.pvalues.get(v, float("nan")) for v in predictors]
})

coef_table = coef_table.sort_values("p-value")

print("\nCoefficients Table:")
print(coef_table.to_string(index=False))

stats_table.to_csv("OLS_model_stats_table.csv", index=False)
coef_table.to_csv("OLS_coefficients_table.csv", index=False)

LABEL_MAP = {
    "temperature_2m_max": "Max temperature",
    "vapour_pressure_deficit_max": "Vapour pressure deficit",
    "relative_humidity_2m_mean": "Relative humidity",
    "precipitation_sum": "Total precipitation",
    "soil_moisture_0_to_7cm_mean": "Soil moisture"
}

coef = model.params.drop("const")
conf = model.conf_int().loc[coef.index]

plot_df = pd.DataFrame({
    "Variable": coef.index,
    "Beta": coef.values,
    "CI_low": conf[0].values,
    "CI_high": conf[1].values
})

plot_df["Variable"] = plot_df["Variable"].map(LABEL_MAP).fillna(plot_df["Variable"])
plot_df = plot_df.sort_values("Beta")

plt.figure(figsize=(5.5, 3.0))
plt.errorbar(
    plot_df["Beta"],
    plot_df["Variable"],
    xerr=[
        plot_df["Beta"] - plot_df["CI_low"],
        plot_df["CI_high"] - plot_df["Beta"]
    ],
    fmt="o"
)

plt.axvline(0, linestyle="--")
plt.xlabel("Regression Coefficient (β)")
plt.title("Effect of Climate Factors on Pathogen Impact")
plt.tight_layout()
plt.savefig("OLS_coefficient_plot_clean_labels.png", dpi=200)
plt.show()

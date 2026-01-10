import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import re

PATHOGEN_FILE = "/CanadaPathogenIncSevData.xlsx"
CLIMATE_FILE  = "/CanadaMonthyClimateData.csv"

p = pd.read_excel(PATHOGEN_FILE)   
c = pd.read_csv(CLIMATE_FILE)

p["Region"] = p["Region"].astype(str).str.upper().str.strip()
c["Region"] = c["Region"].astype(str).str.upper().str.strip()

c = c[c["Month"].isin([5, 6, 7, 8])].copy()

climate_vars = [col for col in c.columns if col not in ["Region", "Year", "Month"]]
c = c.groupby(["Region", "Year"], as_index=False)[climate_vars].mean()

df = p.merge(c, on=["Region", "Year"], how="inner").dropna()

Y = "PathogenImpact"
X_cols = [col for col in c.columns if col not in ["Region", "Year"]]

rows = []
for x in X_cols:
    if not np.issubdtype(df[x].dtype, np.number):
        continue
    if df[x].nunique() < 2 or df[Y].nunique() < 2:
        continue

    r, pval = pearsonr(df[x], df[Y])
    rows.append([x, r, pval])

corr_table = (
    pd.DataFrame(rows, columns=["Variable", "Pearson_r", "p_value"])
      .sort_values("p_value")
      .reset_index(drop=True)
)

corr_table.to_csv("pearson_correlations.csv", index=False)
print("Saved: pearson_correlations.csv")
print("Original:", corr_table["Variable"].tolist())

def scientific_label(var: str) -> str:
    var = str(var)

    if var == "precipitation_sum":
        return "Total Precipitation"
    if var == "relative_humidity_2m_mean":
        return "Relative Humidity"
    if var == "vapour_pressure_deficit_max":
        return "Vapour Pressure Deficit"
    if var == "soil_moisture_0_to_7cm_mean":
        return "Soil Moisture"
    if var == "temperature_2m_max":
        return "Maximum Temperature"
    s = re.sub(r'[_-]+', ' ', var).strip()
    s = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.title().replace("Vpd", "VPD")
    return s

plot_labels = [scientific_label(v) for v in corr_table["Variable"]]
print("Cleaned :", plot_labels)
plt.figure(figsize=(6, 4))
plt.barh(plot_labels[::-1], corr_table["Pearson_r"].values[::-1])
plt.axvline(0, linewidth=0.8)
plt.xlabel("Pearson correlation (r)", fontsize=11)
plt.tight_layout()
plt.savefig("pearson_r_bar.png", dpi=300)
plt.show()

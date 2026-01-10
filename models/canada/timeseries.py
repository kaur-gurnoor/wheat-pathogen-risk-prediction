import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATHOGEN_FILE = "/content/CanadaPathogenImpactweightedData.csv"
CLIMATE_FILE  = "/content/CanadaClimateMayAugweightedData.csv"

pathogen_df = pd.read_csv(PATHOGEN_FILE)
climate_df  = pd.read_csv(CLIMATE_FILE)

req_p = {"ProvCode", "Province", "Year", "PathogenImpact"}
req_c = {
    "Province", "Year",
    "temperature_2m_max_core_MayAug",
    "vapour_pressure_deficit_max_core_MayAug",
    "soil_moisture_0_to_7cm_mean_core_MayAug"
}

missing_p = req_p - set(pathogen_df.columns)
missing_c = req_c - set(climate_df.columns)

if missing_p:
    raise ValueError(f"Pathogen file missing columns: {missing_p}")
if missing_c:
    raise ValueError(f"Climate file missing columns: {missing_c}")

def clean_upper(s):
    return s.astype(str).str.strip().str.upper()

pathogen_df["ProvCode"] = clean_upper(pathogen_df["ProvCode"])
climate_df["Province"]  = clean_upper(climate_df["Province"])

pathogen_df["Year"] = pd.to_numeric(pathogen_df["Year"], errors="coerce").astype("Int64")
climate_df["Year"]  = pd.to_numeric(climate_df["Year"], errors="coerce").astype("Int64")

pathogen_df["PathogenImpact"] = pd.to_numeric(pathogen_df["PathogenImpact"], errors="coerce")
data = pathogen_df.merge(
    climate_df,
    left_on=["ProvCode", "Year"],
    right_on=["Province", "Year"],
    how="inner",
    suffixes=("", "_clim")
)
data = data[data["ProvCode"].isin(["AB", "MB", "SK"])].copy()

data["ProvinceName"] = data["Province_y"] if "Province_y" in data.columns else data.get("Province", data["ProvCode"])

data = data.dropna(subset=[
    "Year", "ProvCode", "PathogenImpact",
    "temperature_2m_max_core_MayAug",
    "vapour_pressure_deficit_max_core_MayAug",
    "soil_moisture_0_to_7cm_mean_core_MayAug"
]).copy()

data["MoistureStress_vpd_div_soilM_core_MayAug"] = (
    data["vapour_pressure_deficit_max_core_MayAug"] /
    (data["soil_moisture_0_to_7cm_mean_core_MayAug"].replace(0, np.nan))
)
data.to_csv("province_year_weighted_merged_for_timeseries_and_regime.csv", index=False)
print("Saved: province_year_weighted_merged_for_timeseries_and_regime.csv")
print("Rows:", len(data))

data_sorted = data.sort_values(["ProvCode", "Year"]).copy()

fig, ax = plt.subplots(figsize=(11, 6))
for code in ["AB", "MB", "SK"]:
    g = data_sorted[data_sorted["ProvCode"] == code]
    if len(g) == 0:
        continue
    ax.plot(g["Year"], g["PathogenImpact"], marker="o", label=code)

ax.set_xlabel("Year")
ax.set_ylabel("Pathogen Impact")
ax.set_title("CWRS Fusarium Pathogen Impact Over Time")
ax.grid(True, alpha=0.2)
ax.legend()
plt.tight_layout()
plt.savefig("cwrs_pathogenimpact_timeseries.png", dpi=250)
plt.show()
print("Saved: cwrs_pathogenimpact_timeseries.png")
def fig_moisture_stress_vs_temp(df):
    plt.figure(figsize=(7, 4))

    for code in sorted(df["ProvCode"].unique()):
        g = df[df["ProvCode"] == code].dropna(subset=[
            "temperature_2m_max_core_MayAug",
            "MoistureStress_vpd_div_soilM_core_MayAug",
            "PathogenImpact"
        ]).copy()

        if len(g) == 0:
            continue

        z = g["PathogenImpact"].astype(float)
        z_min, z_max = z.min(), z.max()
        sizes = 20 + 200 * (z - z_min) / (z_max - z_min + 1e-9)

        prov_label = g["ProvinceName"].iloc[0] if "ProvinceName" in g.columns else code

        plt.scatter(
            g["temperature_2m_max_core_MayAug"],
            g["MoistureStress_vpd_div_soilM_core_MayAug"],
            s=sizes,
            alpha=0.6,
            label=prov_label
        )

    plt.xlabel("Mean Temperature (°C)")
    plt.ylabel("Moisture Stress (↑ drier, ↓ wetter)")
    plt.title("Moisture Stress vs Temp for Pathogen Impact")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("moisture_stress_vs_temp_regime.png", dpi=250)
    plt.show()
    print("Saved: moisture_stress_vs_temp_regime.png")

fig_moisture_stress_vs_temp(data_sorted)

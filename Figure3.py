from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(
    "d:/IstinyeTesticularCancer/"
)

input_file = (
    BASE_DIR / "stage3_output"
    / "walk_forward_validation_predictions.xlsx"
)
output_file = BASE_DIR / "Figure3/figure3_walk_forward_validation.png"

df = pd.read_excel(input_file).sort_values("Year").reset_index(drop=True)

models = [
    "ETS",
    "Drift",
    "AutoARIMA",
    "Naive",
    "XGBoost",
    "Prophet",
    "Linear Regression",
    "Random Forest",
]

required = ["Year", "Actual"] + models
missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Missing columns in validation file: {missing}")

if df[required].isna().any().any():
    raise ValueError("Validation data contain missing values.")

# Compare validation observations with the current annual dataset.
annual_file = (
    BASE_DIR / "stage2_output"
    / "forecast_ready_dataset.xlsx"
)

annual = pd.read_excel(annual_file)

annual_required = ["Year", "HospitalBasedBurden"]
missing_annual = [
    column for column in annual_required
    if column not in annual.columns
]

if missing_annual:
    raise ValueError(
        f"Missing columns in annual dataset: {missing_annual}"
    )

annual = annual[annual_required].copy()

for column in annual_required:
    annual[column] = pd.to_numeric(annual[column], errors="raise")

for column in ["Year", "Actual"]:
    df[column] = pd.to_numeric(df[column], errors="raise")

if annual.isna().any().any():
    raise ValueError("Annual dataset contains missing values.")

if annual["Year"].duplicated().any() or df["Year"].duplicated().any():
    raise ValueError("Duplicate years found in the input files.")

annual = annual.sort_values("Year").reset_index(drop=True)
df = df.sort_values("Year").reset_index(drop=True)

print("Okunan dosya:", annual_file.resolve())
print("Dosyadaki yıllar:", annual["Year"].tolist())
print("Yıllık gözlem sayısı:", len(annual))

# Current analysis period: 2010–2024.
if annual["Year"].tolist() != list(range(2010, 2025)):
    raise ValueError(
        "The annual dataset must contain every year from 2010 to 2024. "
        "Regenerate Stage 1 and Stage 2 using the revised analysis period."
    )

# Primary validation uses the last five annual observations.
expected = annual.tail(5)
expected_years = expected["Year"].astype(int).tolist()
expected_actual = expected["HospitalBasedBurden"].tolist()

if df["Year"].tolist() != expected_years:
    raise ValueError(
        f"Validation years must be {expected_years}. "
        "Regenerate Stage 3 using the current annual dataset."
    )

comparison = pd.DataFrame({
    "Year": expected_years,
    "ValidationActual": df["Actual"].tolist(),
    "CurrentAnnualCount": expected_actual,
})

mismatch = (
    comparison["ValidationActual"]
    != comparison["CurrentAnnualCount"]
)

if mismatch.any():
    raise ValueError(
        "Validation observations differ from the current annual dataset:\n"
        + comparison.loc[mismatch].to_string(index=False)
        + "\nRegenerate Stage 3 using the current annual dataset."
    )

styles = {
    "ETS": ("D", "#D55E00", "--"),
    "Drift": ("s", "#009E73", "--"),
    "AutoARIMA": ("^", "#0072B2", "--"),
    "Naive": ("v", "#666666", ":"),
    "XGBoost": ("X", "#CC79A7", "--"),
    "Prophet": ("P", "#E69F00", "--"),
    "Linear Regression": ("+", "#56B4E9", ":"),
    "Random Forest": ("x", "#6A3D9A", "--"),
}

fig, ax = plt.subplots(figsize=(11, 6.5))

ax.plot(
    df["Year"],
    df["Actual"],
    color="black",
    marker="o",
    linewidth=2.8,
    markersize=7,
    label="Observed",
    zorder=5,
)

for model in models:
    marker, color, linestyle = styles[model]
    ax.plot(
        df["Year"],
        df[model],
        marker=marker,
        color=color,
        linestyle=linestyle,
        linewidth=1.7,
        markersize=6,
        label=model,
    )

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Hospital-observed patient count", fontsize=12)
ax.set_xticks(expected_years)
ax.ticklabel_format(axis="x", style="plain", useOffset=False)
ax.grid(True, alpha=0.25)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.18),
    ncol=3,
    fontsize=9,
    frameon=False,
)

fig.tight_layout()
fig.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Figure 3 saved: {output_file}")

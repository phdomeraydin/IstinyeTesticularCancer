import pandas as pd
import numpy as np

# ================================
# 1. READ DATA
# ================================
input_file = "d:/IstinyeTesticularCancer/data/testicular.csv"
output_file = "d:/IstinyeTesticularCancer/stage1_output/dataset.xlsx"

# Read CSV correctly
# First row contains column names.
df = pd.read_csv(
    input_file,
    sep=",",
    header=0,
    encoding="utf-8-sig"
)
df.columns = df.columns.str.strip()

required_columns = [
    "Gender",
    "Year of Diagnosis",
    "Age",
    "Survival Month",
    "Exitus Status",
    "Exitus Date",
] + [
    f"{year} Hospital Application"
    for year in range(2005, 2025)
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if len(df) != 5153:
    raise ValueError(
        f"Expected 5153 patient records, but found {len(df)}. "
        "Check the input CSV before continuing."
    )

gender = df["Gender"].astype("string").str.strip().str.casefold()

female_mask = gender.isin(["female", "kadın", "kadin", "f"])
if female_mask.any():
    raise ValueError(
        f"{int(female_mask.sum())} female-coded records remain. "
        "Check that the updated CSV is being read."
    )

male_mask = gender.isin(["male", "erkek", "m"])
if not male_mask.all():
    unexpected = (
        df.loc[~male_mask, "Gender"]
        .astype("string")
        .fillna("<missing>")
        .unique()
        .tolist()
    )
    raise ValueError(
        f"Missing or unrecognized Gender values: {unexpected}. "
        "Verify their meaning before continuing."
    )

print(f"Input validated: {len(df)} male-coded patient records.")

# ================================
# >>> NEW: DATA QUALITY CHECKS <<<
# ================================
print("="*50)
print("DATA QUALITY REPORT")
print("="*50)
print(f"Total patients: {len(df)}")
print(f"\nColumn names found: {len(df.columns)} columns")
print(f"\nFirst 5 column names:\n{list(df.columns[:5])}")
print(f"\nDiagnosis year range: {df['Year of Diagnosis'].min()} - {df['Year of Diagnosis'].max()}")
print(f"\nMissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("="*50)

# ================================
# 2. STANDARDIZE BINARY VARIABLES
# ================================

# Automatically capture Hospital Application columns
hospital_cols = [col for col in df.columns if "Hospital Application" in col]

# >>> NEW: Print detected hospital columns <<<
print(f"\nDetected {len(hospital_cols)} hospital application columns:")
print(hospital_cols)

binary_cols = hospital_cols + [
    "Exitus Status",
    "Is There Heart Failure?",
    "Is There Heart Failure After Diagnosis?",
    "Is There Ischemic Heart Disease After Diagnosis?",
    "Is There Chronic Kidney Failure?",
    "Is There Chronic Kidney Failure After Diagnosis?",
    "Is There Cerebral Infarction?",
    "Is There Cerebral Infarction After Diagnosis?",
    "Is There Diabetes?",
    "Is There Blood Pressure?",
    "Is There COPD?",
    "Is There Asthma?",
    "Is There a Stroke After Diagnosis?",
    "Is There a Stroke or Cerebral Event After Diagnosis?"
]

binary_mapping = {
    "yes": 1, "no": 0,
    "1": 1, "0": 0,
    "1.0": 1, "0.0": 0,
    "true": 1, "false": 0
}
        
binary_issues = []

for col in binary_cols:
    if col not in df.columns:
        continue

    raw = df[col]
    normalized = raw.astype("string").str.strip().str.lower()
    converted = normalized.map(binary_mapping)

    invalid = converted.isna()

    if invalid.any():
        examples = raw.loc[invalid].astype(str).unique()[:5]
        binary_issues.append(
            f"{col}: {int(invalid.sum())} missing/unrecognized values; "
            f"examples={list(examples)}"
        )

    df[col] = converted

if binary_issues:
    raise ValueError(
        "Binary data require review before analysis:\n"
        + "\n".join(binary_issues)
    )

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)
# ================================
# 3. DATES AND DEFINITIONS
# =================================

df["Diagnosis month-year"] = pd.to_datetime(
    df["Diagnosis month-year"], errors="coerce"
)

df["Year_of_Diagnosis"] = df["Year of Diagnosis"].astype(int)
# ================================
# EXTRACT DEATH YEAR
# ================================

df["Exitus Date"] = pd.to_datetime(
    df["Exitus Date"],
    errors="coerce"
)

df["DeathYear"] = df["Exitus Date"].dt.year
# ================================
# 4. INCIDENT CASE (DIAGNOSIS DENSITY)
# ================================

diagnosis_density = (
    df.groupby("Year_of_Diagnosis")
      .size()
      .rename("DiagnosisCount")
      .reset_index()
      .rename(columns={"Year_of_Diagnosis": "Year"})
)

# ================================
# 5. HOSPITAL-BASED PREVALENCE + TOTAL APPLICATION
# ================================

# >>> CHANGED: Auto-detect years from hospital columns instead of hardcoding <<<
years = sorted([int(col.split()[0]) for col in hospital_cols])

WINDOW_YEARS = 5
DIAGNOSIS_HISTORY_START = 2006
ANALYSIS_END_YEAR = 2024

first_valid_year = DIAGNOSIS_HISTORY_START + WINDOW_YEARS - 1

required_years = list(range(first_valid_year, ANALYSIS_END_YEAR + 1))
missing_years = [year for year in required_years if year not in years]

if missing_years:
    raise ValueError(
        f"Missing hospital application columns for years: {missing_years}"
    )

print(f"\nYears to process: {years}")
print(f"Year range: {min(years)} to {max(years)}")

records = []

for year in years:
    app_col = f"{year} Hospital Application"
    
    if app_col not in df.columns:
        print(f"WARNING: Missing column {app_col}")
        continue
    
    active = df[df[app_col] == 1]
    
    prevalence = active[
        active["Year_of_Diagnosis"].between(year - WINDOW_YEARS + 1, year)
    ].shape[0]
    
    total_apps = df[app_col].sum()
    
    records.append({
        "Year": year,
        "HospitalBasedBurden": prevalence,
        "TotalApplications": total_apps
    })

yearly_base = pd.DataFrame(records)
annual_deaths = (
    df.loc[
        df["Exitus Status"].eq(1) & df["DeathYear"].notna()
    ]
    .groupby("DeathYear")
    .size()
)

yearly_base["AnnualDeaths"] = (
    yearly_base["Year"]
    .map(annual_deaths)
    .fillna(0)
    .astype(int)
)
# ================================
# 6. DIAGNOSIS DENSITY COMBINE
# =================================

yearly = yearly_base.merge(
    diagnosis_density,
    on="Year",
    how="left"
)

yearly["DiagnosisCount"] = yearly["DiagnosisCount"].fillna(0).astype(int)

# ================================
# 7. NORMALIZED DIAGNOSIS RATE
# ================================

yearly["Diagnosis-to-Application-Ratio"] = (
    yearly["DiagnosisCount"] /
    yearly["TotalApplications"].replace(0, np.nan)
)

# ================================
# 8. DEMOGRAPHY + COMORBIDITY AGGREGATES
# ================================

agg_records = []

for year in years:
    app_col = f"{year} Hospital Application"
    if app_col not in df.columns:
        continue
    
    cohort = df[
        (df[app_col] == 1) &
        (df["Year_of_Diagnosis"].between(year - WINDOW_YEARS + 1, year))
    ]
    
    if cohort.empty:
        continue
    
    agg_records.append({
        "Year": year,
        "MeanAge": cohort["Age"].mean(),
        "COPD_Rate": cohort["Is There COPD?"].mean() if "Is There COPD?" in df.columns else np.nan,
        "HT_Rate": cohort["Is There Blood Pressure?"].mean() if "Is There Blood Pressure?" in df.columns else np.nan,
        "DM_Rate": cohort["Is There Diabetes?"].mean() if "Is There Diabetes?" in df.columns else np.nan,
        "CHF_Rate": cohort["Is There Heart Failure?"].mean() if "Is There Heart Failure?" in df.columns else np.nan,
        "CKF_Rate": cohort["Is There Chronic Kidney Failure?"].mean() if "Is There Chronic Kidney Failure?" in df.columns else np.nan,
        "MeanSurvivalMonth": cohort["Survival Month"].mean() if "Survival Month" in df.columns else np.nan,
#        "TotalDeaths": cohort["Exitus Status"].sum() if "Exitus Status" in df.columns else np.nan,
#        "AnnualDeaths": df[(df["Exitus Status"] == 1) & (df["DeathYear"] == year)].shape[0],
        "MeanEmergencyApplications":cohort["Total Emergency Applications"].mean() if "Total Emergency Applications" in df.columns else np.nan,
        "MeanMedicalOncologyApplications":cohort["Total Medical Oncology Applications"].mean() if "Total Medical Oncology Applications" in df.columns else np.nan,
        "MeanRadiationApplications":cohort["Total Radiation Oncology Applications"].mean() if "Total Radiation Oncology Applications" in df.columns else np.nan,
        "MeanBloodCount":cohort["Post-Diagnosis Blood Count Number"].mean() if "Post-Diagnosis Blood Count Number" in df.columns else np.nan,
        "MeanApplicationsAfterDiagnosis":cohort["Total All Applications After Diagnosis"].mean() if "Total All Applications After Diagnosis" in df.columns else np.nan
    })

yearly_agg = pd.DataFrame(agg_records)

# ================================
# 9. CREATE TIME-SERIES DATASET
# ================================

ts_data = yearly.merge(
    yearly_agg,
    on="Year",
    how="left"
).sort_values("Year").reset_index(drop=True)

# ================================
# 10. LAG FEATURES
# ================================

lag_features = [
    "HospitalBasedBurden",
    "DiagnosisCount",
    "TotalApplications",
    "Diagnosis-to-Application-Ratio"
]

for col in lag_features:
    ts_data[f"{col}_lag1"] = ts_data[col].shift(1)
    ts_data[f"{col}_lag2"] = ts_data[col].shift(2)
    ts_data[f"{col}_lag3"] = ts_data[col].shift(3)

rolling_features = [
    "HospitalBasedBurden",
    "DiagnosisCount"
]

# Rolling Mean
for col in rolling_features:
    ts_data[f"{col}_RollingMean3"] = (
        ts_data[col]
        .rolling(window=3)
        .mean()
    )

# Total Applications Rolling Mean
ts_data["TotalApplications_RollingMean3"] = (
    ts_data["TotalApplications"]
    .rolling(window=3)
    .mean()
)

# Rolling Standard Deviation

#ts_data["DiagnosisCount_RollingStd3"] = (
#    ts_data["DiagnosisCount"]
#    .rolling(window=3)
#    .std()
#)

#ts_data["HospitalBasedBurden_RollingStd3"] = (
#    ts_data["HospitalBasedBurden"]
#    .rolling(window=3)
#    .std()
#)

# Growth Rates
ts_data["DiagnosisGrowthRate"] = (
    ts_data["DiagnosisCount"]
    .pct_change()
    .replace([np.inf, -np.inf], np.nan)
)

ts_data["BurdenGrowthRate"] = (
    ts_data["HospitalBasedBurden"]
    .pct_change()
    .replace([np.inf, -np.inf], np.nan)
)

# Year Index
ts_data["YearIndex"] = (
    ts_data["Year"] - ts_data["Year"].min()
)

ts_data_final = (
    ts_data.loc[
        ts_data["Year"].between(first_valid_year, ANALYSIS_END_YEAR)
    ]
    .sort_values("Year")
    .reset_index(drop=True)
)

expected_years = list(range(first_valid_year, ANALYSIS_END_YEAR + 1))

if ts_data_final["Year"].tolist() != expected_years:
    raise ValueError(
        "The final annual dataset has missing or duplicate years. "
        f"Expected: {expected_years}; "
        f"Found: {ts_data_final['Year'].tolist()}"
    )

print(
    f"\nFirst five-year diagnosis window: "
    f"{DIAGNOSIS_HISTORY_START}-{first_valid_year}"
)
print(f"Final dataset years: {first_valid_year}-{ANALYSIS_END_YEAR}")
print(f"Annual observations: {len(ts_data_final)}")

# >>> NEW: Final dataset summary <<<
print("\n" + "="*50)
print("FINAL DATASET SUMMARY")
print("="*50)
print(f"Total years in final dataset: {len(ts_data_final)}")
print(f"Year range: {ts_data_final['Year'].min()} - {ts_data_final['Year'].max()}")
print(f"\nColumns in final dataset: {len(ts_data_final.columns)}")
print(f"\nFirst 5 rows:")
print(ts_data_final.head())
print("\nDescriptive statistics:")
print("\nNaN values in final dataset:")
print(ts_data_final.isna().sum())

print("\nInfinite values:")
print(np.isinf(ts_data_final.select_dtypes(include=np.number)).sum())
ts_data_final = ts_data_final.replace(
    [np.inf, -np.inf],
    np.nan
)
print(ts_data_final.describe())
#print("\nCorrelation Matrix")
#print(ts_data_final.corr(numeric_only=True))
print("="*50)

# ================================
# 11. SAVE AS EXCEL FILE
# ================================

ts_data_final.to_excel(output_file, index=False)

print(f"\n✓ Successfully created dataset.xlsx!")
print(f"✓ File saved to: {output_file}")

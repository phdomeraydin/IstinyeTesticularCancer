# ================================
# 1. READ THE DATA AND INITIAL CHECK
# ================================

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymannkendall as mk
import numpy as np
# >>> NEW: Import for correlation heatmap <<<
import seaborn as sns
# >>> NEW: Import for stationarity test <<<
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

warnings.filterwarnings(
    "ignore",
    category=InterpolationWarning
)
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


# ================================
# test_kpss
# ================================
def test_kpss(series, name):
    statistic, p_value, lags, critical = kpss(
        series.dropna(),
        regression="c",
        nlags="auto"
    )

    print(f"\n{name}")
    print(f"KPSS Statistic : {statistic:.4f}")
    print(f"P-value        : {p_value:.4f}")

    if p_value > 0.05:
        print("Stationary")
    else:
        print("Non-stationary")
# ================================
# test_kpss
# ================================     
      
file_path = "d:/IstinyeTesticularCancer/stage1_output/dataset.xlsx"
df = pd.read_excel(file_path)
# Stage 1 writes the ratio with a hyphen before "Ratio". Normalize its
# name, including lag columns, for the analyses below.
df = df.rename(columns={
    column: column.replace("Diagnosis-to-Application-Ratio", "Diagnosis-to-Application Ratio")
    for column in df.columns
})
if "Diagnosis-to-Application Ratio" not in df.columns:
    raise ValueError("Missing diagnosis-to-application ratio in the Stage 1 dataset.")

print("First 5 rows of dataset:")
print(df.head())

# ================================
# 2. BASIC DEFINITIVE TABLE
# ================================

summary_table = df.describe()
print("\nDescriptive Statistics:")
print(summary_table)

# >>> NEW: Save summary table to Excel <<<
summary_table.to_excel("d:/IstinyeTesticularCancer/stage2_output/summary_statistics.xlsx")
print("\n✓ Summary statistics saved to summary_statistics.xlsx")

# ================================
# >>> NEW: CORRELATION ANALYSIS <<<
# ================================

print("\n" + "="*50)
print("CORRELATION ANALYSIS")
print("="*50)

key_vars = ['HospitalBasedBurden','DiagnosisCount','TotalApplications',
            'Diagnosis-to-Application Ratio','MeanAge','DM_Rate','HT_Rate','COPD_Rate',
            'CHF_Rate','CKF_Rate','MeanSurvivalMonth','AnnualDeaths','MeanEmergencyApplications',
            'MeanMedicalOncologyApplications','MeanRadiationApplications','MeanBloodCount','MeanApplicationsAfterDiagnosis']
            
# Filter only existing columns
existing_vars = [var for var in key_vars if var in df.columns]

correlation_matrix = df[existing_vars].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Save correlation matrix
correlation_matrix.to_excel("d:/IstinyeTesticularCancer/stage2_output/correlation_matrix.xlsx")

# Correlation heatmap
plt.figure(figsize=(12, 10))
#sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=1)
#18 değişkende rakamlar üst üste gelir diye aşağıdaki (annot=False) gibi değiştirdim.
sns.heatmap(correlation_matrix, annot=False, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=1)
plt.title("Correlation Matrix of Key Variables", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("d:/IstinyeTesticularCancer/stage2_output/correlation_heatmap.png", 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Correlation heatmap saved")

# ================================
# >>> NEW: STATIONARITY TEST <<<
# ================================

print("\n" + "="*50)
print("STATIONARITY TEST (Augmented Dickey-Fuller)")
print("="*50)

def test_stationarity(series, name):
    """Perform ADF test and print results"""
    result = adfuller(series.dropna(), autolag='AIC')
    print(f'\n{name}:')
    print(f'  ADF Statistic: {result[0]:.4f}')
    print(f'  p-value: {result[1]:.4f}')
    print(f'  Critical Values:')
    for key, value in result[4].items():
        print(f'    {key}: {value:.4f}')
    
    if result[1] <= 0.05:
        print(f'  → Result: STATIONARY (reject null hypothesis)')
    else:
        print(f'  → Result: NON-STATIONARY (fail to reject null hypothesis)')
    
    return result

# Test key time series
stationarity_results = {}

test_vars = [
    "HospitalBasedBurden",
    "DiagnosisCount",
    "Diagnosis-to-Application Ratio",
    "MeanSurvivalMonth"
]

for var in test_vars:
    if var in df.columns:
        print("\n" + "="*60)
        print(var)
        print("="*60)
        adf_result = test_stationarity(df[var], var)
        test_kpss(df[var], var)
        stationarity_results[var] = adf_result
# ================================
# ACF & PACF
# ================================

for var in ["DiagnosisCount", "HospitalBasedBurden"]:
    if var in df.columns:
        plt.figure(figsize=(12,4))
#        plot_acf(df[var].dropna(), ax=plt.gca(), lags=min(10, len(df)//2))
        plot_acf(df[var].dropna(), ax=plt.gca(), lags=8) #daha okunaklı grafik olsun diye
        plt.title(f"ACF - {var}")
        plt.tight_layout()
        plt.savefig(f"d:/IstinyeTesticularCancer/stage2_output/{var}_ACF.png",dpi=300)
        plt.close()
        plt.figure(figsize=(12,4))
        plot_pacf(df[var].dropna(),ax=plt.gca(),lags=min(10, len(df)//2),method="ywm")
        plt.title(f"PACF - {var}")
        plt.tight_layout()
        plt.savefig(f"d:/IstinyeTesticularCancer/stage2_output/{var}_PACF.png",dpi=300)
        plt.close()    
# ================================
# ACF & PACF
# ================================  
####SUMMARY###############################
stationarity_table = []
for var in test_vars:
    if var in df.columns:
        adf = adfuller(df[var].dropna(), autolag="AIC")
        kp = kpss(df[var].dropna(), regression="c", nlags="auto")
        stationarity_table.append({
            "Variable": var,
            "ADF Statistic": adf[0],
            "ADF p-value": adf[1],
            "KPSS Statistic": kp[0],
            "KPSS p-value": kp[1]
        })

stationarity_df = pd.DataFrame(stationarity_table)
stationarity_df.to_excel("d:/IstinyeTesticularCancer/stage2_output/stationarity_results.xlsx",index=False)

# ================================
# 3. GRAPHS
# Each graph will be separate and simple.
# ================================

# >>> CHANGED: Save figures instead of just showing them <<<
# >>> CHANGED: Add grid and improve styling <<<

# 3.1 Hospital-Based Burden Trend
plt.figure(figsize=(12, 6))
plt.plot(df["Year"], df["HospitalBasedBurden"], marker='o', linewidth=2, markersize=6)
start_year = int(df["Year"].min())
end_year = int(df["Year"].max())
plt.title(f"Hospital-Based Testicular Cancer Burden ({start_year}–{end_year})", fontsize=14, fontweight='bold')
#plt.title("Hospital-Based Testicular Cancer Burden (2002–2025)", fontsize=14, fontweight='bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Hospital-Based Burden", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
# >>> NEW: Save figure <<<
plt.savefig("d:/IstinyeTesticularCancer/stage2_output/fig1_prevalence_trend.png", 
            dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Figure 1 saved: Burden trend")

# 3.2 Diagnosis Count Trend
plt.figure(figsize=(12, 6))
plt.plot(df["Year"], df["DiagnosisCount"], marker='s', linewidth=2, markersize=6, color='green')
plt.title("Annual Testicular Cancer Diagnosis Count", fontsize=14, fontweight='bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Diagnosis Count", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("d:/IstinyeTesticularCancer/stage2_output/fig2_diagnosis_count.png", 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 2 saved: Diagnosis count")

# 3.3 Total Hospital Applications Trend
plt.figure(figsize=(12, 6))
plt.plot(df["Year"], df["TotalApplications"], marker='^', linewidth=2, markersize=6, color='orange')
plt.title("Total Hospital Applications per Year", fontsize=14, fontweight='bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Total Applications", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("d:/IstinyeTesticularCancer/stage2_output/fig3_total_applications.png", 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 3 saved: Total applications")

# 3.4 Diagnosis-to-Application Ratio Trend
plt.figure(figsize=(12, 6))
plt.plot(df["Year"], df["Diagnosis-to-Application Ratio"], marker='D', linewidth=2, markersize=6, color='red')
plt.title("Diagnosis-to-Application Ratio Over Time", fontsize=14, fontweight='bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Diagnosis-to-Application Ratio", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("d:/IstinyeTesticularCancer/stage2_output/fig4_normalized_rate.png", 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 4 saved: Diagnosis-to-Application Ratio")

# ================================
# >>> NEW: COMBINED TREND VISUALIZATION <<<
# ================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Burden
axes[0, 0].plot(df["Year"], df["HospitalBasedBurden"], marker='o', linewidth=2)
axes[0, 0].set_title("Hospital-Observed Patient Count", fontweight='bold')
axes[0, 0].set_xlabel("Year")
axes[0, 0].set_ylabel("Patients")
axes[0, 0].grid(True, alpha=0.3)

# Diagnosis Count
axes[0, 1].plot(df["Year"], df["DiagnosisCount"], marker='s', linewidth=2, color='green')
axes[0, 1].set_title("Diagnosis Count", fontweight='bold')
axes[0, 1].set_xlabel("Year")
axes[0, 1].set_ylabel("Count")
axes[0, 1].grid(True, alpha=0.3)

# Total Applications
axes[1, 0].plot(df["Year"], df["TotalApplications"], marker='^', linewidth=2, color='orange')
axes[1, 0].set_title("Patients With a Recorded Hospital Application", fontweight='bold')
axes[1, 0].set_xlabel("Year")
axes[1, 0].set_ylabel("Patients")
axes[1, 0].grid(True, alpha=0.3)

# Normalized Rate
axes[1, 1].plot(df["Year"], df["Diagnosis-to-Application Ratio"], marker='D', linewidth=2, color='red')
axes[1, 1].set_title("Diagnosis-to-Application Ratio", fontweight='bold')
axes[1, 1].set_xlabel("Year")
axes[1, 1].set_ylabel("Ratio")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("d:/IstinyeTesticularCancer/stage2_output/fig5_combined_trends.png",dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 5 saved: Combined trends")

# ================================
# >>> NEW: COMORBIDITY TRENDS <<<
# ================================

comorbidity_vars = ['COPD_Rate', 'HT_Rate', 'DM_Rate', 'CHF_Rate', 'CKF_Rate']
existing_comorbidities = [var for var in comorbidity_vars if var in df.columns and df[var].notna().any()]

if existing_comorbidities:
    plt.figure(figsize=(14, 7))
    for var in existing_comorbidities:
        plt.plot(df["Year"], df[var], marker='o', linewidth=2, label=var.replace('_Rate', ''))
    
    plt.title("Comorbidity Trends Over Time", fontsize=14, fontweight='bold')
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Rate (Proportion)", fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("d:/IstinyeTesticularCancer/stage2_output/fig6_comorbidity_trends.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 6 saved: Comorbidity trends")
# ================================
# HEALTHCARE UTILIZATION
# ================================
utilization = ["MeanEmergencyApplications","MeanMedicalOncologyApplications","MeanRadiationApplications","MeanBloodCount","MeanApplicationsAfterDiagnosis"]

existing = [
c for c in utilization
if c in df.columns
]
plt.figure(figsize=(14,7))
for col in existing:
    plt.plot(df["Year"], df[col], marker="o", linewidth=2, label=col.replace("Mean",""))
#    plt.plot(df["Year"], df[col], marker="o", linewidth=2, label=col)
plt.legend()
plt.grid(True)
plt.xlabel("Year")
plt.ylabel("Mean Value")
plt.title("Healthcare Utilization Over Time")
plt.tight_layout()
plt.savefig("d:/IstinyeTesticularCancer/stage2_output/fig7_healthcare_utilization.png",dpi=300)
plt.close()    
#############boxplot################### 
plt.figure(figsize=(10,5))
plt.boxplot(df["DiagnosisCount"])
plt.title("Diagnosis Count Distribution")
plt.savefig("d:/IstinyeTesticularCancer/stage2_output/diagnosis_boxplot.png",dpi=300)
plt.close()
##########Histogram####################
plt.figure(figsize=(8,5))
#plt.hist(df["DiagnosisCount"], bins=10)
plt.hist(df["DiagnosisCount"], bins=6) #22 gözlem için daha uygun!
plt.xlabel("Diagnosis Count")
plt.ylabel("Frequency")
plt.title("Diagnosis Count Histogram")
plt.tight_layout()
plt.savefig("d:/IstinyeTesticularCancer/stage2_output/diagnosis_histogram.png",dpi=300)
plt.close()
#############Mann-Kendall###################   
print("\n" + "="*60)
print("MANN-KENDALL TREND TEST")
print("="*60)

trend_results = {}
trend_table = []

for var in [
    "DiagnosisCount",
    "HospitalBasedBurden",
    "Diagnosis-to-Application Ratio"
]:

    if var in df.columns:

        result = mk.original_test(df[var])

        trend_results[var] = result

        print(f"\n{var}")
        print(result)

        trend_table.append({
            "Variable": var,
            "Trend": result.trend,
            "P-value": result.p,
            "Tau": result.Tau,
            "Slope": result.slope
        })

trend_df = pd.DataFrame(trend_table)

trend_df.to_excel("d:/IstinyeTesticularCancer/stage2_output/mann_kendall_results.xlsx",index=False)       
#############Mann-Kendall###################
# ================================
# CREATE FORECAST-READY DATASET
# ================================

forecast_columns = [
    "Year",
    "YearIndex",
    "HospitalBasedBurden",
    "DiagnosisCount",
    "TotalApplications",
    "Diagnosis-to-Application Ratio",
    "MeanAge",
    "COPD_Rate",
    "HT_Rate",
    "DM_Rate",
    "CHF_Rate",
    "CKF_Rate",
    "MeanSurvivalMonth",
    "AnnualDeaths",
    "MeanEmergencyApplications",
    "MeanMedicalOncologyApplications",
    "MeanRadiationApplications",
    "MeanBloodCount",
    "MeanApplicationsAfterDiagnosis",
    "HospitalBasedBurden_lag1",
    "HospitalBasedBurden_lag2",
    "HospitalBasedBurden_lag3",
    "DiagnosisCount_lag1",
    "DiagnosisCount_lag2",
    "DiagnosisCount_lag3",
    "TotalApplications_lag1",
    "TotalApplications_lag2",
    "TotalApplications_lag3",
    "Diagnosis-to-Application Ratio_lag1",
    "Diagnosis-to-Application Ratio_lag2",
    "Diagnosis-to-Application Ratio_lag3",
    "HospitalBasedBurden_RollingMean3",
    "DiagnosisCount_RollingMean3",
    "TotalApplications_RollingMean3",
    "DiagnosisGrowthRate",
    "BurdenGrowthRate"
]

forecast_columns = [c for c in forecast_columns if c in df.columns]
forecast_df = (
    df.sort_values("Year")
    .loc[:, forecast_columns]
    .reset_index(drop=True)
    .copy()
)

forecast_df.to_excel(
    "d:/IstinyeTesticularCancer/stage2_output/forecast_ready_dataset.xlsx",
    index=False
)

print("✓ forecast_ready_dataset.xlsx created")
##########################################################
print("\n" + "="*50)
print("Stage 2 Analysis Completed Successfully!")
print("="*50)
print("\nGenerated files:")
print("\nGenerated Excel Files")

print("----------------------")

print("  - summary_statistics.xlsx")
print("  - correlation_matrix.xlsx")
print("  - stationarity_results.xlsx")
print("  - mann_kendall_results.xlsx")
print("  - correlation_heatmap.png")
print("  - fig1_prevalence_trend.png")
print("  - fig2_diagnosis_count.png")
print("  - fig3_total_applications.png")
print("  - fig4_normalized_rate.png")
print("  - fig5_combined_trends.png")
print("  - fig6_comorbidity_trends.png")

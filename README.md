# Istinye Testicular Cancer Forecasting Study

This repository contains the Python analysis code for *Forecasting Annual Hospital-Observed Testicular Cancer Patient Counts Using Multicenter Real-World Data*.

The study uses records from 33 hospitals in Türkiye to construct an annual series for 2010–2024. Each count includes patients with a recorded hospital visit in that year who were diagnosed with testicular cancer in the same year or the preceding four calendar years. The series measures patient volume within the participating network; it does not estimate population incidence or prevalence.

## Data access

The patient-level dataset is not included in this repository. Request access to the de-identified data through the [Istinye University Dataset Sharing Platform](https://dataset.istinye.edu.tr/dataset?did=24). If access is granted, keep the source CSV locally as `data/testicular.csv`. Stage 1 expects the cleaned study extract containing 5,153 male-coded records.

## Run the analysis

Install the Python dependencies:

```bash
python -m pip install numpy pandas openpyxl matplotlib seaborn statsmodels pymannkendall scikit-learn pmdarima prophet xgboost
```

The scripts use `d:/IstinyeTesticularCancer` as the default project path. If your checkout is elsewhere, update the paths in the scripts. Create the `stage1_output/`, `stage2_output/`, `stage3_output/`, and `Figure3/` directories before running the workflow.

Run these scripts in order from the repository root:

1. [`Code_stage1.py`](Code_stage1.py) builds the annual dataset in `stage1_output/dataset.xlsx`.
2. [`Code_stage2.py`](Code_stage2.py) runs descriptive, trend, and stationarity analyses and writes `stage2_output/forecast_ready_dataset.xlsx`.
3. [`Code_stage3.py`](Code_stage3.py) compares forecasting models with naive and drift baselines, runs one-step and multi-step temporal validation, and saves results in `stage3_output/`. The forecasts through 2030 are in `forecast_to_2030.xlsx` in that directory.

Run [`Figure1.py`](Figure1.py) to generate the study workflow diagram and [`Figure3.py`](Figure3.py) after Stage 3 to plot the one-step validation results.

## Citation and contact

If you use this code, please cite the accompanying article by Omer Aydin, Fatih Safa Erenay, Aybeyan Selim, Levent Korkmaz, Huseyin Kusetogullari, Ferhat Ozgur Catak, and Umit Cali. Use the final publication details when available.

Corresponding author: Umit Cali ([umit.cali@ntnu.no](mailto:umit.cali@ntnu.no)).

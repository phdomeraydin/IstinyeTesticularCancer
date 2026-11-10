# =========================================
# STAGE 3
# ADVANCED TIME-SERIES FORECASTING PIPELINE
# Linear Regression, AutoARIMA, ETS, Prophet,
# Random Forest Regressor, and XGBoost Regressor
# =========================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
# Figures are saved to files; avoid Tkinter GUI objects during model execution.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    print("WARNING: pmdarima is not installed. Install with: pip install pmdarima")

try:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    ETS_AVAILABLE = True
except ImportError:
    ETS_AVAILABLE = False
    print("WARNING: ETSModel is not available. Install/update with: pip install -U statsmodels")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("WARNING: Prophet is not installed. Install with: pip install prophet")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost is not installed. Install with: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

BASE_DIR = "d:/IstinyeTesticularCancer"
INPUT_FILE = f"{BASE_DIR}/stage2_output/forecast_ready_dataset.xlsx"
TARGET = "HospitalBasedBurden"
TEST_YEARS = 5
FORECAST_END_YEAR = 2030
RANDOM_STATE = 42

print("=" * 75)
print("STAGE 3 - ADVANCED FORECASTING ANALYSIS")
print("=" * 75)

# =========================================
# 1. READ FORECAST-READY DATA
# =========================================

df = pd.read_excel(INPUT_FILE)
df = df.sort_values("Year").reset_index(drop=True)

required_columns = ["Year", TARGET]
missing_required = [c for c in required_columns if c not in df.columns]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[TARGET]).reset_index(drop=True)
df["ModelYearIndex"] = df["Year"] - df["Year"].min()

print(f"\nInput file: {INPUT_FILE}")
print(f"Target variable: {TARGET}")
print(f"Available years: {int(df['Year'].min())}-{int(df['Year'].max())}")
print(f"Number of annual observations: {len(df)}")

if len(df) <= TEST_YEARS + 6:
    raise ValueError("The time series is too short for the selected holdout period.")

# =========================================
# 2. TEMPORAL VALIDATION DESIGN
# =========================================

test_start_index = len(df) - TEST_YEARS
train_df = df.iloc[:test_start_index].copy()
test_df = df.iloc[test_start_index:].copy()

print("\n" + "=" * 75)
print("TEMPORAL VALIDATION DESIGN")
print("=" * 75)
print(f"Initial training period: {int(train_df['Year'].min())}-{int(train_df['Year'].max())} ({len(train_df)} years)")
print(f"Holdout period: {int(test_df['Year'].min())}-{int(test_df['Year'].max())} ({len(test_df)} years)")
print("Validation strategy: expanding-window, one-step-ahead walk-forward validation")

# =========================================
# 3. COMMON METRICS
# =========================================

def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
    return rmse, mae, mape, r2


model_performance = []

def append_performance(model_name, y_true, y_pred, aic=np.nan, bic=np.nan, selected_spec=""):
    rmse, mae, mape, r2 = calculate_metrics(y_true, y_pred)
    model_performance.append({
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "AIC": aic,
        "BIC": bic,
        "SelectedSpecification": selected_spec
    })
    print(f"\n{model_name}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  R2   : {r2:.4f}")
    if not np.isnan(aic): print(f"  AIC  : {aic:.4f}")
    if not np.isnan(bic): print(f"  BIC  : {bic:.4f}")
    if selected_spec: print(f"  Specification: {selected_spec}")

# =========================================
# 4. LEAKAGE-SAFE ML FEATURES
# =========================================

ML_FEATURES = [
    "ModelYearIndex",
    f"{TARGET}_lag1",
    f"{TARGET}_lag2",
    f"{TARGET}_lag3"
]


def build_autoregressive_ml_dataset(data):
    temp = data[["Year", "ModelYearIndex", TARGET]].copy()
    temp[f"{TARGET}_lag1"] = temp[TARGET].shift(1)
    temp[f"{TARGET}_lag2"] = temp[TARGET].shift(2)
    temp[f"{TARGET}_lag3"] = temp[TARGET].shift(3)
    return temp.dropna().reset_index(drop=True)


def create_rf_model():
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=3,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def create_xgb_model():
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def recursive_ml_forecast(model, full_data, forecast_years):
    history = list(full_data[TARGET].astype(float).values)
    first_year = int(full_data["Year"].min())
    predictions = []
    for year in forecast_years:
        row = pd.DataFrame({
            "ModelYearIndex": [year - first_year],
            f"{TARGET}_lag1": [history[-1]],
            f"{TARGET}_lag2": [history[-2]],
            f"{TARGET}_lag3": [history[-3]]
        })
        pred = float(model.predict(row[ML_FEATURES])[0])
        pred = max(0.0, pred)
        predictions.append(pred)
        history.append(pred)
    return np.asarray(predictions)

# =========================================
# 5. ETS MODEL SELECTION
# =========================================

def fit_best_ets(y):
    # Use a pandas Series with a consecutive index for ETS predictions.
    y = pd.Series(
        np.asarray(y, dtype=float),
        index=pd.RangeIndex(start=0, stop=len(y)),
        name=TARGET
    )

    candidates = []
    specs = [
        (None, False, "ETS(A,N,N)"),
        ("add", False, "ETS(A,A,N)"),
        ("add", True, "ETS(A,Ad,N)")
    ]
    for trend, damped, name in specs:
        try:
            model = ETSModel(
                y,
                error="add",
                trend=trend,
                damped_trend=damped,
                seasonal=None,
                initialization_method="estimated"
            )
            fit = model.fit(disp=False)
            candidates.append({
                "fit": fit,
                "name": name,
                "aic": float(fit.aic),
                "bic": float(fit.bic)
            })
        except Exception:
            pass
    if not candidates:
        raise RuntimeError("No ETS candidate could be fitted.")
    return min(candidates, key=lambda x: x["aic"])

# =========================================
# 6. WALK-FORWARD VALIDATION
# =========================================

print("\n" + "=" * 75)
print("MODEL VALIDATION")
print("=" * 75)

actual_test = test_df[TARGET].astype(float).values
test_years = test_df["Year"].astype(int).values
validation_predictions = {}

# Reference models: Naive and Drift
naive_predictions = []
drift_predictions = []

for i in range(test_start_index, len(df)):
    fold_train = df.iloc[:i]
    train_values = fold_train[TARGET].to_numpy(dtype=float)
    train_years = fold_train["Year"].to_numpy(dtype=int)
    target_year = int(df.iloc[i]["Year"])

    if len(train_values) < 2:
        raise ValueError("Drift requires at least two training observations.")

    elapsed_years = train_years[-1] - train_years[0]
    if elapsed_years <= 0:
        raise ValueError("Training years must be strictly increasing.")

    # Naive: use the last observed value.
    naive_pred = train_values[-1]

    # Drift: extend the average annual change in the training period.
    annual_drift = (
        (train_values[-1] - train_values[0]) / elapsed_years
    )
    horizon = target_year - train_years[-1]
    drift_pred = train_values[-1] + horizon * annual_drift

    naive_predictions.append(max(0.0, naive_pred))
    drift_predictions.append(max(0.0, drift_pred))

validation_predictions["Naive"] = np.asarray(naive_predictions)
validation_predictions["Drift"] = np.asarray(drift_predictions)

append_performance("Naive", actual_test, naive_predictions)
append_performance("Drift", actual_test, drift_predictions)

# 6.1 Linear Regression
lr_predictions = []
for i in range(test_start_index, len(df)):
    fold_train = df.iloc[:i].copy()
    fold_test = df.iloc[[i]].copy()
    model = LinearRegression()
    model.fit(fold_train[["ModelYearIndex"]], fold_train[TARGET])
    pred = float(model.predict(fold_test[["ModelYearIndex"]])[0])
    lr_predictions.append(max(0.0, pred))
validation_predictions["Linear Regression"] = np.asarray(lr_predictions)
append_performance("Linear Regression", actual_test, lr_predictions)

# 6.2 AutoARIMA
if PMDARIMA_AVAILABLE:
    autoarima_predictions = []
    for i in range(test_start_index, len(df)):
        y_train = df.iloc[:i][TARGET].astype(float).values
        m = auto_arima(
            y_train,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            d=None, max_d=2,
            seasonal=False,
            information_criterion="aicc",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False
        )
        pred = float(m.predict(n_periods=1)[0])
        autoarima_predictions.append(max(0.0, pred))
    validation_predictions["AutoARIMA"] = np.asarray(autoarima_predictions)

    autoarima_full = auto_arima(
        df[TARGET].astype(float).values,
        start_p=0, start_q=0,
        max_p=3, max_q=3,
        d=None, max_d=2,
        seasonal=False,
        information_criterion="aicc",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False
    )
    append_performance(
        "AutoARIMA",
        actual_test,
        autoarima_predictions,
        aic=float(autoarima_full.aic()),
        bic=float(autoarima_full.bic()),
        selected_spec=f"ARIMA{autoarima_full.order}"
    )

# 6.3 ETS
if ETS_AVAILABLE:
    ets_predictions = []
    for i in range(test_start_index, len(df)):
        y_train = df.iloc[:i][TARGET].astype(float).values
        best_ets = fit_best_ets(y_train)
        pred = float(np.asarray(best_ets["fit"].forecast(1))[0])
        ets_predictions.append(max(0.0, pred))
    validation_predictions["ETS"] = np.asarray(ets_predictions)
    ets_full_result = fit_best_ets(df[TARGET].astype(float).values)
    append_performance(
        "ETS",
        actual_test,
        ets_predictions,
        aic=ets_full_result["aic"],
        bic=ets_full_result["bic"],
        selected_spec=ets_full_result["name"]
    )

# 6.4 Prophet
if PROPHET_AVAILABLE:
    prophet_predictions = []
    for i in range(test_start_index, len(df)):
        fold_train = df.iloc[:i].copy()
        ptrain = fold_train[["Year", TARGET]].copy()
        ptrain.columns = ["ds", "y"]
        ptrain["ds"] = pd.to_datetime(ptrain["ds"].astype(int).astype(str), format="%Y")
        pmodel = Prophet(
            growth="linear",
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            interval_width=0.95
        )
        pmodel.fit(ptrain)
        future_one = pd.DataFrame({
            "ds": pd.to_datetime([str(int(df.iloc[i]["Year"]))], format="%Y")
        })
        pred = float(pmodel.predict(future_one)["yhat"].iloc[0])
        prophet_predictions.append(max(0.0, pred))
    validation_predictions["Prophet"] = np.asarray(prophet_predictions)
    append_performance("Prophet", actual_test, prophet_predictions)

# 6.5 Random Forest
rf_predictions = []
for i in range(test_start_index, len(df)):
    fold_train = df.iloc[:i].copy()
    ml_train = build_autoregressive_ml_dataset(fold_train)
    rf_model = create_rf_model()
    rf_model.fit(ml_train[ML_FEATURES], ml_train[TARGET])
    h = fold_train[TARGET].astype(float).values
    current_year = int(df.iloc[i]["Year"])
    x_test = pd.DataFrame({
        "ModelYearIndex": [current_year - int(df["Year"].min())],
        f"{TARGET}_lag1": [h[-1]],
        f"{TARGET}_lag2": [h[-2]],
        f"{TARGET}_lag3": [h[-3]]
    })
    pred = float(rf_model.predict(x_test[ML_FEATURES])[0])
    rf_predictions.append(max(0.0, pred))
validation_predictions["Random Forest"] = np.asarray(rf_predictions)
append_performance("Random Forest", actual_test, rf_predictions)

# 6.6 XGBoost
if XGBOOST_AVAILABLE:
    xgb_predictions = []
    for i in range(test_start_index, len(df)):
        fold_train = df.iloc[:i].copy()
        ml_train = build_autoregressive_ml_dataset(fold_train)
        xgb_model = create_xgb_model()
        xgb_model.fit(ml_train[ML_FEATURES], ml_train[TARGET])
        h = fold_train[TARGET].astype(float).values
        current_year = int(df.iloc[i]["Year"])
        x_test = pd.DataFrame({
            "ModelYearIndex": [current_year - int(df["Year"].min())],
            f"{TARGET}_lag1": [h[-1]],
            f"{TARGET}_lag2": [h[-2]],
            f"{TARGET}_lag3": [h[-3]]
        })
        pred = float(xgb_model.predict(x_test[ML_FEATURES])[0])
        xgb_predictions.append(max(0.0, pred))
    validation_predictions["XGBoost"] = np.asarray(xgb_predictions)
    append_performance("XGBoost", actual_test, xgb_predictions)

# =========================================
# 7. SAVE VALIDATION OUTPUTS
# =========================================

validation_table = pd.DataFrame({"Year": test_years, "Actual": actual_test})
for model_name, preds in validation_predictions.items():
    validation_table[model_name] = preds
validation_table.to_excel(f"{BASE_DIR}/stage3_output/walk_forward_validation_predictions.xlsx", index=False)

performance_df = pd.DataFrame(model_performance).sort_values("RMSE").reset_index(drop=True)
performance_df.to_excel(f"{BASE_DIR}/stage3_output/model_comparison.xlsx", index=False)

print("\n" + "=" * 75)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 75)
print(performance_df.to_string(index=False))

best_model_name = performance_df.iloc[0]["Model"]
print(f"\nBEST MODEL BY RMSE: {best_model_name}")

plt.figure(figsize=(10, 6))
plt.bar(performance_df["Model"], performance_df["RMSE"])
plt.ylabel("RMSE")
plt.xlabel("Model")
plt.title(f"Forecasting Model Comparison - {TARGET}")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(f"{BASE_DIR}/stage3_output/model_comparison_rmse.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================
# 8. REFIT FULL DATA AND FORECAST TO 2030
# =========================================

last_observed_year = int(df["Year"].max())
if last_observed_year >= FORECAST_END_YEAR:
    raise ValueError("FORECAST_END_YEAR must be greater than the last observed year.")

forecast_years = np.arange(last_observed_year + 1, FORECAST_END_YEAR + 1)
forecast_horizon = len(forecast_years)
all_forecasts = {}

# Reference-model forecasts
observed_values = df[TARGET].to_numpy(dtype=float)
observed_years = df["Year"].to_numpy(dtype=int)

elapsed_years = observed_years[-1] - observed_years[0]
if len(observed_values) < 2 or elapsed_years <= 0:
    raise ValueError("Drift requires at least two distinct observed years.")

annual_drift = (
    (observed_values[-1] - observed_values[0]) / elapsed_years
)
future_horizons = forecast_years - observed_years[-1]

all_forecasts["Naive"] = {
    "predictions": np.full(
        forecast_horizon,
        observed_values[-1],
        dtype=float
    )
}

all_forecasts["Drift"] = {
    "predictions": np.maximum(
        observed_values[-1] + future_horizons * annual_drift,
        0.0
    )
}

# 8.1 Linear Regression
lr_full = LinearRegression()
lr_full.fit(df[["ModelYearIndex"]], df[TARGET])
future_lr = pd.DataFrame({"ModelYearIndex": forecast_years - int(df["Year"].min())})
lr_future_pred = np.maximum(lr_full.predict(future_lr), 0)
all_forecasts["Linear Regression"] = {
    "predictions": lr_future_pred
}

# 8.2 AutoARIMA
if PMDARIMA_AVAILABLE:
    pred, ci = autoarima_full.predict(n_periods=forecast_horizon, return_conf_int=True, alpha=0.05)
    all_forecasts["AutoARIMA"] = {
        "predictions": np.maximum(pred, 0),
        "lower_ci": np.maximum(ci[:, 0], 0),
        "upper_ci": ci[:, 1]
    }

# 8.3 ETS
if ETS_AVAILABLE:
    ets_fit = ets_full_result["fit"]
    ets_pred = np.maximum(np.asarray(ets_fit.forecast(forecast_horizon)), 0)
    try:
        pred_obj = ets_fit.get_prediction(start=len(df), end=len(df) + forecast_horizon - 1)
        sf = pred_obj.summary_frame(alpha=0.05)
        print("\nETS prediction interval output:")
        print(sf.to_string())
        lower_cols = [c for c in sf.columns if "lower" in c.lower()]
        upper_cols = [c for c in sf.columns if "upper" in c.lower()]
        if not lower_cols or not upper_cols:
            raise ValueError
        ets_lower = sf[lower_cols[0]].values
        ets_upper = sf[upper_cols[0]].values
    except Exception as exc:
        raise RuntimeError(
            "ETS prediction intervals could not be generated. "
            "No residual-based fallback interval was substituted. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    all_forecasts["ETS"] = {
        "predictions": ets_pred,
        "lower_ci": np.maximum(ets_lower, 0),
        "upper_ci": ets_upper
    }

# 8.4 Prophet
if PROPHET_AVAILABLE:
    pfull = df[["Year", TARGET]].copy()
    pfull.columns = ["ds", "y"]
    pfull["ds"] = pd.to_datetime(pfull["ds"].astype(int).astype(str), format="%Y")
    pmodel = Prophet(
        growth="linear",
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.95
    )
    pmodel.fit(pfull)
    pfuture = pd.DataFrame({"ds": pd.to_datetime(forecast_years.astype(str), format="%Y")})
    pout = pmodel.predict(pfuture)
    all_forecasts["Prophet"] = {
        "predictions": np.maximum(pout["yhat"].values, 0),
        "lower_ci": np.maximum(pout["yhat_lower"].values, 0),
        "upper_ci": pout["yhat_upper"].values
    }

# 8.5 Random Forest
ml_full = build_autoregressive_ml_dataset(df)
rf_full = create_rf_model()
rf_full.fit(ml_full[ML_FEATURES], ml_full[TARGET])
rf_future = recursive_ml_forecast(rf_full, df, forecast_years)
all_forecasts["Random Forest"] = {
    "predictions": rf_future
}

# 8.6 XGBoost
if XGBOOST_AVAILABLE:
    xgb_full = create_xgb_model()
    xgb_full.fit(ml_full[ML_FEATURES], ml_full[TARGET])
    xgb_future = recursive_ml_forecast(xgb_full, df, forecast_years)
    all_forecasts["XGBoost"] = {
        "predictions": xgb_future
    }

# =========================================
# 9. SAVE FINAL FORECASTS
# =========================================

forecast_table = pd.DataFrame({"Year": forecast_years})
for model_name, values in all_forecasts.items():
    safe = model_name.replace(" ", "_")
    forecast_table[f"{safe}_Forecast"] = values["predictions"]

    if "lower_ci" in values and "upper_ci" in values:
        forecast_table[f"{safe}_Lower_95"] = values["lower_ci"]
        forecast_table[f"{safe}_Upper_95"] = values["upper_ci"]

forecast_table.to_excel(f"{BASE_DIR}/stage3_output/forecast_to_2030.xlsx", index=False)
print("\n✓ forecast_to_2030.xlsx saved")
print(forecast_table.to_string(index=False))

# =========================================
# 10. FORECAST FIGURES
# =========================================

n_models = len(all_forecasts)
fig, axes = plt.subplots(
    n_models,
    1,
    figsize=(14, 4.5 * n_models),
    sharex=True,
    sharey=True
)
if n_models == 1:
    axes = [axes]

for ax, (model_name, forecast_data) in zip(axes, all_forecasts.items()):
    ax.plot(df["Year"], df[TARGET], marker="o", linewidth=2, label="Observed")
    ax.plot(forecast_years, forecast_data["predictions"], marker="s", linestyle="--", linewidth=2, label="Forecast")
    if "lower_ci" in forecast_data and "upper_ci" in forecast_data:
        ax.fill_between(
            forecast_years,
            forecast_data["lower_ci"],
            forecast_data["upper_ci"],
            alpha=0.20,
            label="95% model-based interval"
        )
    ax.axvline(x=last_observed_year + 0.5, linestyle=":", linewidth=1.5)
    ax.set_title(model_name)
    ax.set_xlabel("Year")
    ax.set_ylabel("Patient count")
    ax.tick_params(axis="x", labelbottom=True)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    
for ax in plt.gcf().axes:
    ax.tick_params(axis="x", labelbottom=True)
plt.tight_layout()
plt.savefig(f"{BASE_DIR}/stage3_output/forecast_all_models.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================
# 11. FEATURE IMPORTANCE
# =========================================

rf_importance = pd.DataFrame({
    "Feature": ML_FEATURES,
    "Importance": rf_full.feature_importances_
}).sort_values("Importance", ascending=False)
rf_importance.to_excel(f"{BASE_DIR}/stage3_output/random_forest_feature_importance.xlsx", index=False)

plt.figure(figsize=(9, 5))
plt.barh(rf_importance["Feature"][::-1], rf_importance["Importance"][::-1])
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(f"{BASE_DIR}/stage3_output/random_forest_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

if XGBOOST_AVAILABLE:
    xgb_importance = pd.DataFrame({
        "Feature": ML_FEATURES,
        "Importance": xgb_full.feature_importances_
    }).sort_values("Importance", ascending=False)
    xgb_importance.to_excel(f"{BASE_DIR}/stage3_output/xgboost_feature_importance.xlsx", index=False)

    plt.figure(figsize=(9, 5))
    plt.barh(xgb_importance["Feature"][::-1], xgb_importance["Importance"][::-1])
    plt.xlabel("Importance")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/stage3_output/xgboost_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(xgb_full)
            shap_values = explainer.shap_values(ml_full[ML_FEATURES])
            shap.summary_plot(shap_values, ml_full[ML_FEATURES], show=False)
            plt.tight_layout()
            plt.savefig(f"{BASE_DIR}/stage3_output/xgboost_shap_summary.png", dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"SHAP analysis skipped: {e}")

# =========================================
# 12. SAVE MODEL INFORMATION
# =========================================

model_info = []
if PMDARIMA_AVAILABLE:
    model_info.append({
        "Model": "AutoARIMA",
        "Specification": f"ARIMA{autoarima_full.order}",
        "AIC": float(autoarima_full.aic()),
        "BIC": float(autoarima_full.bic())
    })
if ETS_AVAILABLE:
    model_info.append({
        "Model": "ETS",
        "Specification": ets_full_result["name"],
        "AIC": ets_full_result["aic"],
        "BIC": ets_full_result["bic"]
    })

pd.DataFrame(model_info).to_excel(f"{BASE_DIR}/stage3_output/selected_model_information.xlsx", index=False)

# =========================================
# 13. FINAL SUMMARY
# =========================================

print("\n" + "=" * 75)
print("STAGE 3 COMPLETED SUCCESSFULLY")
print("=" * 75)
print(f"\nPrimary target: {TARGET}")
print(f"Observed period: {int(df['Year'].min())}-{last_observed_year}")
print(f"Forecast period: {int(forecast_years.min())}-{int(forecast_years.max())}")
print(f"Best model by walk-forward RMSE: {best_model_name}")
print("\nGenerated files:")
print("  - walk_forward_validation_predictions.xlsx")
print("  - model_comparison.xlsx")
print("  - model_comparison_rmse.png")
print("  - forecast_to_2030.xlsx")
print("  - forecast_all_models.png")
print("  - random_forest_feature_importance.xlsx")
print("  - random_forest_feature_importance.png")
print("  - xgboost_feature_importance.xlsx (if available)")
print("  - xgboost_feature_importance.png (if available)")
print("  - xgboost_shap_summary.png (if SHAP available)")
print("  - selected_model_information.xlsx")


# =========================================
# 14. MULTI-STEP ROLLING-ORIGIN VALIDATION
# =========================================

MIN_TRAIN_YEARS = 10
MAX_HORIZON = 5

multi_data = df.sort_values("Year").reset_index(drop=True).copy()
years = multi_data["Year"].to_numpy(dtype=int)

if not np.all(np.diff(years) == 1):
    raise ValueError(
        "Multi-step validation requires consecutive annual observations."
    )

if len(multi_data) < MIN_TRAIN_YEARS + MAX_HORIZON:
    raise ValueError(
        "Insufficient observations for the selected training window "
        "and maximum forecast horizon."
    )

multi_records = []

for split in range(MIN_TRAIN_YEARS, len(multi_data)):
    train = multi_data.iloc[:split].copy()

    horizon_count = min(MAX_HORIZON, len(multi_data) - split)
    future = multi_data.iloc[split:split + horizon_count].copy()

    future_years = future["Year"].to_numpy(dtype=int)
    actual_values = future[TARGET].to_numpy(dtype=float)
    train_values = train[TARGET].to_numpy(dtype=float)

    origin_year = int(train["Year"].iloc[-1])
    first_train_year = int(train["Year"].iloc[0])
    steps = future_years - origin_year

    print(
        f"\nMulti-step validation: training through {origin_year}, "
        f"forecasting {horizon_count} year(s)"
    )

    predictions = {}

    # Naive
    predictions["Naive"] = np.repeat(
        train_values[-1], horizon_count
    )

    # Drift
    annual_change = (
        (train_values[-1] - train_values[0])
        / (origin_year - first_train_year)
    )
    predictions["Drift"] = (
        train_values[-1] + steps * annual_change
    )

    # Linear Regression
    model = LinearRegression()
    model.fit(train[["ModelYearIndex"]], train[TARGET])
    predictions["Linear Regression"] = model.predict(
        future[["ModelYearIndex"]]
    )

    # AutoARIMA: select the specification using training data only.
    if PMDARIMA_AVAILABLE:
        model = auto_arima(
            train_values,
            start_p=0,
            start_q=0,
            max_p=3,
            max_q=3,
            d=None,
            max_d=2,
            seasonal=False,
            information_criterion="aicc",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False
        )
        predictions["AutoARIMA"] = model.predict(
            n_periods=horizon_count
        )

    # ETS: select the specification using training data only.
    if ETS_AVAILABLE:
        selected = fit_best_ets(train_values)
        predictions["ETS"] = selected["fit"].forecast(
            horizon_count
        )

    # Prophet
    if PROPHET_AVAILABLE:
        prophet_train = pd.DataFrame({
            "ds": pd.to_datetime(
                train["Year"].astype(str), format="%Y"
            ),
            "y": train_values
        })

        model = Prophet(
            growth="linear",
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            interval_width=0.95
        )
        model.fit(prophet_train)

        prophet_future = pd.DataFrame({
            "ds": pd.to_datetime(
                future_years.astype(str), format="%Y"
            )
        })
        predictions["Prophet"] = (
            model.predict(prophet_future)["yhat"].to_numpy()
        )

    # Machine-learning models use recursive predictions.
    # Actual future target values are never added to their lag history.
    ml_train = build_autoregressive_ml_dataset(train)

    model = create_rf_model()
    model.fit(ml_train[ML_FEATURES], ml_train[TARGET])
    predictions["Random Forest"] = recursive_ml_forecast(
        model, train, future_years
    )

    if XGBOOST_AVAILABLE:
        model = create_xgb_model()
        model.fit(ml_train[ML_FEATURES], ml_train[TARGET])
        predictions["XGBoost"] = recursive_ml_forecast(
            model, train, future_years
        )

    for model_name, predicted in predictions.items():
        predicted = np.maximum(
            np.asarray(predicted, dtype=float).reshape(-1), 0.0
        )

        if (
            len(predicted) != horizon_count
            or not np.isfinite(predicted).all()
        ):
            raise ValueError(
                f"Invalid predictions: {model_name}, "
                f"origin={origin_year}"
            )

        for j in range(horizon_count):
            multi_records.append({
                "Model": model_name,
                "OriginYear": origin_year,
                "TargetYear": int(future_years[j]),
                "Horizon": int(steps[j]),
                "Actual": float(actual_values[j]),
                "Predicted": float(predicted[j])
            })

multi_predictions = pd.DataFrame(multi_records)

summary_records = []

for (model_name, horizon), group in multi_predictions.groupby(
    ["Model", "Horizon"]
):
    actual = group["Actual"].to_numpy()
    predicted = group["Predicted"].to_numpy()
    error = predicted - actual

    nonzero = actual != 0
    mape = (
        100 * np.mean(np.abs(error[nonzero] / actual[nonzero]))
        if nonzero.any()
        else np.nan
    )

    summary_records.append({
        "Model": model_name,
        "Horizon": int(horizon),
        "N_predictions": len(group),
        "FirstOrigin": int(group["OriginYear"].min()),
        "LastOrigin": int(group["OriginYear"].max()),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "MAPE": float(mape),
        "MeanError": float(np.mean(error))
    })

multi_summary = (
    pd.DataFrame(summary_records)
    .sort_values(["Horizon", "RMSE"])
    .reset_index(drop=True)
)

multi_predictions.to_excel(
    f"{BASE_DIR}/stage3_output/multistep_validation_predictions.xlsx",
    index=False
)

multi_summary.to_excel(
    f"{BASE_DIR}/stage3_output/multistep_validation_summary.xlsx",
    index=False
)

print("\nMULTI-STEP VALIDATION RESULTS")
print(multi_summary.to_string(index=False))
print("\nMulti-step validation completed successfully.")
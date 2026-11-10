from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --------------------------------------------------
# OUTPUT DIRECTORY
# --------------------------------------------------
OUTPUT_DIR = Path(
    "d:/IstinyeTesticularCancer/Figure1"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(18, 10))
fig.patch.set_facecolor("white")

ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")

# --------------------------------------------------
# DRAWING FUNCTIONS
# --------------------------------------------------
def panel(x, y, width, height, title, color):
    """Draw a stage panel and its colored heading."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.3,
        edgecolor=color,
        facecolor="#FAFBFC",
        zorder=1,
    )
    ax.add_patch(box)

    header_height = 0.72
    header = FancyBboxPatch(
        (x, y + height - header_height),
        width, header_height,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=0,
        facecolor=color,
        zorder=2,
    )
    ax.add_patch(header)

    ax.text(
        x + width / 2,
        y + height - header_height / 2,
        title,
        ha="center",
        va="center",
        color="white",
        fontsize=11,
        fontweight="bold",
        linespacing=1.3,
        zorder=3,
    )


def section(x, top, heading, lines, width=31):
    """Add a heading and wrapped text; return the next free position."""
    ax.text(
        x, top, heading,
        ha="left", va="top",
        fontsize=10.5,
        fontweight="bold",
        color="#17324D",
    )
    cursor = top - 0.33

    for line in lines:
        wrapped = textwrap.wrap(
            line,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]

        for item in wrapped:
            ax.text(
                x, cursor, item,
                ha="left", va="top",
                fontsize=10,
                color="#263238",
            )
            cursor -= 0.235

        cursor -= 0.10

    return cursor - 0.20


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=19,
        linewidth=1.8,
        color="#607D8B",
        zorder=4,
    ))


# --------------------------------------------------
# LAYOUT
# --------------------------------------------------
panel_width = 3.18
panel_height = 7.65
panel_bottom = 1.70

positions = [0.20, 3.80, 7.40, 11.00, 14.60]
colors = ["#254E70", "#28689C", "#4C7835", "#B65C21", "#70508C"]

titles = [
    "DATA SOURCE",
    "STAGE 1\nDATA PREPARATION",
    "STAGE 2\nTIME-SERIES ANALYSIS",
    "STAGE 3\nMODEL EVALUATION",
    "FINAL FORECASTING",
]

for x, title, color in zip(positions, titles, colors):
    panel(
        x, panel_bottom,
        panel_width, panel_height,
        title, color,
    )

for left, right in zip(positions[:-1], positions[1:]):
    arrow(
        left + panel_width + 0.03, 5.50,
        right - 0.05, 5.50,
    )

content_top = 8.35
padding = 0.18

# --------------------------------------------------
# DATA SOURCE
# --------------------------------------------------
x = positions[0] + padding
top = section(x, content_top, "Source and selection", [
    "MLP Care network, Türkiye",
    "Source: 33 hospitals",
    "5,173 original records",
    "20 female-coded records excluded",
    "5,153 records retained",
])

top = section(x, top, "Source variables", [
    "Diagnosis year",
    "Annual hospital applications",
    "Demographic and clinical data",
    "Recorded deaths",
    "Healthcare utilization",
])

section(x, top, "Study setting", [
    "Retrospective, multicenter hospital data",
    "Hospital-observed patient counts",
])

# --------------------------------------------------
# STAGE 1
# --------------------------------------------------
x = positions[1] + padding
top = section(x, content_top, "Data preparation", [
    "Variable standardization",
    "Date processing",
    "Data quality checks",
])

top = section(x, top, "Primary annual outcome", [
    "Patients attending in year t",
    "Diagnosis between t - 4 and t, inclusive",
    "Five-calendar-year window",
])

top = section(x, top, "Annual dataset", [
    "2010–2024: 15 observations",
    "First window: 2006–2010",
    "2025 diagnoses excluded",
    "Counts and clinical summaries",
])

section(x, top, "Forecasting predictors", [
    "Linear Regression: year index",
    "RF/XGBoost: year index and outcome lags 1–3",
])

# --------------------------------------------------
# STAGE 2
# --------------------------------------------------
x = positions[2] + padding
top = section(x, content_top, "Descriptive analysis", [
    "Annual summary statistics",
    "Time-series visualization",
    "Exploratory correlations",
])

top = section(x, top, "Trend analysis", [
    "Mann–Kendall test",
    "Sen's slope",
])

top = section(x, top, "Stationarity analysis", [
    "Augmented Dickey–Fuller test",
    "KPSS test",
])

section(x, top, "Temporal dependence", [
    "Autocorrelation",
    "Partial autocorrelation",
])

# --------------------------------------------------
# STAGE 3
# --------------------------------------------------
x = positions[3] + padding
top = section(x, content_top, "Eight forecasting methods", [
    "Linear Regression, AutoARIMA",
    "ETS, Prophet",
    "Random Forest, XGBoost",
    "References: Naive and Drift",
])

top = section(x, top, "Primary validation", [
    "Expanding training window",
    "One-step-ahead forecasts",
    "Evaluation years: 2020–2024",
    "RMSE, MAE, MAPE and R²",
])

top = section(x, top, "Supplementary validation", [
    "Rolling forecast origins",
    "Initial training: 10 years",
    "Forecast horizons: 1–5 years",
    "Origins for horizons 1–5: 5, 4, 3, 2, 1",
])

section(x, top, "Primary ranking criterion", [
    "Lowest validation RMSE",
])

# --------------------------------------------------
# FINAL FORECASTING
# --------------------------------------------------
x = positions[4] + padding
top = section(x, content_top, "Use complete annual series", [
    "Refit models on 2010–2024",
    "Apply reference methods to the full series",
])

top = section(x, top, "Forecast period", [
    "2025–2030",
    "Point forecasts from all eight methods",
])

top = section(x, top, "Model-specific intervals", [
    "95% prediction intervals: AutoARIMA and ETS",
    "95% uncertainty intervals: Prophet",
    "Interval coverage not evaluated",
])

section(x, top, "Interpretation", [
    "Conditional network-level projections",
    "Six-year horizon not backtested",
])

# --------------------------------------------------
# BOTTOM NOTE
# --------------------------------------------------
note = FancyBboxPatch(
    (0.20, 0.25), 17.58, 1.05,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    linewidth=1,
    edgecolor="#AAB7C4",
    facecolor="#F1F5F9",
)
ax.add_patch(note)

ax.text(
    8.99, 0.95,
    "STUDY OUTPUTS",
    ha="center", va="center",
    fontsize=11,
    fontweight="bold",
    color="#17324D",
)

ax.text(
    8.99, 0.56,
    "Annual hospital-observed counts  •  Temporal validation results  "
    "•  Conditional forecasts through 2030",
    ha="center", va="center",
    fontsize=11,
    color="#263238",
)

# --------------------------------------------------
# SAVE
# --------------------------------------------------
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

for extension in ["png", "pdf", "svg"]:
    output_file = OUTPUT_DIR / f"figure1_study_workflow.{extension}"
    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Saved: {output_file}")

plt.close(fig)
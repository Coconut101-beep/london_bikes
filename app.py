"""London bikes dashboard: explore daily hires, predict, and compare models."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

from open_meteo import open_meteo, open_meteo_history

ROOT = Path(__file__).resolve().parent
BIKES_URL = (
    "https://raw.githubusercontent.com/kostis-christodoulou/"
    "am01-code-sep2026/main/data/london_bikes.csv"
)
COEF_PATH = ROOT / "model_coefficients.csv"

DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]
SET2 = px.colors.qualitative.Set2
PALETTE = {"A": "#2e86ab", "B": "#e09f3e", "dashboard": "#1b365d"}

WEATHER_OPTIONS = [
    {"label": "Temperature (°C)", "value": "temp"},
    {"label": "Humidity (%)", "value": "humidity"},
    {"label": "Precipitation (mm)", "value": "precip"},
    {"label": "Wind speed (km/h)", "value": "windspeed"},
    {"label": "Cloud cover (%)", "value": "cloudcover"},
]
WEATHER_LABELS = {opt["value"]: opt["label"] for opt in WEATHER_OPTIONS}
WEATHER_ICONS = {
    "temp": "temperature-half",
    "humidity": "droplet",
    "precip": "cloud-rain",
    "windspeed": "wind",
    "cloudcover": "cloud",
}

COLOUR_OPTIONS = [
    {"label": "Weekend", "value": "weekend"},
    {"label": "Season", "value": "season_name"},
]

MODEL_LABELS = {
    "M1_temp": "M1 · Temp only",
    "M2_weekday": "M2 · Weekday + extras",
    "M3_weather": "M3 · Core weather",
    "M4_season": "M4 · Season + daylight",
}
FIT_ROWS = {
    "R-squared": "r2",
    "R-squared Adj.": "r2_adj_rounded",
    "AIC": "aic",
    "Adj R2": "adj_r2",
    "N": "n",
    "Residual SE": "residual_se",
}
COEF_FOCUS = ["temp", "precip", "humidity", "windspeed", "visibility"]

EMPTY_FIG = go.Figure()
EMPTY_FIG.update_layout(
    template="plotly_white",
    annotations=[
        {
            "text": "No data to display",
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": 0.5,
            "showarrow": False,
            "font": {"size": 16, "color": "#5b6770"},
        }
    ],
    xaxis={"visible": False},
    yaxis={"visible": False},
    margin={"l": 40, "r": 20, "t": 50, "b": 40},
)

TABLE_STYLE = {
    "style_table": {"overflowX": "auto"},
    "style_header": {
        "backgroundColor": "#1b365d",
        "color": "white",
        "fontWeight": "600",
        "border": "none",
    },
    "style_cell": {
        "fontFamily": "system-ui, -apple-system, Segoe UI, sans-serif",
        "fontSize": 13,
        "padding": "8px 10px",
        "textAlign": "center",
        "border": "1px solid #e6ebef",
    },
    "style_data_conditional": [
        {"if": {"row_index": "odd"}, "backgroundColor": "#f7f9fb"}
    ],
}


def fa(name, extra=""):
    return html.I(className=f"fa-solid fa-{name} {extra}".strip())


def kpi_card(icon, value, label, hint=None):
    children = [
        html.Div(fa(icon), className="kpi-icon"),
        html.Div(value, className="kpi-value"),
        html.Div(label, className="kpi-label"),
    ]
    if hint:
        children.append(html.Div(hint, className="kpi-hint"))
    return html.Div(children, className="kpi")


def equation_card(icon, title, badge, equation, note):
    return html.Div(
        [
            html.Div(
                [
                    html.Span([fa(icon), " ", title], className="eq-title"),
                    html.Span(badge, className="eq-badge"),
                ],
                className="eq-head",
            ),
            html.P(["bikes_hired  =  ", html.Span(equation, className="eq-math")]),
            html.P(note, className="eq-note"),
        ],
        className="eq-card",
    )


MODEL_EQUATIONS = html.Div(
    [
        html.H2(
            [fa("square-root-variable"), " The three models"],
            className="section-title",
        ),
        html.Div(
            [
                equation_card(
                    "rocket",
                    "Dashboard model",
                    "Simpler alternative",
                    "Intercept + 813.11 × temperature + (−203.57) × humidity + a weekday effect",
                    "Intercept = 33,530.  Weekday (Mon = 0): Tue +2,382 · Wed +2,389 · Thu +2,385 · Fri +788 · Sat −1,394 · Sun −3,691. Kept as a simpler alternative on Predict.",
                ),
                equation_card(
                    "layer-group",
                    "Table A · M4 Season + daylight",
                    "model_coff2 · adj. R² 0.67",
                    (
                        "Intercept + 879.1 × temperature + (−182.0) × precipitation + "
                        "(−158.1) × wind speed + 99.0 × visibility + 78.6 × sea-level pressure + "
                        "(−38.3) × precip cover + (−282.2) × feels-like min + 769.6 × UV index + "
                        "389.6 × daylight + a season effect + a post-2023 shift + a weekday effect"
                    ),
                    "Intercept = −63,329. Autumn is the season baseline; Monday is the weekday baseline.",
                ),
                equation_card(
                    "layer-group",
                    "Table B · M2 Weekday + extras",
                    "Used on Predict · adj. R² 0.69",
                    (
                        "Intercept + 589.4 × temperature + (−211.8) × precipitation + "
                        "(−107.8) × humidity + (−192.2) × wind speed + (−10.7) × cloud cover + "
                        "40.8 × solar radiation + 54.0 × visibility + a year effect + "
                        "a month effect + a weekday effect"
                    ),
                    "Intercept = 27,283. Strongest in-sample fit of the three (lowest AIC). Monday is the weekday baseline. Years after 2025 use the 2025 effect.",
                ),
            ],
            className="equation-grid",
        ),
    ],
    className="equations",
)


def _weekend_label(value) -> str:
    text = str(value).strip().upper()
    if value is True or text in {"TRUE", "1", "WEEKEND"}:
        return "Weekend"
    return "Weekday"


def parse_num(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = (
        str(value)
        .replace(",", "")
        .replace("*", "")
        .replace("(", "")
        .replace(")", "")
        .replace('"', "")
        .strip()
    )
    if text == "" or text.lower() == "nan":
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def load_bikes():
    """Load the London bikes CSV; return (dataframe, error_message)."""
    try:
        bike = pd.read_csv(BIKES_URL)
        bike["date"] = pd.to_datetime(bike["date"], utc=True, errors="coerce")
        bike["weekend_label"] = bike["weekend"].map(_weekend_label)
        bike["day_of_week"] = pd.Categorical(
            bike["day_of_week"], categories=DAY_ORDER, ordered=True
        )
        if "season_name" in bike.columns:
            present = [s for s in SEASON_ORDER if s in set(bike["season_name"].dropna())]
            extras = [
                s
                for s in bike["season_name"].dropna().unique()
                if s not in SEASON_ORDER
            ]
            bike["season_name"] = pd.Categorical(
                bike["season_name"], categories=present + extras
            )
        if "year" in bike.columns:
            bike["year"] = pd.to_numeric(bike["year"], errors="coerce")
        bike["date_label"] = bike["date"].dt.strftime("%Y-%m-%d")
        return bike, None
    except Exception as exc:
        return pd.DataFrame(), f"Could not load the London bikes data: {exc}"


def load_coefficients():
    """Read term / coefficient pairs from the exported model CSV."""
    coefs = pd.read_csv(COEF_PATH)
    mapping = dict(zip(coefs["term"], coefs["coefficient"]))
    intercept = float(mapping.pop("Intercept", 0.0))
    day_coefs = {
        term: float(value)
        for term, value in mapping.items()
        if str(term).startswith("day_")
    }
    numeric_coefs = {
        term: float(value)
        for term, value in mapping.items()
        if not str(term).startswith("day_")
    }
    return intercept, numeric_coefs, day_coefs


def parse_model_table(path: Path, source_id: str, source_label: str):
    """Turn a wide nested-model CSV into fit-stats and coefficient frames."""
    raw = pd.read_csv(path)
    model_ids = [c for c in raw.columns if c in MODEL_LABELS]
    fit_rows = []
    coef_rows = []
    for model_id in model_ids:
        stats = {"source": source_id, "source_label": source_label, "model": model_id}
        for _, row in raw.iterrows():
            term = str(row["term"]).strip()
            value = parse_num(row[model_id])
            if term in FIT_ROWS:
                stats[FIT_ROWS[term]] = value
            elif term and not term.startswith("R-") and pd.notna(value):
                coef_rows.append(
                    {
                        "source": source_id,
                        "source_label": source_label,
                        "model": model_id,
                        "model_label": MODEL_LABELS[model_id],
                        "term": term,
                        "coefficient": value,
                    }
                )
        stats["model_label"] = MODEL_LABELS[model_id]
        fit_rows.append(stats)
    return pd.DataFrame(fit_rows), pd.DataFrame(coef_rows)


def predict_hires(weather: pd.DataFrame, intercept, numeric_coefs, day_coefs):
    """Apply Intercept + numeric terms + day-of-week coefficient."""
    predicted = []
    for _, row in weather.iterrows():
        yhat = intercept
        for term, coef in numeric_coefs.items():
            value = row.get(term)
            if pd.isna(value):
                value = 0.0
            yhat += coef * float(value)
        day_key = f"day_{row['day_of_week']}"
        yhat += day_coefs.get(day_key, 0.0)
        predicted.append(yhat)

    out = weather.copy()
    out["predicted_hires"] = predicted
    return out


DAY_TERM = re.compile(r"^C\(day_of_week\)\[T\.(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\]$")
MONTH_TERM = re.compile(r"^C\(month\)\[T\.(\d+)\]$")
MONTH_NAME_TERM = re.compile(r"^C\(month_name\)\[T\.([A-Za-z]+)\]$")
YEAR_TERM = re.compile(r"^C\(year\)\[T\.(\d{4})\]$")
SEASON_TERM = re.compile(r"^C\(season_name\)\[T\.(Winter|Spring|Summer|Autumn)\]$")
SQUARED_TERM = re.compile(r"^I\((\w+) \*\* 2\)$")

TRAINING_MEANS = {
    "solarradiation": 109.9,
    "visibility": 22.8,
    "sealevelpressure": 1015.0,
}


def enrich_calendar(weather: pd.DataFrame) -> pd.DataFrame:
    """Add month/year fields and fill M2 extras Open-Meteo may omit."""
    out = weather.copy()
    dates = pd.to_datetime(out["date"])
    out["year"] = dates.dt.year.astype(int)
    out["month"] = dates.dt.month.astype(int)
    out["month_name"] = dates.dt.strftime("%b")
    for col, mean in TRAINING_MEANS.items():
        if col not in out.columns:
            out[col] = mean
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(mean)
    return out


def apply_pat_model(weather: pd.DataFrame, coefs: dict, year_fallback: int | None = 2025):
    """Score a statsmodels-style formula (C(day), C(month), C(year), I(x**2))."""
    dummy_years = {int(m.group(1)) for term in coefs if (m := YEAR_TERM.match(term))}
    intercept = float(coefs.get("Intercept", 0.0))
    predicted = []
    for _, row in weather.iterrows():
        yhat = intercept
        year = int(row.get("year", 0) or 0)
        if dummy_years and year not in dummy_years:
            year = year_fallback if year_fallback in dummy_years else max(dummy_years)
        for term, coef in coefs.items():
            if term == "Intercept" or pd.isna(coef):
                continue
            match = DAY_TERM.match(term)
            if match:
                if str(row.get("day_of_week")) == match.group(1):
                    yhat += coef
                continue
            match = MONTH_TERM.match(term)
            if match:
                if int(row.get("month", 0) or 0) == int(match.group(1)):
                    yhat += coef
                continue
            match = MONTH_NAME_TERM.match(term)
            if match:
                if str(row.get("month_name")) == match.group(1):
                    yhat += coef
                continue
            match = YEAR_TERM.match(term)
            if match:
                if year == int(match.group(1)):
                    yhat += coef
                continue
            match = SEASON_TERM.match(term)
            if match:
                if str(row.get("season_name")) == match.group(1):
                    yhat += coef
                continue
            if term == "C(post_2023)[T.True]":
                if int(row.get("year", 0) or 0) >= 2023:
                    yhat += coef
                continue
            match = SQUARED_TERM.match(term)
            if match:
                value = row.get(match.group(1), 0.0)
                yhat += coef * float(0.0 if pd.isna(value) else value) ** 2
                continue
            value = row.get(term)
            if pd.isna(value):
                value = TRAINING_MEANS.get(term, 0.0)
            yhat += coef * float(value)
        predicted.append(yhat)
    out = weather.copy()
    out["predicted_hires"] = predicted
    return out


def dashboard_in_sample(bike, intercept, numeric_coefs, day_coefs):
    """In-sample R² / RMSE for the deployed dashboard model (2014+)."""
    df = bike.copy()
    if "date" in df.columns:
        df = df[df["date"] >= pd.Timestamp("2014-01-01", tz="UTC")]
    needed = ["bikes_hired", "day_of_week", *numeric_coefs.keys()]
    df = df.dropna(subset=[c for c in needed if c in df.columns])
    if df.empty:
        return None
    yhat = np.full(len(df), intercept, dtype=float)
    for term, coef in numeric_coefs.items():
        yhat += coef * df[term].astype(float).to_numpy()
    day_map = {key.replace("day_", ""): val for key, val in day_coefs.items()}
    yhat += df["day_of_week"].astype(str).map(day_map).fillna(0).to_numpy()
    y = df["bikes_hired"].astype(float).to_numpy()
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return {
        "source": "dashboard",
        "source_label": "Dashboard model",
        "model": "dashboard",
        "model_label": "Deployed (temp + humidity + weekday)",
        "r2": r2,
        "adj_r2": r2,
        "aic": np.nan,
        "n": float(len(df)),
        "residual_se": rmse,
    }


def fetch_weather(kind: str):
    """Fetch Open-Meteo history or forecast; never raise to the caller."""
    try:
        if kind == "history":
            df = open_meteo_history("London", "2026-01-01", "2026-01-07")
        else:
            df = open_meteo("London", 5)
        return df, None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def style_figure(fig, title):
    fig.update_layout(
        title=title,
        template="plotly_white",
        legend_title_text="",
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif"},
        hovermode="closest",
    )
    return fig


def ols_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 60)
    return xs, intercept + slope * xs, float(slope), float(intercept), float(r2)


def add_trendline(fig, x, y, name, color, dash="solid"):
    fit = ols_fit(x, y)
    if fit is None:
        return None
    xs, ys, slope, intercept, r2 = fit
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=name,
            line={"color": color, "width": 3, "dash": dash},
            hovertemplate=(
                f"{name}<br>y = {intercept:,.0f} + {slope:,.1f}x"
                f"<br>R² = {r2:.3f}<extra></extra>"
            ),
        )
    )
    return slope, intercept, r2


def prediction_table(df: pd.DataFrame):
    if df.empty:
        return html.Div("No predictions available.", className="notice")

    display = df.copy()
    display["date"] = pd.to_datetime(display["date"]).dt.strftime("%Y-%m-%d")
    display["predicted_hires"] = display["predicted_hires"].round(0).astype(int)
    for col in [
        "temp",
        "humidity",
        "precip",
        "windspeed",
        "cloudcover",
        "solarradiation",
        "visibility",
    ]:
        if col in display.columns:
            display[col] = display[col].round(1)

    columns = [
        {"name": "Date", "id": "date"},
        {"name": "Day", "id": "day_of_week"},
        {"name": "Temp (°C)", "id": "temp"},
        {"name": "Humidity (%)", "id": "humidity"},
        {"name": "Precip (mm)", "id": "precip"},
        {"name": "Wind (km/h)", "id": "windspeed"},
        {"name": "Cloud (%)", "id": "cloudcover"},
        {"name": "Solar (W/m²)", "id": "solarradiation"},
        {"name": "Visibility (km)", "id": "visibility"},
        {"name": "Predicted hires", "id": "predicted_hires"},
    ]
    visible = [c["id"] for c in columns if c["id"] in display.columns]
    return dash_table.DataTable(
        data=display[visible].to_dict("records"),
        columns=[c for c in columns if c["id"] in visible],
        page_size=10,
        **TABLE_STYLE,
    )


def prediction_bar(df: pd.DataFrame, title: str):
    if df.empty:
        return EMPTY_FIG
    plot_df = df.copy()
    plot_df["date_label"] = pd.to_datetime(plot_df["date"]).dt.strftime("%a %d %b")
    fig = px.bar(
        plot_df,
        x="date_label",
        y="predicted_hires",
        labels={"date_label": "Date", "predicted_hires": "Predicted bikes hired"},
        color_discrete_sequence=["#2e86ab"],
    )
    fig.update_traces(
        hovertemplate="%{x}<br>Predicted hires: %{y:,.0f}<extra></extra>"
    )
    return style_figure(fig, title)


BIKES, BIKES_ERROR = load_bikes()
try:
    INTERCEPT, NUMERIC_COEFS, DAY_COEFS = load_coefficients()
    COEF_ERROR = None
except Exception as exc:
    INTERCEPT, NUMERIC_COEFS, DAY_COEFS = 0.0, {}, {}
    COEF_ERROR = f"Could not load model_coefficients.csv: {exc}"

try:
    FIT_A, COEF_A = parse_model_table(
        ROOT / "model_coff2.csv", "A", "Table A · model_coff2"
    )
    FIT_B, COEF_B = parse_model_table(
        ROOT / "model_coff3.csv", "B", "Table B · model_coff3"
    )
    FIT_ALL = pd.concat([FIT_A, FIT_B], ignore_index=True)
    COEF_ALL = pd.concat([COEF_A, COEF_B], ignore_index=True)
    COMPARE_ERROR = None
except Exception as exc:
    FIT_ALL, COEF_ALL = pd.DataFrame(), pd.DataFrame()
    COMPARE_ERROR = f"Could not load the comparison tables: {exc}"

M2_COEFS = {}
if not COEF_ALL.empty:
    m2 = COEF_ALL[(COEF_ALL["source"] == "B") & (COEF_ALL["model"] == "M2_weekday")]
    M2_COEFS = dict(zip(m2["term"], m2["coefficient"]))
M2_ERROR = None if M2_COEFS else "Could not load Table B M2 coefficients."

DASHBOARD_FIT = (
    dashboard_in_sample(BIKES, INTERCEPT, NUMERIC_COEFS, DAY_COEFS)
    if not BIKES.empty and not COEF_ERROR
    else None
)

YEAR_MIN = int(BIKES["year"].min()) if not BIKES.empty and "year" in BIKES.columns else 2010
YEAR_MAX = int(BIKES["year"].max()) if not BIKES.empty and "year" in BIKES.columns else 2026

_WEATHER_CACHE = {}


def get_weather(kind: str):
    """Fetch and cache Open-Meteo history or forecast."""
    if kind not in _WEATHER_CACHE:
        weather, error = fetch_weather(kind)
        if error is None and not weather.empty:
            weather = enrich_calendar(weather)
        _WEATHER_CACHE[kind] = (weather, error)
    return _WEATHER_CACHE[kind]


def weather_predictions(kind: str, model: str = "m2"):
    """Return (predictions_df, error_message) for history or forecast."""
    weather, error = get_weather(kind)
    if error:
        return pd.DataFrame(), error
    if model == "dashboard":
        if COEF_ERROR:
            return pd.DataFrame(), COEF_ERROR
        return predict_hires(weather, INTERCEPT, NUMERIC_COEFS, DAY_COEFS), None
    if M2_ERROR:
        return pd.DataFrame(), M2_ERROR
    return apply_pat_model(weather, M2_COEFS), None


app = Dash(__name__)
app.title = "London Bike Hires"
server = app.server

app.layout = html.Div(
    [
        html.Header(
            [
                html.H1([fa("bicycle", "hero-icon"), " London Bike Hires"]),
                html.P(
                    "Explore how weather and the day of week shape daily Santander "
                    "Cycle hires, predict with the deployed linear model, and compare "
                    "two alternative nested-model specifications."
                ),
            ],
            className="hero",
        ),
        dcc.Tabs(
            id="tabs",
            value="explore",
            children=[
                dcc.Tab(
                    label="Explore the data",
                    value="explore",
                    className="tab",
                    selected_className="tab--selected",
                    children=html.Div(
                        [
                            html.Div(id="explore-banner"),
                            MODEL_EQUATIONS,
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label(
                                                [fa("cloud-sun"), " Weather variable"]
                                            ),
                                            dcc.Dropdown(
                                                id="weather-var",
                                                options=WEATHER_OPTIONS,
                                                value="temp",
                                                clearable=False,
                                            ),
                                        ],
                                        className="control",
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                [fa("palette"), " Colour points by"]
                                            ),
                                            dcc.Dropdown(
                                                id="colour-by",
                                                options=COLOUR_OPTIONS,
                                                value="weekend",
                                                clearable=False,
                                            ),
                                        ],
                                        className="control",
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                [fa("chart-line"), " Regression line"]
                                            ),
                                            dcc.RadioItems(
                                                id="trendline-mode",
                                                options=[
                                                    {"label": " Off", "value": "off"},
                                                    {
                                                        "label": " Overall OLS",
                                                        "value": "overall",
                                                    },
                                                    {
                                                        "label": " Line per group",
                                                        "value": "group",
                                                    },
                                                ],
                                                value="overall",
                                                inline=True,
                                                className="radio-row",
                                            ),
                                        ],
                                        className="control",
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                [fa("calendar-week"), " Day type"]
                                            ),
                                            dcc.RadioItems(
                                                id="weekend-filter",
                                                options=[
                                                    {"label": " All", "value": "all"},
                                                    {
                                                        "label": " Weekdays",
                                                        "value": "Weekday",
                                                    },
                                                    {
                                                        "label": " Weekends",
                                                        "value": "Weekend",
                                                    },
                                                ],
                                                value="all",
                                                inline=True,
                                                className="radio-row",
                                            ),
                                        ],
                                        className="control",
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                [fa("leaf"), " Seasons"]
                                            ),
                                            dcc.Checklist(
                                                id="season-filter",
                                                options=[
                                                    {"label": f" {s}", "value": s}
                                                    for s in SEASON_ORDER
                                                ],
                                                value=list(SEASON_ORDER),
                                                inline=True,
                                                className="radio-row",
                                            ),
                                        ],
                                        className="control control--wide",
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                [fa("clock"), " Years"]
                                            ),
                                            dcc.RangeSlider(
                                                id="year-range",
                                                min=YEAR_MIN,
                                                max=YEAR_MAX,
                                                step=1,
                                                value=[YEAR_MIN, YEAR_MAX],
                                                marks={
                                                    y: str(y)
                                                    for y in range(
                                                        YEAR_MIN, YEAR_MAX + 1, 3
                                                    )
                                                },
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": False,
                                                },
                                            ),
                                        ],
                                        className="control control--wide",
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                [fa("circle-half-stroke"), " Point opacity"]
                                            ),
                                            dcc.Slider(
                                                id="opacity",
                                                min=0.15,
                                                max=0.9,
                                                step=0.05,
                                                value=0.45,
                                                marks={
                                                    0.15: "light",
                                                    0.45: "mid",
                                                    0.9: "solid",
                                                },
                                            ),
                                        ],
                                        className="control",
                                    ),
                                ],
                                className="controls card card--controls",
                            ),
                            html.Div(id="explore-kpis", className="kpi-grid"),
                            dcc.Loading(
                                dcc.Graph(id="scatter-plot"), type="circle"
                            ),
                            html.Div(id="point-detail", className="point-detail"),
                            dcc.Graph(id="dow-bar"),
                        ],
                        className="panel",
                    ),
                ),
                dcc.Tab(
                    label="Predict",
                    value="predict",
                    className="tab",
                    selected_className="tab--selected",
                    children=html.Div(
                        [
                            html.P(
                                [
                                    fa("wand-magic-sparkles"),
                                    " Predictions default to Table B’s M2 (weekday + year/month "
                                    "weather model). Open-Meteo supplies temperature, humidity, "
                                    "rain, wind, cloud, solar radiation and visibility; 2026 uses "
                                    "the 2025 year effect because that dummy is the latest in the "
                                    "fit. The simpler dashboard model remains available below.",
                                ],
                                className="lede",
                            ),
                            html.Div(
                                [
                                    html.Label([fa("calculator"), " Prediction model"]),
                                    dcc.RadioItems(
                                        id="predict-model",
                                        options=[
                                            {
                                                "label": " Table B · M2 (weekday + extras)",
                                                "value": "m2",
                                            },
                                            {
                                                "label": " Dashboard model (temp + humidity + weekday)",
                                                "value": "dashboard",
                                            },
                                        ],
                                        value="m2",
                                        inline=True,
                                        className="radio-row",
                                    ),
                                ],
                                className="card card--controls",
                            ),
                            html.Div(id="predict-banner"),
                            html.Div(
                                [
                                    html.H2(
                                        [fa("clock-rotate-left"), " First week of January 2026"]
                                    ),
                                    html.P(
                                        "Historical archive for London, 1–7 January 2026."
                                    ),
                                    html.Div(id="history-table"),
                                    dcc.Graph(id="history-bar"),
                                ],
                                className="card",
                            ),
                            html.Div(
                                [
                                    html.H2(
                                        [fa("cloud-sun-rain"), " Next five days"]
                                    ),
                                    html.P("Live Open-Meteo forecast for London."),
                                    html.Div(id="forecast-table"),
                                    dcc.Graph(id="forecast-bar"),
                                ],
                                className="card",
                            ),
                        ],
                        className="panel",
                    ),
                ),
                dcc.Tab(
                    label="Compare models",
                    value="compare",
                    className="tab",
                    selected_className="tab--selected",
                    children=html.Div(
                        [
                            html.P(
                                [
                                    fa("scale-balanced"),
                                    " Table A (`model_coff2.csv`) and Table B "
                                    "(`model_coff3.csv`) are two nested-model ladders. "
                                    "M1 is temperature only; M3 adds core weather and "
                                    "weekdays; M2 and M4 add calendar structure "
                                    "(months/years vs seasons and daylight). The "
                                    "deployed dashboard model is shown as a reference "
                                    "using in-sample fit on 2014+ data.",
                                ],
                                className="lede",
                            ),
                            html.Div(
                                [
                                    html.Label([fa("layer-group"), " Show"]),
                                    dcc.RadioItems(
                                        id="compare-source",
                                        options=[
                                            {"label": " Table A", "value": "A"},
                                            {"label": " Table B", "value": "B"},
                                            {"label": " Both tables", "value": "both"},
                                        ],
                                        value="both",
                                        inline=True,
                                        className="radio-row",
                                    ),
                                ],
                                className="card card--controls",
                            ),
                            html.Div(id="compare-kpis", className="kpi-grid"),
                            html.Div(id="compare-note", className="compare-note"),
                            dcc.Graph(id="compare-fit-fig"),
                            dcc.Graph(id="compare-aic-fig"),
                            dcc.Graph(id="compare-coef-fig"),
                            html.Div(id="compare-table"),
                        ],
                        className="panel",
                    ),
                ),
            ],
        ),
    ]
)


@app.callback(
    Output("explore-banner", "children"),
    Output("explore-kpis", "children"),
    Output("scatter-plot", "figure"),
    Output("dow-bar", "figure"),
    Input("weather-var", "value"),
    Input("colour-by", "value"),
    Input("trendline-mode", "value"),
    Input("weekend-filter", "value"),
    Input("season-filter", "value"),
    Input("year-range", "value"),
    Input("opacity", "value"),
)
def update_explore(
    weather_var,
    colour_by,
    trendline_mode,
    weekend_filter,
    seasons,
    year_range,
    opacity,
):
    if BIKES_ERROR:
        banner = html.Div(BIKES_ERROR, className="banner banner--error")
        return banner, [], EMPTY_FIG, EMPTY_FIG

    plot_df = BIKES.copy()
    if year_range and "year" in plot_df.columns:
        lo, hi = year_range
        plot_df = plot_df[plot_df["year"].between(lo, hi)]
    if seasons:
        plot_df = plot_df[plot_df["season_name"].isin(seasons)]
    if weekend_filter and weekend_filter != "all":
        plot_df = plot_df[plot_df["weekend_label"] == weekend_filter]

    needed = ["bikes_hired", weather_var, colour_by]
    plot_df = plot_df.dropna(subset=[c for c in needed if c in plot_df.columns]).copy()

    n = len(plot_df)
    if n == 0:
        kpis = [kpi_card("ban", "0", "Days in view", "Widen the filters")]
        return None, kpis, EMPTY_FIG, EMPTY_FIG

    mean_hires = plot_df["bikes_hired"].mean()
    weekday_mean = plot_df.loc[
        plot_df["weekend_label"] == "Weekday", "bikes_hired"
    ].mean()
    weekend_mean = plot_df.loc[
        plot_df["weekend_label"] == "Weekend", "bikes_hired"
    ].mean()
    weekend_gap = (
        (weekend_mean - weekday_mean) / weekday_mean * 100
        if pd.notna(weekday_mean) and weekday_mean
        else np.nan
    )
    fit = ols_fit(plot_df[weather_var], plot_df["bikes_hired"])
    slope_txt = f"{fit[2]:,.0f}" if fit else "—"
    r2_txt = f"{fit[4]:.2f}" if fit else "—"

    if weekend_filter and weekend_filter != "all":
        weekend_kpi = kpi_card(
            "filter",
            f"{weekend_filter}s",
            "Day-type filter",
            "Weekend gap is hidden while filtered",
        )
    else:
        weekend_kpi = kpi_card(
            "moon",
            f"{weekend_gap:+.0f}%" if pd.notna(weekend_gap) else "—",
            "Weekend vs weekday",
            "Negative means fewer weekend hires",
        )

    kpis = [
        kpi_card("calendar-days", f"{n:,}", "Days in view"),
        kpi_card("bicycle", f"{mean_hires:,.0f}", "Mean daily hires"),
        weekend_kpi,
        kpi_card(
            WEATHER_ICONS.get(weather_var, "chart-line"),
            slope_txt,
            f"OLS slope vs {WEATHER_LABELS.get(weather_var, weather_var)}",
            f"R² = {r2_txt}",
        ),
    ]

    if colour_by == "weekend":
        plot_df["colour"] = plot_df["weekend_label"]
        colour_order = ["Weekday", "Weekend"]
        colour_title = "Weekend"
    else:
        plot_df["colour"] = plot_df["season_name"].astype(str)
        colour_order = [s for s in SEASON_ORDER if s in set(plot_df["colour"])]
        colour_title = "Season"

    scatter = px.scatter(
        plot_df,
        x=weather_var,
        y="bikes_hired",
        color="colour",
        opacity=opacity or 0.45,
        hover_data={"date_label": True, "day_of_week": True, "colour": False},
        category_orders={"colour": colour_order},
        color_discrete_sequence=SET2,
        labels={
            weather_var: WEATHER_LABELS.get(weather_var, weather_var),
            "bikes_hired": "Bikes hired",
            "date_label": "Date",
            "day_of_week": "Day",
            "colour": colour_title,
        },
    )
    scatter.update_traces(marker={"size": 8})

    if trendline_mode == "overall":
        add_trendline(
            scatter,
            plot_df[weather_var],
            plot_df["bikes_hired"],
            "OLS fit",
            "#1b365d",
        )
    elif trendline_mode == "group":
        for i, group in enumerate(colour_order):
            subset = plot_df[plot_df["colour"] == group]
            add_trendline(
                scatter,
                subset[weather_var],
                subset["bikes_hired"],
                f"OLS · {group}",
                SET2[i % len(SET2)],
            )

    weather_label = WEATHER_LABELS.get(weather_var, weather_var)
    scatter = style_figure(scatter, f"Daily bikes hired vs {weather_label}")

    dow = (
        plot_df.dropna(subset=["bikes_hired", "day_of_week"])
        .groupby("day_of_week", observed=True)["bikes_hired"]
        .mean()
        .reindex(DAY_ORDER)
        .reset_index()
    )
    bar = px.bar(
        dow,
        x="day_of_week",
        y="bikes_hired",
        labels={"day_of_week": "Day of week", "bikes_hired": "Average bikes hired"},
        color_discrete_sequence=["#1b365d"],
    )
    bar.update_traces(
        hovertemplate="%{x}<br>Average hires: %{y:,.0f}<extra></extra>"
    )
    bar = style_figure(bar, "Average daily hires by day of week")
    return None, kpis, scatter, bar


@app.callback(
    Output("point-detail", "children"),
    Input("scatter-plot", "clickData"),
    Input("weather-var", "value"),
)
def update_point_detail(click_data, weather_var):
    weather_label = WEATHER_LABELS.get(weather_var, weather_var)
    if not click_data or not click_data.get("points"):
        return html.Span(
            [fa("arrow-pointer"), " Click a point to inspect that day."],
            className="muted",
        )
    pt = click_data["points"][0]
    hover = pt.get("customdata") or []
    date_txt = hover[0] if len(hover) > 0 else ""
    day_txt = hover[1] if len(hover) > 1 else ""
    return html.Div(
        [
            fa("location-dot"),
            f" {date_txt} ({day_txt}): {pt['y']:,.0f} hires when "
            f"{weather_label.split(' (')[0].lower()} was {pt['x']:.1f}.",
        ],
        className="point-detail-inner",
    )


@app.callback(
    Output("predict-banner", "children"),
    Output("history-table", "children"),
    Output("history-bar", "figure"),
    Output("forecast-table", "children"),
    Output("forecast-bar", "figure"),
    Input("tabs", "value"),
    Input("predict-model", "value"),
)
def update_predict(_tab, model):
    model = model or "m2"
    history_preds, history_error = weather_predictions("history", model)
    forecast_preds, forecast_error = weather_predictions("forecast", model)

    messages = []
    if model == "m2" and M2_ERROR:
        messages.append(M2_ERROR)
    if model == "dashboard" and COEF_ERROR:
        messages.append(COEF_ERROR)
    if history_error and history_error not in messages:
        messages.append(
            "January 2026 weather could not be loaded. "
            "The rest of the dashboard is still available."
        )
    if forecast_error and forecast_error not in messages:
        messages.append(
            "The five-day forecast could not be loaded. "
            "The rest of the dashboard is still available."
        )

    banner = (
        html.Div(
            [html.Div(msg, className="banner-line") for msg in messages],
            className="banner banner--error",
        )
        if messages
        else None
    )
    label = "M2" if model == "m2" else "dashboard model"
    return (
        banner,
        prediction_table(history_preds),
        prediction_bar(
            history_preds, f"Predicted bikes hired, 1–7 January 2026 ({label})"
        ),
        prediction_table(forecast_preds),
        prediction_bar(forecast_preds, f"Predicted bikes hired, next five days ({label})"),
    )


@app.callback(
    Output("compare-kpis", "children"),
    Output("compare-note", "children"),
    Output("compare-fit-fig", "figure"),
    Output("compare-aic-fig", "figure"),
    Output("compare-coef-fig", "figure"),
    Output("compare-table", "children"),
    Input("compare-source", "value"),
)
def update_compare(source):
    if COMPARE_ERROR:
        note = html.Div(COMPARE_ERROR, className="banner banner--error")
        return [], note, EMPTY_FIG, EMPTY_FIG, EMPTY_FIG, None

    fit = FIT_ALL.copy()
    coefs = COEF_ALL.copy()
    if source in {"A", "B"}:
        fit = fit[fit["source"] == source]
        coefs = coefs[coefs["source"] == source]

    if fit.empty:
        return [], html.Div("No comparison data."), EMPTY_FIG, EMPTY_FIG, EMPTY_FIG, None

    best_r2 = fit.sort_values("adj_r2", ascending=False).iloc[0]
    best_aic = fit.dropna(subset=["aic"]).sort_values("aic").iloc[0]
    best_se = fit.sort_values("residual_se").iloc[0]

    kpis = [
        kpi_card(
            "trophy",
            best_r2["model_label"].split("·")[0].strip(),
            "Highest adjusted R²",
            f"{best_r2['source_label']}: {best_r2['adj_r2']:.2f}",
        ),
        kpi_card(
            "arrow-trend-down",
            best_aic["model_label"].split("·")[0].strip(),
            "Lowest AIC",
            f"{best_aic['source_label']}: {best_aic['aic']:,.0f}",
        ),
        kpi_card(
            "bullseye",
            best_se["model_label"].split("·")[0].strip(),
            "Lowest residual SE",
            f"{best_se['source_label']}: {best_se['residual_se']:,.0f} hires",
        ),
        kpi_card(
            "database",
            f"{int(fit['n'].dropna().iloc[0]):,}" if fit["n"].notna().any() else "—",
            "Sample size",
            "Same N across nested models",
        ),
    ]

    note = html.Div(
        [
            fa("lightbulb"),
            f" Best in-sample description in this view is {best_aic['model_label']} "
            f"from {best_aic['source_label']} (AIC {best_aic['aic']:,.0f}, "
            f"adj. R² {best_aic['adj_r2']:.2f}). Table A’s M2 leans on month and "
            "pressure terms; Table B’s M2 adds year dummies. M4 in both files "
            "uses season, daylight and a post-2023 shift. The dashboard model is "
            "kept small (temp, humidity, weekday) so it can score live weather.",
        ]
    )

    plot_fit = fit.copy()
    if DASHBOARD_FIT and source == "both":
        plot_fit = pd.concat([plot_fit, pd.DataFrame([DASHBOARD_FIT])], ignore_index=True)

    r2_fig = px.bar(
        plot_fit.sort_values("adj_r2"),
        x="adj_r2",
        y="model_label",
        color="source_label",
        orientation="h",
        barmode="group",
        color_discrete_map={
            "Table A · model_coff2": PALETTE["A"],
            "Table B · model_coff3": PALETTE["B"],
            "Dashboard model": PALETTE["dashboard"],
        },
        labels={"adj_r2": "Adjusted R²", "model_label": "Model", "source_label": ""},
    )
    r2_fig.update_traces(hovertemplate="%{y}<br>Adj. R² = %{x:.3f}<extra>%{fullData.name}</extra>")
    r2_fig = style_figure(r2_fig, "In-sample adjusted R²")

    aic_df = fit.dropna(subset=["aic"])
    aic_fig = px.bar(
        aic_df.sort_values("aic", ascending=False),
        x="aic",
        y="model_label",
        color="source_label",
        orientation="h",
        barmode="group",
        color_discrete_map={
            "Table A · model_coff2": PALETTE["A"],
            "Table B · model_coff3": PALETTE["B"],
        },
        labels={"aic": "AIC (lower is better)", "model_label": "Model", "source_label": ""},
    )
    aic_fig.update_traces(hovertemplate="%{y}<br>AIC = %{x:,.0f}<extra>%{fullData.name}</extra>")
    aic_fig = style_figure(aic_fig, "AIC — lower is a better in-sample fit")

    focus = coefs[coefs["term"].isin(COEF_FOCUS)].copy()
    if focus.empty:
        coef_fig = EMPTY_FIG
    else:
        coef_fig = px.bar(
            focus,
            x="coefficient",
            y="term",
            color="model_label",
            facet_col="source_label" if source == "both" else None,
            orientation="h",
            barmode="group",
            labels={
                "coefficient": "Coefficient (hires per unit)",
                "term": "Weather term",
                "model_label": "Model",
            },
        )
        coef_fig.update_traces(
            hovertemplate="%{y}: %{x:,.1f}<extra>%{fullData.name}</extra>"
        )
        coef_fig = style_figure(
            coef_fig, "Weather coefficients across nested models"
        )
        coef_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    table_df = fit[
        ["source_label", "model_label", "adj_r2", "r2", "aic", "residual_se", "n"]
    ].copy()
    table_df = table_df.rename(
        columns={
            "source_label": "Table",
            "model_label": "Model",
            "adj_r2": "Adj. R²",
            "r2": "R² (rounded)",
            "aic": "AIC",
            "residual_se": "Residual SE",
            "n": "N",
        }
    )
    for col in ["Adj. R²", "R² (rounded)"]:
        table_df[col] = table_df[col].round(3)
    table_df["AIC"] = table_df["AIC"].round(0)
    table_df["Residual SE"] = table_df["Residual SE"].round(0)
    table_df["N"] = table_df["N"].round(0)
    table = dash_table.DataTable(
        data=table_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in table_df.columns],
        sort_action="native",
        **TABLE_STYLE,
    )
    return kpis, note, r2_fig, aic_fig, coef_fig, table


app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
        <style>
            body {
                margin: 0;
                background: #f4f6f8;
                color: #1b242c;
                font-family: system-ui, -apple-system, Segoe UI, sans-serif;
            }
            .hero {
                background: #1b365d;
                color: white;
                padding: 28px 32px 24px;
            }
            .hero h1 { margin: 0 0 8px; font-size: 28px; font-weight: 650; }
            .hero p { margin: 0; max-width: 820px; opacity: 0.9; line-height: 1.45; }
            .hero-icon { margin-right: 8px; }
            .tab { padding: 12px 18px; font-weight: 600; }
            .tab--selected { border-top: 3px solid #2e86ab !important; }
            .panel { padding: 20px 28px 36px; max-width: 1140px; margin: 0 auto; }
            .controls { display: flex; gap: 18px; flex-wrap: wrap; }
            .control { min-width: 220px; flex: 1; }
            .control--wide { min-width: 100%; flex-basis: 100%; }
            .control label {
                display: block;
                font-size: 13px;
                font-weight: 600;
                margin-bottom: 6px;
                color: #3d4a54;
            }
            .control label .fa-solid { margin-right: 6px; color: #2e86ab; }
            .radio-row label { font-weight: 500; margin-right: 12px; }
            .lede { color: #3d4a54; line-height: 1.55; }
            .lede .fa-solid { margin-right: 8px; color: #2e86ab; }
            .card {
                background: white;
                border-radius: 10px;
                padding: 20px 22px 12px;
                margin: 18px 0;
                box-shadow: 0 1px 3px rgba(27, 54, 93, 0.08);
            }
            .card--controls { padding-bottom: 18px; }
            .card h2 { margin: 0 0 4px; font-size: 20px; color: #1b365d; }
            .card h2 .fa-solid { margin-right: 8px; color: #2e86ab; }
            .card p { margin: 0 0 14px; color: #5b6770; }
            .banner {
                border-radius: 8px;
                padding: 12px 14px;
                margin-bottom: 16px;
            }
            .banner--error { background: #fdecea; color: #8a1f11; }
            .notice { color: #5b6770; padding: 8px 0 16px; }
            .section-title {
                font-size: 18px;
                color: #1b365d;
                margin: 4px 0 12px;
            }
            .section-title .fa-solid { margin-right: 8px; color: #2e86ab; }
            .equation-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 12px;
                margin-bottom: 16px;
            }
            .eq-card {
                background: white;
                border-radius: 10px;
                padding: 14px 16px 12px;
                box-shadow: 0 1px 3px rgba(27, 54, 93, 0.08);
                border-left: 4px solid #2e86ab;
            }
            .eq-head {
                display: flex;
                justify-content: space-between;
                align-items: baseline;
                gap: 10px;
                flex-wrap: wrap;
                margin-bottom: 8px;
            }
            .eq-title { font-weight: 700; color: #1b365d; }
            .eq-title .fa-solid { margin-right: 6px; color: #2e86ab; }
            .eq-badge {
                font-size: 11px;
                font-weight: 600;
                color: #2e86ab;
                background: #eef6fb;
                padding: 3px 8px;
                border-radius: 999px;
            }
            .eq-card p { margin: 0; }
            .eq-math {
                font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
                font-size: 13.5px;
                line-height: 1.55;
                color: #1b365d;
            }
            .eq-note { font-size: 12px; color: #5b6770; margin-top: 8px !important; }
            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px;
                margin: 8px 0 16px;
            }
            .kpi {
                background: white;
                border-radius: 10px;
                padding: 14px 16px;
                box-shadow: 0 1px 3px rgba(27, 54, 93, 0.08);
            }
            .kpi-icon { color: #2e86ab; font-size: 18px; margin-bottom: 6px; }
            .kpi-value { font-size: 26px; font-weight: 700; color: #1b365d; }
            .kpi-label { font-size: 13px; color: #3d4a54; margin-top: 2px; }
            .kpi-hint { font-size: 11px; color: #7a8790; margin-top: 4px; }
            .point-detail { min-height: 24px; margin: 4px 0 8px; color: #3d4a54; }
            .point-detail-inner .fa-solid { margin-right: 6px; color: #2e86ab; }
            .muted { color: #7a8790; }
            .compare-note {
                background: #eef6fb;
                border-left: 4px solid #2e86ab;
                padding: 12px 14px;
                border-radius: 0 8px 8px 0;
                color: #1b365d;
                line-height: 1.5;
                margin-bottom: 12px;
            }
            .compare-note .fa-solid { margin-right: 8px; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)

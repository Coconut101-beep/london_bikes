"""London bikes dashboard: explore daily hires and predict with M4."""

from __future__ import annotations

import os
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
            "M4 · Season + daylight",
            className="section-title",
        ),
        equation_card(
            "rocket",
            "M4 model",
            "model_coefficients.csv",
            (
                "Intercept + 562.36 × temperature + (−183.65) × precipitation + "
                "105.53 × visibility + (−149.15) × wind speed + (−39.73) × precip cover + "
                "80.89 × sea-level pressure + 886.21 × UV index + 335.62 × daylight + "
                "a weekday effect + a season effect + a post-2023 shift"
            ),
            "Intercept = −64,093. Monday and Autumn are the baselines. post_2023 = −4,788 from 2023 onward.",
        ),
    ],
    className="equations",
)


def _weekend_label(value) -> str:
    text = str(value).strip().upper()
    if value is True or text in {"TRUE", "1", "WEEKEND"}:
        return "Weekend"
    return "Weekday"


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
        if "daylight" not in bike.columns and {"sunrise", "sunset"} <= set(bike.columns):
            sunrise = pd.to_datetime(bike["sunrise"], utc=True, errors="coerce")
            sunset = pd.to_datetime(bike["sunset"], utc=True, errors="coerce")
            bike["daylight"] = (sunset - sunrise).dt.total_seconds() / 3600.0
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


TRAINING_MEANS = {
    "visibility": 22.8,
    "sealevelpressure": 1015.0,
    "uvindex": 4.2,
    "daylight": 12.1,
    "precipcover": 9.1,
}


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def add_model_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Add season_* and post_2023 columns used by model_coefficients.csv."""
    out = df.copy()
    if "season_name" not in out.columns and "month" in out.columns:
        out["season_name"] = out["month"].map(season_from_month)
    if "season_name" in out.columns:
        for season in SEASON_ORDER:
            out[f"season_{season}"] = (
                out["season_name"].astype(str) == season
            ).astype(float)
    if "year" in out.columns:
        out["post_2023"] = (pd.to_numeric(out["year"], errors="coerce") >= 2023).astype(
            float
        )
    return out


def enrich_calendar(weather: pd.DataFrame) -> pd.DataFrame:
    """Add calendar dummies and fill extras Open-Meteo may omit."""
    out = weather.copy()
    dates = pd.to_datetime(out["date"])
    out["year"] = dates.dt.year.astype(int)
    out["month"] = dates.dt.month.astype(int)
    out["month_name"] = dates.dt.strftime("%b")
    out = add_model_dummies(out)
    for col, mean in TRAINING_MEANS.items():
        if col not in out.columns:
            out[col] = mean
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(mean)
    return out


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
        "uvindex",
        "daylight",
        "precipcover",
        "sealevelpressure",
    ]:
        if col in display.columns:
            display[col] = display[col].round(1)

    columns = [
        {"name": "Date", "id": "date"},
        {"name": "Day", "id": "day_of_week"},
        {"name": "Temp (°C)", "id": "temp"},
        {"name": "Precip (mm)", "id": "precip"},
        {"name": "Wind (km/h)", "id": "windspeed"},
        {"name": "Visibility (km)", "id": "visibility"},
        {"name": "UV", "id": "uvindex"},
        {"name": "Daylight (h)", "id": "daylight"},
        {"name": "Precip cover (%)", "id": "precipcover"},
        {"name": "Pressure (hPa)", "id": "sealevelpressure"},
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


def weather_predictions(kind: str):
    """Return (predictions_df, error_message) for history or forecast."""
    weather, error = get_weather(kind)
    if error:
        return pd.DataFrame(), error
    if COEF_ERROR:
        return pd.DataFrame(), COEF_ERROR
    return predict_hires(weather, INTERCEPT, NUMERIC_COEFS, DAY_COEFS), None


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
                    "Cycle hires, then predict with the M4 season-and-daylight model."
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
                                    " Predictions use the M4 season-and-daylight model in "
                                    "`model_coefficients.csv` (weather, daylight, UV, season, "
                                    "weekday, and a post-2023 shift). Open-Meteo supplies the "
                                    "weather inputs.",
                                ],
                                className="lede",
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
)
def update_predict(_tab):
    history_preds, history_error = weather_predictions("history")
    forecast_preds, forecast_error = weather_predictions("forecast")

    messages = []
    if COEF_ERROR:
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
    return (
        banner,
        prediction_table(history_preds),
        prediction_bar(
            history_preds, "Predicted bikes hired, 1–7 January 2026"
        ),
        prediction_table(forecast_preds),
        prediction_bar(forecast_preds, "Predicted bikes hired, next five days"),
    )


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

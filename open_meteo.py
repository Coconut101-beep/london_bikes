"""
open_meteo.py - fetch daily weather from Open-Meteo.

Open-Meteo (https://open-meteo.com) is free for non-commercial use and needs no
API key. This module has two functions, both returning the same tidy shape:

    from open_meteo import open_meteo, open_meteo_history

    open_meteo("London", 5)                              # next few days (forecast)
    open_meteo_history("London", "2026-01-01", "2026-01-07")  # a past date range

Both return a pandas DataFrame with one row per day and the columns:

    date, day_of_week, temp, humidity, precip, windspeed, cloudcover,
    solarradiation, visibility, sealevelpressure

Temperature is in degrees Celsius, wind in km/h, precipitation in mm, humidity
and cloud cover in percent, solar radiation in W/m² (24-hour mean), visibility
in km, and pressure in hPa.

The forecast reaches about 7 days ahead. For any date in the past (for example
the first week of January 2026) use open_meteo_history, which reads Open-Meteo's
historical archive.
"""

import pandas as pd
import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Hourly Open-Meteo field -> model column. Precipitation is summed over the
# day; everything else is a 24-hour mean. Visibility is returned in metres
# and converted to km to match the training data.
HOURLY_FIELDS = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "humidity",
    "precipitation": "precip",
    "wind_speed_10m": "windspeed",
    "cloud_cover": "cloudcover",
    "shortwave_radiation": "solarradiation",
    "visibility": "visibility",
    "pressure_msl": "sealevelpressure",
}

CORE_HOURLY_FIELDS = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "humidity",
    "precipitation": "precip",
    "wind_speed_10m": "windspeed",
    "cloud_cover": "cloudcover",
}


def geocode(location):
    """Turn a place name into (latitude, longitude, label) using Open-Meteo."""
    resp = requests.get(
        GEOCODE_URL,
        params={"name": location, "count": 1, "language": "en"},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        raise ValueError(f"Open-Meteo could not find a location called {location!r}.")
    top = results[0]
    label = ", ".join(p for p in [top.get("name"), top.get("country")] if p)
    return top["latitude"], top["longitude"], label


def _hourly_to_daily(hourly):
    """Aggregate an Open-Meteo hourly payload to one row per day."""
    available = {
        field: col
        for field, col in HOURLY_FIELDS.items()
        if field in hourly and hourly[field] is not None
    }
    frame = pd.DataFrame({col: hourly[field] for field, col in available.items()})
    frame["date"] = pd.to_datetime(hourly["time"]).normalize()

    agg = {}
    for col in frame.columns:
        if col == "date":
            continue
        agg[col] = "sum" if col == "precip" else "mean"
    daily = frame.groupby("date").agg(agg).reset_index()

    if "visibility" in daily.columns:
        # Open-Meteo visibility is metres; the bikes data uses kilometres.
        daily["visibility"] = daily["visibility"] / 1000.0

    daily.insert(1, "day_of_week", daily["date"].dt.strftime("%a"))
    return daily


def _fetch_hourly(url, extra_params, timeout):
    """Request the full hourly set; retry with core fields if extras are rejected."""
    params = {
        **extra_params,
        "hourly": ",".join(HOURLY_FIELDS),
        "timezone": "auto",
        "wind_speed_unit": "kmh",
    }
    resp = requests.get(url, params=params, timeout=timeout)
    if resp.status_code >= 400:
        params["hourly"] = ",".join(CORE_HOURLY_FIELDS)
        resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if "hourly" not in payload:
        raise ValueError("Open-Meteo response did not include hourly weather.")
    return payload["hourly"]


def open_meteo(location="London", days_to_forecast=5):
    """Return a daily weather forecast for a location as a tidy DataFrame."""
    days_to_forecast = int(days_to_forecast)
    if not 1 <= days_to_forecast <= 7:
        raise ValueError(
            "days_to_forecast must be between 1 and 7 (Open-Meteo's forecast limit)."
        )

    lat, lon, label = geocode(location)
    hourly = _fetch_hourly(
        FORECAST_URL,
        {"latitude": lat, "longitude": lon, "forecast_days": days_to_forecast},
        timeout=15,
    )
    daily = _hourly_to_daily(hourly)
    daily.attrs["location"] = label
    return daily


def open_meteo_history(location, start_date, end_date):
    """Return daily weather for a past date range from Open-Meteo's archive."""
    lat, lon, label = geocode(location)
    hourly = _fetch_hourly(
        ARCHIVE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
        },
        timeout=30,
    )
    daily = _hourly_to_daily(hourly)
    daily.attrs["location"] = label
    return daily


if __name__ == "__main__":
    print("Forecast (next 5 days):")
    print(open_meteo("London", 5).to_string(index=False))
    print("\nHistory (first week of January 2026):")
    print(open_meteo_history("London", "2026-01-01", "2026-01-07").to_string(index=False))

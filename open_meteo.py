"""
open_meteo.py - fetch daily weather from Open-Meteo.

Open-Meteo (https://open-meteo.com) is free for non-commercial use and needs no
API key. This module has two functions, both returning the same tidy shape:

    from open_meteo import open_meteo, open_meteo_history

    open_meteo("London", 5)                              # next few days (forecast)
    open_meteo_history("London", "2026-01-01", "2026-01-07")  # a past date range

Both return a pandas DataFrame with one row per day and the columns:

    date, day_of_week, season_name,
    temp, humidity, precip, windspeed, cloudcover, sealevelpressure, solarradiation

which line up with the model's predictors. Temperature is in degrees Celsius,
wind in km/h, precipitation in mm, humidity and cloud cover in percent, and sea
level pressure in hPa. Day of week and season come from the date itself.

Sea level pressure was added because the archive and the training file agree on
it closely: over 2024 they correlate at r = 0.999 and their means differ by
0.19 hPa. Solar radiation needed calibrating first, and is described where the
constants are. Visibility is the one field the model wanted and this module does
not supply: Open-Meteo's historical archive returns nothing for it at all, and
the archive is what the January panel reads.

The forecast reaches about 7 days ahead. For any date in the past (for example
the first week of January 2026) use open_meteo_history, which reads Open-Meteo's
historical archive.
"""

import time

import requests
import pandas as pd

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Forecast API daily field  ->  the column name our model uses
DAILY_FIELDS = {
    "temperature_2m_mean": "temp",
    "relative_humidity_2m_mean": "humidity",
    "precipitation_sum": "precip",
    "wind_speed_10m_mean": "windspeed",
    "cloud_cover_mean": "cloudcover",
    "pressure_msl_mean": "sealevelpressure",
}

# Solar radiation is the one field the two sources disagree on. Open-Meteo's
# shortwave radiation runs about 60% above the training file's solarradiation, so
# it is mapped onto the file's scale by a line fitted over the 1,096 days the two
# share: file = 1.951 + 0.6200 * open-meteo, R squared 0.822, mean absolute error
# 18 W/m2. Both are daily means of the hourly value, which is why the forecast
# asks for the hourly series and averages it here rather than taking the daily sum.
SOLAR_INTERCEPT = 1.951
SOLAR_SLOPE = 0.6200


def calibrate_solar(mean_shortwave):
    return SOLAR_INTERCEPT + SOLAR_SLOPE * mean_shortwave

# The archive API has no daily means, so we pull these hourly fields and
# aggregate them ourselves (mean for most, sum for precipitation).
HOURLY_FIELDS = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "humidity",
    "precipitation": "precip",
    "wind_speed_10m": "windspeed",
    "cloud_cover": "cloudcover",
    "pressure_msl": "sealevelpressure",
    "shortwave_radiation": "shortwave",
}

# The training file's own convention, read off the data: December to February is
# Winter, and the other three follow in three-month blocks.
SEASON_BY_MONTH = {12: "Winter", 1: "Winter", 2: "Winter",
                   3: "Spring", 4: "Spring", 5: "Spring",
                   6: "Summer", 7: "Summer", 8: "Summer",
                   9: "Autumn", 10: "Autumn", 11: "Autumn"}


def add_calendar(df):
    """Day of week and season, both of which the date already contains."""
    df.insert(1, "day_of_week", df["date"].dt.strftime("%a"))
    df.insert(2, "season_name", df["date"].dt.month.map(SEASON_BY_MONTH))
    return df


# Open-Meteo rate limits by IP, and a shared host such as Render reaches that
# limit on somebody else's traffic. Every answer below is therefore fetched once
# and kept: a place does not move, a past week does not change, and a forecast is
# worth re-reading once an hour rather than once a page load.
_CACHE = {}
FORECAST_TTL = 3600


def _get(url, params, timeout):
    """One request, retried through a 429 before it gives up."""
    for attempt in range(4):
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 429 and attempt < 3:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()


def geocode(location):
    """Turn a place name into (latitude, longitude, label) using Open-Meteo."""
    if ("geo", location) in _CACHE:
        return _CACHE[("geo", location)]
    resp = _get(GEOCODE_URL, {"name": location, "count": 1, "language": "en"}, 15)
    results = resp.json().get("results")
    if not results:
        raise ValueError(f"Open-Meteo could not find a location called {location!r}.")
    top = results[0]
    label = ", ".join(p for p in [top.get("name"), top.get("country")] if p)
    _CACHE[("geo", location)] = (top["latitude"], top["longitude"], label)
    return _CACHE[("geo", location)]


def open_meteo(location="London", days_to_forecast=5):
    """Return a daily weather forecast for a location as a tidy DataFrame.

    Args:
        location (str): a place name, e.g. "London" or "Paris".
        days_to_forecast (int): number of days ahead, from 1 to 7
            (Open-Meteo's forecast does not go beyond 7 days).

    Returns:
        pandas.DataFrame with columns date, day_of_week, season_name, temp,
        humidity, precip, windspeed, cloudcover, sealevelpressure. The resolved
        place name is stored in df.attrs["location"].
    """
    days_to_forecast = int(days_to_forecast)
    if not 1 <= days_to_forecast <= 7:
        raise ValueError("days_to_forecast must be between 1 and 7 (Open-Meteo's forecast limit).")

    lat, lon, label = geocode(location)

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY_FIELDS),
        "hourly": "shortwave_radiation",
        "forecast_days": days_to_forecast,
        "timezone": "auto",
        "wind_speed_unit": "kmh",   # matches the training data units
    }
    key = ("forecast", location, days_to_forecast)
    cached = _CACHE.get(key)
    if cached and time.time() - cached[0] < FORECAST_TTL:
        return cached[1].copy()

    resp = _get(FORECAST_URL, params, 15)
    daily = resp.json()["daily"]

    df = pd.DataFrame({col: daily[field] for field, col in DAILY_FIELDS.items()})
    df.insert(0, "date", pd.to_datetime(daily["time"]))

    hourly = resp.json()["hourly"]
    sw = pd.DataFrame({"shortwave": hourly["shortwave_radiation"]})
    sw["date"] = pd.to_datetime(hourly["time"]).normalize()
    df["solarradiation"] = calibrate_solar(
        df["date"].map(sw.groupby("date")["shortwave"].mean()))

    add_calendar(df)
    df.attrs["location"] = label
    _CACHE[key] = (time.time(), df)
    return df.copy()


def open_meteo_history(location, start_date, end_date):
    """Return daily weather for a past date range from Open-Meteo's archive.

    Use this for dates the forecast cannot reach, such as the first week of
    January 2026. The archive has no daily means, so we pull the hourly values
    and aggregate them to one row per day here.

    Args:
        location (str): a place name, e.g. "London".
        start_date (str): first day, "YYYY-MM-DD".
        end_date (str): last day, "YYYY-MM-DD".

    Returns:
        pandas.DataFrame with the same columns as open_meteo().
    """
    lat, lon, label = geocode(location)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_FIELDS),
        "timezone": "auto",
        "wind_speed_unit": "kmh",   # matches the training data units
    }
    key = ("archive", location, start_date, end_date)
    if key in _CACHE:
        return _CACHE[key].copy()

    resp = _get(ARCHIVE_URL, params, 30)
    hourly = resp.json()["hourly"]

    hf = pd.DataFrame({col: hourly[field] for field, col in HOURLY_FIELDS.items()})
    hf["date"] = pd.to_datetime(hourly["time"]).normalize()

    # Aggregate hours to days: mean for levels, sum for precipitation
    daily = hf.groupby("date").agg(
        temp=("temp", "mean"),
        humidity=("humidity", "mean"),
        precip=("precip", "sum"),
        windspeed=("windspeed", "mean"),
        cloudcover=("cloudcover", "mean"),
        sealevelpressure=("sealevelpressure", "mean"),
        shortwave=("shortwave", "mean"),
    ).reset_index()
    daily["solarradiation"] = calibrate_solar(daily.pop("shortwave"))

    add_calendar(daily)
    daily.attrs["location"] = label
    _CACHE[key] = daily
    return daily.copy()


if __name__ == "__main__":
    # Quick manual checks (need internet)
    print("Forecast (next 5 days):")
    print(open_meteo("London", 5).to_string(index=False))
    print("\nHistory (first week of January 2026):")
    print(open_meteo_history("London", "2026-01-01", "2026-01-07").to_string(index=False))

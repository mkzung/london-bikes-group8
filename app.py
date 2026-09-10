"""London bikes dashboard for TfL planners.

Two views. Explore lets a planner choose a weather variable and see how daily hires
respond, coloured by weekend or by season, alongside the weekly pattern and the whole
history month by month. Predict applies the group's regression coefficients to real
Open-Meteo weather: the first week of January 2026 from the historical archive, and
the next five days from the live forecast.

The model is not refitted here. It arrives as model_coefficients.csv, written by
bikes_assignment_group8.ipynb.
"""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate

from open_meteo import open_meteo, open_meteo_history, SEASON_BY_MONTH
# ---------------------------------------------------------------------- theme
# Both palettes carry the same hues at different lightness, so the accent is one
# colour lifted rather than two colours sharing a name. Red is reserved for
# negative and out of range; weekends are amber, which reads as a different
# rhythm rather than an error and survives red-green colour blindness.

LIGHT = {
    "name": "light",
    "page": "#F0F3F6",
    "panel": "#FFFFFF",
    "panel_alt": "#F7F9FB",
    "border": "#D5DDE4",
    "grid": "#E4EAEF",
    "field": "#FFFFFF",
    "hover": "#EDF4F7",
    "rail": "#D5DDE4",
    "ink": "#0F1A22",
    "muted": "#4A5864",
    "faint": "#5F6D78",
    "accent": "#00637A",
    "accent_ink": "#FFFFFF",
    "accent_soft": "#DCEDF3",
    "weekend": "#A94C08",
    "weekend_soft": "#FBEFE3",
    "red": "#A82A1F",
    "red_soft": "#FAE3E0",
    "header_bg": "#0F2933",
    "header_ink": "#EDF4F6",
    "header_muted": "#93AEB6",
    "header_rule": "#24444F",
    "season_band": "#E9EEF3",
    "halo": "rgba(0,99,122,.26)",
    "scrim": "rgba(15,26,34,.06)",
    "shadow": "0 1px 2px rgba(15,26,34,.05), 0 8px 24px -14px rgba(15,26,34,.20)",
    "shadow_lift": "0 2px 4px rgba(15,26,34,.06), 0 20px 44px -20px rgba(15,26,34,.30)",
    "drop": "0 14px 34px -10px rgba(15,26,34,.28)",
    "series": ["#00637A", "#A94C08", "#5B4B9E", "#1F6B45", "#96437A"],
    "season": {"Winter": "#356E92", "Spring": "#3C7A4C",
               "Summer": "#9A6A0F", "Autumn": "#8E4A2A"},
}

DARK = {
    "name": "dark",
    "page": "#0D141A",
    "panel": "#141D25",
    "panel_alt": "#1A242D",
    "border": "#28343E",
    "grid": "#222E38",
    "field": "#18222B",
    "hover": "#1E2B35",
    "rail": "#2A3742",
    "ink": "#E7EDF2",
    "muted": "#9DAAB6",
    "faint": "#7D8B98",
    "accent": "#46AECA",
    "accent_ink": "#06202A",
    "accent_soft": "#123642",
    "weekend": "#E09248",
    "weekend_soft": "#3A2413",
    "red": "#EF8A80",
    "red_soft": "#3A1A16",
    "header_bg": "#101A21",
    "header_ink": "#E9F1F4",
    "header_muted": "#8FA6AE",
    "header_rule": "#223038",
    "season_band": "#18222B",
    "halo": "rgba(70,174,202,.30)",
    "scrim": "rgba(0,0,0,.28)",
    "shadow": "0 1px 2px rgba(0,0,0,.42), 0 8px 24px -14px rgba(0,0,0,.68)",
    "shadow_lift": "0 2px 4px rgba(0,0,0,.5), 0 20px 44px -20px rgba(0,0,0,.75)",
    "drop": "0 14px 34px -10px rgba(0,0,0,.72)",
    "series": ["#57B7D1", "#E09248", "#A296E8", "#5FBE8B", "#DB8CBE"],
    "season": {"Winter": "#6FA8CB", "Spring": "#6BBE86",
               "Summer": "#D6A64A", "Autumn": "#D08A66"},
}

PALETTES = {"light": LIGHT, "dark": DARK}


def palette(name):
    return PALETTES.get(name, LIGHT)


# Anything a Dash component paints inside itself can only be reached through a
# custom property, so every token the stylesheet names has to appear on this list.
CSS_VARS = ["page", "panel", "panel_alt", "border", "grid", "field", "hover", "rail",
            "ink", "muted", "faint", "accent", "accent_ink", "accent_soft",
            "weekend", "weekend_soft", "red", "red_soft", "header_bg", "header_ink",
            "header_muted", "header_rule", "season_band", "halo", "scrim",
            "shadow", "shadow_lift", "drop"]


def css_vars(p):
    return {f"--{k.replace('_', '-')}": p[k] for k in CSS_VARS}


# Durations and curves are named once here and mirrored in the stylesheet, so a
# figure and the panel around it move on the same clock.
MOTION = {
    "--dur-instant": "90ms", "--dur-quick": "140ms", "--dur-theme": "160ms",
    "--dur-hover": "200ms", "--dur-menu": "180ms", "--dur-enter": "460ms",
    "--dur-wipe": "620ms", "--stagger": "60ms",
    "--ease-out": "cubic-bezier(.16,.84,.44,1)",
    "--ease-standard": "cubic-bezier(.2,0,0,1)",
    "--ease-exit": "cubic-bezier(.4,0,1,1)",
}

FONT = "IBM Plex Sans, system-ui, Helvetica, Arial, sans-serif"


# ------------------------------------------------------------------ plotly
def base_layout(p, height=None, margin=None, title=None):
    """What every figure shares. Plotly's own transition stays at zero: the server
    redraws these on each filter change and animating that is the trap.

    Height, margin and title are parameters rather than something a caller adds
    afterwards, because update_layout refuses a keyword this dict already holds.
    """
    layout = dict(
        font=dict(family=FONT, size=13, color=p["muted"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=64, r=16, t=8, b=48),
        colorway=p["series"],
        transition=dict(duration=0),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=p["faint"],
                     activecolor=p["accent"], orientation="h"),
        hoverlabel=dict(bgcolor=p["panel"], bordercolor=p["border"], align="left",
                        font=dict(family=FONT, size=13, color=p["ink"])),
        # Left, because the modebar owns the top right corner and the two collide
        # there.
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(family=FONT, size=12, color=p["muted"])),
        xaxis=dict(showgrid=False, zeroline=False, ticks="",
                   linecolor=p["border"], linewidth=1,
                   tickfont=dict(family=FONT, size=12, color=p["faint"]),
                   title_font=dict(family=FONT, size=12.5, color=p["muted"])),
        yaxis=dict(showgrid=True, gridcolor=p["grid"], gridwidth=1,
                   zeroline=False, showline=False, ticks="", tickformat=",",
                   tickfont=dict(family=FONT, size=12, color=p["faint"]),
                   title_font=dict(family=FONT, size=12.5, color=p["muted"])),
    )
    if height:
        layout["height"] = height
    if margin:
        layout["margin"] = margin
    if title:
        layout["title"] = dict(text=title, font=dict(size=15, color=p["ink"]),
                               x=0, xanchor="left", pad=dict(b=12))
    return layout


def spikes(p):
    """A crosshair back to the axis, for the two charts where a point sits alone in
    space. Bars carry their value above them and need none."""
    return dict(showspikes=True, spikemode="across", spikesnap="cursor",
                spikecolor=p["accent"], spikethickness=1, spikedash="dot")


def winter_bands(p, first=2014, last=2025):
    """December to February behind the monthly line, so the trough reads as winter
    rather than as a dip somebody has to date by eye."""
    return [dict(type="rect", xref="x", yref="paper", layer="below",
                 x0=f"{y - 1}-12-01", x1=f"{y}-03-01", y0=0, y1=1,
                 fillcolor=p["season_band"], line=dict(width=0))
            for y in range(first, last + 1)]


def rangeslider(p):
    return dict(visible=True, thickness=0.13, bgcolor=p["panel_alt"],
                bordercolor=p["border"], borderwidth=1)


def rangeselector(p):
    return dict(bgcolor=p["panel_alt"], activecolor=p["accent_soft"],
                bordercolor=p["border"], borderwidth=1,
                font=dict(family=FONT, size=12, color=p["muted"]),
                x=0, y=1.06, xanchor="left", yanchor="bottom",
                buttons=[dict(count=1, label="1y", step="year", stepmode="backward"),
                         dict(count=3, label="3y", step="year", stepmode="backward"),
                         dict(count=5, label="5y", step="year", stepmode="backward"),
                         dict(step="all", label="All")])


# The five-day panel is refetched every morning. A free y-axis would rescale
# overnight and make Tuesday and Wednesday incomparable at a glance, so it is pinned
# above the highest this panel's own arithmetic reaches on the training weather,
# 49,300, plus the 9,377 the prediction band adds on top of it.
FORECAST_CEILING = 60000

GRAPH_CONFIG = {
    "displayModeBar": True,          # not "hover": a toolbar nobody finds is none
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,             # the page scrolls past these; wheel would trap it
    "doubleClick": "reset",
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d",
                               "toggleSpikelines", "hoverCompareCartesian",
                               "hoverClosestCartesian"],
    "toImageButtonOptions": {"format": "png", "filename": "santander-cycles",
                             "scale": 2, "width": 1440, "height": 560},
}

# Roping a cluster and reading its count is worth a button on the one chart with a
# cloud of points in it.
SCATTER_CONFIG = {**GRAPH_CONFIG,
                  "modeBarButtonsToRemove":
                      [b for b in GRAPH_CONFIG["modeBarButtonsToRemove"]
                       if b != "select2d"]}

# Seven bars have nothing to zoom into, and a stray drag that zooms them is a bug
# report rather than a feature.
STATIC_CONFIG = {**GRAPH_CONFIG,
                 "modeBarButtonsToRemove": GRAPH_CONFIG["modeBarButtonsToRemove"]
                 + ["zoom2d", "pan2d", "zoomIn2d", "zoomOut2d", "resetScale2d"]}


# ------------------------------------------------------------------ constants

DATA_URL = ("https://raw.githubusercontent.com/kostis-christodoulou/"
            "am01-code-sep2026/main/data/london_bikes.csv")
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_FULL = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday",
            "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday"}
# How each weather column is labelled and rounded wherever it is shown. The model
# decides which of them appear; this only decides what they look like.
COLUMNS = {
    "temp": ("Temp °C", 1),
    "humidity": ("Humidity %", 0),
    "precip": ("Precip mm", 1),
    "windspeed": ("Wind km/h", 1),
    "cloudcover": ("Cloud %", 0),
    "sealevelpressure": ("Pressure hPa", 0),
    "visibility": ("Visibility km", 1),
    "solarradiation": ("Solar W/m²", 0),
}

# The eight open_meteo.py returns, which is wider than the model uses: humidity and
# cloud cover are worth looking at even though the fit dropped them. The training
# file also carries feels-like, which tracks temp at r = 0.99 and is the collinearity
# Part 3 is about, but the forecast cannot supply it.
WEATHER = {
    "temp": ("Temperature", "°C"),
    "solarradiation": ("Solar radiation", "W/m²"),
    "visibility": ("Visibility", "km"),
    "humidity": ("Humidity", "%"),
    "precip": ("Precipitation", "mm"),
    "windspeed": ("Wind speed", ""),
    "cloudcover": ("Cloud cover", "%"),
    "sealevelpressure": ("Sea level pressure", "hPa"),
}
COLOUR_BY = {
    "weekend": "Weekday / weekend",
    "season_name": "Season",
    "none": "Nothing",
}

# The app never refits, so the figures in the masthead are quoted rather than
# computed. They come from the M4 summary in bikes_assignment_group8.ipynb and have
# to be re-read from it whenever the model changes.
FIT = {"adj_r2": "0.728", "resid_se": "4,784", "days": "4,382"}

# The prediction band needs the residual standard error, and the notebook's export
# writes one row per model term and nothing else. Read from the file where a row
# for it exists, and otherwise from here, where it is quoted from the same summary
# the two figures above come from.
RESIDUAL_SE = 4784.003

# Open-Meteo documents its wind in km/h; the training file's wind runs seven units
# above it, so the Explore axis stays unitless and only the forecast tables, which
# are Open-Meteo throughout, name the unit.


# Two copies of an Open-Meteo answer, kept beside the app for when the API will
# not give another. January is a week that has already happened and never changes;
# the forecast seed is only a starting point, replaced by the newest good answer.
JANUARY_FALLBACK = "january_2026.csv"
FORECAST_CACHE = "forecast_seed.csv"


def load_bikes():
    """The dataset, cleaned the way the model was fitted."""
    df = pd.read_csv(DATA_URL)
    df["date"] = pd.to_datetime(df["date"])
    df["weekend"] = df["wday"].isin(["Sat", "Sun"])
    df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=DAY_ORDER,
                                       ordered=True)
    df["day_label"] = df["weekend"].map({True: "Weekend", False: "Weekday"})
    return df[df["date"] >= pd.to_datetime("2014-01-01", utc=True)].copy()


def load_coefficients(path="model_coefficients.csv"):
    """term -> coefficient, as written by the notebook."""
    return pd.read_csv(path).set_index("term")["coefficient"].to_dict()


# Rows in the coefficients file that are not terms in the model.
METADATA = {"residual_se"}

# Yesterday's hires. It is a term like any other in the file, but the only one no
# weather service can hand over for a day that has not happened, so it is filled
# in a step at a time rather than read off a column.
LAG_TERM = "bikes_hired_lag1"


def drivers(coef):
    """The weather terms in a coefficients file, in the order it lists them."""
    return [t for t in coef if t != "Intercept" and t not in METADATA
            and not t.startswith(("day_", "season_"))]


def seed_lag(first_day):
    """Yesterday's hires for the day a forecast starts, and where they came from.

    If the dataset happens to reach the day before, that is the real number and the
    forecast starts from solid ground. Otherwise the average for that weekday in
    that season stands in, which is only defensible because the lag coefficient is
    well under one: by the fifth day the seed is worth about 2% of whatever error
    it carried.
    """
    day_before = pd.Timestamp(first_day).normalize() - pd.Timedelta(days=1)
    observed = bikes[bikes["date"].dt.tz_localize(None).dt.normalize() == day_before]
    if len(observed):
        value = float(observed["bikes_hired"].iloc[0])
        return value, f"the {value:,.0f} hires actually recorded on {day_before:%-d %B %Y}"
    weekday = day_before.strftime("%a")
    season = SEASON_BY_MONTH[day_before.month]
    like = bikes[(bikes["day_of_week"].astype(str) == weekday)
                 & (bikes["season_name"] == season)]
    value = float(like["bikes_hired"].mean())
    return value, (f"the average {weekday} in {season.lower()}, {value:,.0f} hires, "
                   f"because the data stops on {bikes['date'].max():%-d %B %Y}")


def predict(weather, coef, seed=None):
    """Apply whatever terms the coefficients file holds to a frame from open_meteo.

    The file decides the model rather than this function: a weather term multiplies
    the column of the same name, and a day_ or season_ term is chosen by the row's
    own calendar. Saying which term has no column behind it beats a KeyError raised
    three frames down with the name of a column nobody asked for.

    Yesterday's hires are the exception. A week being forecast has no yesterday, so
    the days are walked in order and each prediction becomes the next day's lag,
    which is how a model with a lagged dependent variable forecasts more than one
    step ahead.

    A linear model has nothing stopping it going below zero, and on the wettest days
    in the training file it does. Hires cannot be negative, so the floor is 0.
    """
    out = weather.copy()
    weather_terms = [t for t in drivers(coef) if t != LAG_TERM]
    missing = [t for t in weather_terms if t not in out.columns]
    if missing:
        raise KeyError(
            f"model_coefficients.csv is fitted on {', '.join(missing)}, which "
            f"open_meteo.py does not return. It returns "
            f"{', '.join(c for c in out.columns if c != 'date')}.")
    total = coef["Intercept"]
    for v in weather_terms:
        total = total + coef[v] * out[v]
    # astype(str) first: open_meteo hands back plain strings, the bikes frame holds a
    # categorical, and mapping a categorical returns one that will not add to a float
    total = total + out["day_of_week"].astype(str).map(
        lambda d: coef.get(f"day_{d}", 0.0))
    if "season_name" in out.columns:
        total = total + out["season_name"].astype(str).map(
            lambda s: coef.get(f"season_{s}", 0.0))

    if LAG_TERM in coef:
        phi = coef[LAG_TERM]
        previous = seed if seed is not None else seed_lag(out["date"].min())[0]
        running = []
        for base in total:
            previous = max(0.0, base + phi * previous)
            running.append(previous)
        out["predicted_hires"] = pd.Series(running, index=out.index).round(0)
    else:
        out["predicted_hires"] = total.clip(lower=0).round(0)
    return out


bikes = load_bikes()
coefficients = load_coefficients()
YEARS = (int(bikes["date"].dt.year.min()), int(bikes["date"].dt.year.max()))

app = Dash(__name__, title="London bikes — TfL planning",
           update_title=None, suppress_callback_exceptions=True)
server = app.server                                  # Render serves this object

app.index_string = """<!DOCTYPE html><html><head>{%metas%}
<meta name="color-scheme" content="light dark">
<title>{%title%}</title>
{%favicon%}{%css%}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap"
      rel="stylesheet">
<style>
  *{box-sizing:border-box}
  html,body{margin:0;padding:0}
  body{font-family:"IBM Plex Sans",system-ui,Helvetica,Arial,sans-serif;
    font-size:15px;line-height:1.5;font-variant-numeric:tabular-nums;
    -webkit-font-smoothing:antialiased}
  ::-webkit-scrollbar{width:10px;height:10px}
  ::-webkit-scrollbar-thumb{background:var(--faint);opacity:.4;border-radius:6px}

  /* geometry, which no callback needs to vary */
  :root{--r-sm:3px;--r-md:5px;--r-lg:8px;--hair:1px;--gut:20px}

  /* ── keyframes: transform, opacity and clip-path only ───────────────── */
  @keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
  @keyframes wipe-x{from{transform:scaleX(0)}to{transform:scaleX(1)}}
  @keyframes reveal-num{from{clip-path:inset(0 100% 0 0);transform:translateY(4px)}
    to{clip-path:inset(0 0 0 0);transform:none}}

  /* Every surface crossfades on one clock, so a theme flip reads as one move. */
  .masthead,.panel,.stats,.data th,.data td,.dash-dropdown-trigger,
  .dash-dropdown-content,.dash-range-slider-input,.dash-slider-track,
  .dash-slider-range,.dash-slider-thumb{
    transition:background-color var(--dur-theme) var(--ease-standard),
      border-color var(--dur-theme) var(--ease-standard),
      color var(--dur-theme) var(--ease-standard),
      box-shadow var(--dur-theme) var(--ease-standard)}

  /* One focus treatment everywhere, never the browser default. */
  :focus-visible{outline:none}
  .dash-dropdown-trigger:focus-visible,.dash-range-slider-input:focus-visible,
  .dash-slider-thumb:focus-visible,.theme-toggle:focus-visible,
  .dash-tab:focus-visible{
    outline:2px solid var(--accent)!important;outline-offset:2px;
    box-shadow:0 0 0 5px var(--halo)!important}

  /* ── masthead ───────────────────────────────────────────────────────── */
  .masthead{background:var(--header-bg);color:var(--header-ink);
    border-bottom:var(--hair) solid var(--header-rule);
    position:relative;overflow:hidden}
  /* One hairline of signal, encoding the weekday and weekend share of the week. */
  .masthead::after{content:"";position:absolute;left:0;right:0;bottom:0;height:2px;
    background:linear-gradient(90deg,var(--accent) 0%,var(--accent) 34%,
      var(--weekend) 34%,var(--weekend) 46%,
      var(--header-rule) 46%,var(--header-rule) 100%);
    transform:scaleX(0);transform-origin:0 50%;
    animation:wipe-x var(--dur-wipe) var(--ease-out) 120ms both}
  .mast-inner{max-width:1440px;margin:0 auto;padding:22px var(--gut) 20px;
    display:grid;grid-template-columns:minmax(0,1fr) auto;
    align-items:start;gap:24px}
  .brandline{margin:0 0 6px;font-size:11px;font-weight:600;letter-spacing:.16em;
    text-transform:uppercase;color:var(--header-muted);
    animation:rise var(--dur-enter) var(--ease-out) both}
  .title{margin:0;font-size:clamp(21px,2.1vw,27px);font-weight:600;
    letter-spacing:-.012em;line-height:1.15;text-wrap:balance;
    animation:rise var(--dur-enter) var(--ease-out) 60ms both}
  .provenance{margin:8px 0 0;font-size:13px;color:var(--header-muted);
    text-wrap:pretty;animation:rise var(--dur-enter) var(--ease-out) 120ms both}
  .provenance b{color:var(--header-ink);font-weight:500}
  .toggle-wrap{display:flex;align-items:center;gap:10px;
    animation:rise var(--dur-enter) var(--ease-out) 180ms both}
  .toggle-legend{font-size:11px;font-weight:500;letter-spacing:.1em;
    text-transform:uppercase;color:var(--header-muted)}
  .theme-toggle{display:block;position:relative;width:52px;height:28px;flex:none;
    padding:0;border-radius:999px;background:var(--header-rule);
    border:var(--hair) solid var(--header-rule);cursor:pointer;
    transition:background-color var(--dur-theme) var(--ease-standard)}
  /* The track is 52 by 28 with a 1px border, so the box inside it is 50 by 26 and
     a 20px knob needs 3px on every side to sit centred and to travel 24px into a
     mirror image of where it started. */
  .theme-toggle::before{content:"";position:absolute;top:3px;left:3px;
    width:20px;height:20px;border-radius:50%;background:var(--header-ink);
    transition:transform var(--dur-hover) var(--ease-out)}
  /* A second disc in the track colour bites the knob into a crescent when dark. */
  .theme-toggle::after{content:"";position:absolute;top:3px;left:3px;
    width:20px;height:20px;border-radius:50%;background:var(--header-rule);
    transform:translate(9px,-9px) scale(.1);
    transition:transform var(--dur-hover) var(--ease-out),
      background-color var(--dur-theme) var(--ease-standard)}
  .theme-toggle:hover,.theme-toggle:hover::after{background:var(--accent)}
  .theme-dark .theme-toggle::before{transform:translateX(24px)}
  .theme-dark .theme-toggle::after{transform:translate(29px,-5px) scale(.86)}

  /* ── shell and tabs ─────────────────────────────────────────────────── */
  .shell{max-width:1440px;margin:0 auto;padding:0 var(--gut) 56px}
  .dash-tabs{animation:rise var(--dur-enter) var(--ease-out) 200ms both}
  /* Dash gives each tab flex:1, which spreads two of them over the whole page and
     turns the hover scrim into a half-page slab. They size to their labels. */
  .dash-tab{flex:0 0 auto!important;position:relative;
    border-radius:var(--r-sm) var(--r-sm) 0 0;
    transition:color var(--dur-hover) var(--ease-standard),
      background-color var(--dur-hover) var(--ease-standard)}
  .dash-tab::after{content:"";position:absolute;left:10px;right:10px;bottom:-1px;
    height:2px;background:var(--accent);transform:scaleX(0);
    transform-origin:50% 50%;transition:transform var(--dur-hover) var(--ease-out)}
  .dash-tab:hover{color:var(--ink)!important;background:var(--scrim)!important}
  .dash-tab:hover::after{transform:scaleX(.4)}
  .dash-tab--selected::after{transform:scaleX(1)}

  /* ── control row ────────────────────────────────────────────────────── */
  .controls{display:grid;
    grid-template-columns:minmax(0,1fr) minmax(0,1fr) minmax(0,1.5fr);
    gap:18px;align-items:end;
    animation:rise var(--dur-enter) var(--ease-out) 240ms both}
  .field-group{min-width:0}
  .field-label{display:block;margin-bottom:7px;font-size:11px;font-weight:600;
    letter-spacing:.1em;text-transform:uppercase;color:var(--faint)}

  /* The field is the span inside the dropdown's button, so the button itself
     keeps nothing: its own border and outline would draw a second ring. */
  button.dash-dropdown{background:transparent!important;border:none!important;
    outline:none!important;padding:0!important;box-shadow:none!important;
    width:100%;position:relative}
  .dash-dropdown-trigger{width:100%;min-height:42px;padding:10px 12px!important;
    font:inherit;font-size:14px!important;color:var(--ink)!important;
    text-align:left;background:var(--field)!important;
    border:var(--hair) solid var(--border)!important;
    border-radius:var(--r-md)!important;cursor:pointer;
    transition:border-color var(--dur-hover) var(--ease-standard),
      box-shadow var(--dur-hover) var(--ease-standard),
      transform var(--dur-hover) var(--ease-out)}
  .dash-dropdown-trigger:hover{border-color:var(--accent)!important;
    box-shadow:inset 0 0 0 1px var(--accent)!important;transform:translateY(-1px)}
  .dash-dropdown-trigger[aria-expanded="true"]{border-color:var(--accent)!important;
    box-shadow:0 0 0 3px var(--halo)!important}
  .dash-dropdown-value,.dash-dropdown-value-item{color:var(--ink)!important;
    font-size:14px!important}
  .dash-dropdown-trigger-icon{color:var(--faint)!important;
    transition:transform var(--dur-instant) var(--ease-standard),
      color var(--dur-quick) var(--ease-standard)}
  .dash-dropdown-trigger:hover .dash-dropdown-trigger-icon{color:var(--accent)!important}
  .dash-dropdown-content{background:var(--panel)!important;
    border:var(--hair) solid var(--border)!important;
    border-radius:var(--r-md)!important;box-shadow:var(--drop)!important;
    padding:5px!important;overflow:hidden;max-height:340px!important}
  /* An open menu lives inside the control row, and the entrance animation leaves
     that row holding an identity transform, which makes it a stacking context the
     menu cannot escape. Raising the row itself lifts the menu with it, above the
     stat row and the panels that follow it in the document. */
  .controls{position:relative;z-index:50}
  .dash-dropdown-search-container{background:transparent!important;
    border-bottom:var(--hair) solid var(--border)!important}
  .dash-dropdown-search{background:var(--field)!important;color:var(--ink)!important;
    border:var(--hair) solid var(--border)!important;border-radius:var(--r-sm)!important}
  .dash-dropdown-search-icon{color:var(--faint)!important}
  .dash-dropdown-options{background:transparent!important}
  .dash-dropdown-option{color:var(--muted)!important;font-size:14px!important;
    padding:8px 10px!important;border-radius:var(--r-sm)!important;
    transition:background-color var(--dur-quick) var(--ease-standard),
      color var(--dur-quick) var(--ease-standard)}
  .dash-dropdown-option .dash-options-list-option-text{color:inherit!important}
  .dash-dropdown-option:hover{background:var(--hover)!important;color:var(--ink)!important}
  .dash-dropdown-option.selected{background:var(--accent-soft)!important;
    color:var(--ink)!important;font-weight:500}
  .dash-dropdown-option.selected .dash-options-list-option-text{color:var(--ink)!important}
  .dash-options-list-option-checkbox{display:none!important}

  /* ── slider ─────────────────────────────────────────────────────────── */
  .dash-slider-track{background:var(--rail)!important;height:4px!important}
  .dash-slider-range{background:var(--accent)!important;height:4px!important;
    transform-origin:0 50%;
    animation:wipe-x var(--dur-wipe) var(--ease-out) 320ms both}
  .dash-slider-thumb{width:18px!important;height:18px!important;
    background:var(--panel)!important;border:2px solid var(--accent)!important;
    box-shadow:none!important;cursor:grab;
    transition:transform var(--dur-hover) var(--ease-out),
      box-shadow var(--dur-hover) var(--ease-standard)}
  .dash-slider-thumb:hover{transform:scale(1.18);
    box-shadow:0 0 0 6px var(--halo)!important}
  .dash-slider-thumb:active{cursor:grabbing;transform:scale(1.06)}
  .dash-slider-mark{color:var(--faint)!important;font-size:10.5px!important}
  .dash-slider-tooltip{background:var(--panel)!important;color:var(--ink)!important;
    border:var(--hair) solid var(--border)!important}
  .dash-range-slider-input{width:68px!important;padding:9px 8px!important;
    text-align:center;font-size:14px!important;
    background:var(--field)!important;color:var(--ink)!important;
    border:var(--hair) solid var(--border)!important;
    border-radius:var(--r-md)!important}
  .dash-range-slider-input:hover{border-color:var(--accent)!important}
  .dash-range-slider-input:focus{border-color:var(--accent)!important;
    outline:none!important;box-shadow:0 0 0 3px var(--halo)!important}

  /* ── stat row ───────────────────────────────────────────────────────── */
  .stats{container-type:inline-size;display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));margin:26px 0 24px;
    border-top:var(--hair) solid var(--border);
    animation:rise var(--dur-enter) var(--ease-out) 340ms both}
  .stat{position:relative;padding:18px 22px 16px 0;min-width:0}
  .stat+.stat{padding-left:22px;border-left:var(--hair) solid var(--border)}
  .stat::before{content:"";position:absolute;top:-1px;left:0;width:100%;height:2px;
    background:var(--faint);transform:scaleX(0);transform-origin:0 50%;
    animation:wipe-x var(--dur-wipe) var(--ease-out) both}
  .stat:nth-child(1)::before{animation-delay:380ms}
  .stat:nth-child(2)::before{animation-delay:calc(380ms + var(--stagger))}
  .stat:nth-child(3)::before{animation-delay:calc(380ms + var(--stagger)*2);
    background:var(--accent)}
  .stat:nth-child(4)::before{animation-delay:calc(380ms + var(--stagger)*3);
    background:var(--weekend)}
  /* A wipe, not a count-up: a count-up needs JavaScript. */
  .stat-num{display:block;font-size:clamp(26px,2.6vw,34px);font-weight:600;
    letter-spacing:-.02em;line-height:1.05;color:var(--ink);
    clip-path:inset(0 100% 0 0);
    animation:reveal-num var(--dur-wipe) var(--ease-out) both}
  .stat:nth-child(1) .stat-num{animation-delay:400ms}
  .stat:nth-child(2) .stat-num{animation-delay:calc(400ms + var(--stagger))}
  .stat:nth-child(3) .stat-num{animation-delay:calc(400ms + var(--stagger)*2);
    color:var(--accent)}
  .stat:nth-child(4) .stat-num{animation-delay:calc(400ms + var(--stagger)*3);
    color:var(--weekend)}
  .stat-label{display:block;margin-top:6px;font-size:12.5px;color:var(--muted);
    text-wrap:pretty}
  @container (max-width:700px){
    .stats{grid-template-columns:repeat(2,minmax(0,1fr))}
    .stat:nth-child(3){border-left:0;padding-left:0}
    .stat:nth-child(3),.stat:nth-child(4){border-top:var(--hair) solid var(--border)}}

  /* ── panels ─────────────────────────────────────────────────────────── */
  .panel{position:relative;background:var(--panel);
    border:var(--hair) solid var(--border);border-radius:var(--r-lg);
    box-shadow:var(--shadow);padding:20px 22px 22px;min-width:0;
    animation:rise var(--dur-enter) var(--ease-out) both;
    transition:transform var(--dur-hover) var(--ease-out),
      border-color var(--dur-hover) var(--ease-standard),
      box-shadow var(--dur-hover) var(--ease-standard),
      background-color var(--dur-theme) var(--ease-standard)}
  .panel::before{content:"";position:absolute;top:-1px;left:12px;right:12px;
    height:2px;background:var(--accent);border-radius:2px;transform:scaleX(0);
    transition:transform var(--dur-hover) var(--ease-out)}
  .panel:hover{border-color:var(--accent);box-shadow:var(--shadow-lift);
    transform:translateY(-2px)}
  .panel:hover::before{transform:scaleX(1)}
  .stack{display:grid;gap:20px}
  .grid-2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}
  .stack>.panel:nth-child(1),.grid-2>.panel:nth-child(1){animation-delay:380ms}
  .stack>.panel:nth-child(2),.grid-2>.panel:nth-child(2){
    animation-delay:calc(380ms + var(--stagger))}
  .stack>.panel:nth-child(3){animation-delay:calc(380ms + var(--stagger)*2)}
  .panel-title{margin:0;font-size:15px;font-weight:600;letter-spacing:-.005em;
    color:var(--ink)}
  .panel-note{margin:4px 0 14px;font-size:12.5px;color:var(--faint);text-wrap:pretty}

  /* ── tables ─────────────────────────────────────────────────────────── */
  .table-scroll{margin-top:18px;overflow-x:auto;overscroll-behavior-x:contain;
    border:var(--hair) solid var(--border);border-radius:var(--r-md)}
  .data{width:100%;border-collapse:collapse;font-size:13.5px;min-width:640px}
  .data.narrow{min-width:0}
  .data th{padding:10px 14px;font-size:11px;font-weight:600;letter-spacing:.08em;
    text-transform:uppercase;color:var(--faint);background:var(--panel-alt);
    text-align:left;white-space:nowrap;
    border-bottom:var(--hair) solid var(--border)}
  .data td{padding:9px 14px;color:var(--muted);
    border-bottom:var(--hair) solid var(--grid);white-space:nowrap}
  .data tbody tr:last-child td{border-bottom:0}
  .data .num{text-align:right}
  .data .out{color:var(--ink);font-weight:600;position:relative}
  .data .neg{color:var(--red)}
  .data tbody tr{transition:background-color var(--dur-hover) var(--ease-standard)}
  .data tbody tr:hover{background:var(--hover)}
  .data tbody tr:hover .out{color:var(--accent)}
  .data tbody tr td:first-child{box-shadow:inset 0 0 0 0 var(--accent);
    transition:box-shadow var(--dur-hover) var(--ease-out),
      color var(--dur-hover) var(--ease-standard)}
  .data tbody tr:hover td:first-child{box-shadow:inset 3px 0 0 0 var(--accent);
    color:var(--ink)}
  .data tbody tr.we{background:var(--weekend-soft)}
  .data tbody tr.we .out,.data tbody tr.we:hover .out{color:var(--weekend)}
  .data tbody tr.we td:first-child{color:var(--ink)}
  .data tbody tr.we:hover td:first-child{box-shadow:inset 3px 0 0 0 var(--weekend)}
  .coef{max-width:560px}

  /* ── responsive: 1440 down to 768, one column below ─────────────────── */
  @media (max-width:1080px){
    .controls{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}
    .controls .slider-group{grid-column:1/-1}}
  @media (max-width:860px){
    :root{--gut:16px}
    .grid-2,.controls,.mast-inner{grid-template-columns:minmax(0,1fr)}
    .toggle-wrap{justify-content:flex-start}
    .stats{grid-template-columns:repeat(2,minmax(0,1fr))}}
  @media (max-width:560px){
    .stats{grid-template-columns:minmax(0,1fr)}
    .stat+.stat{padding-left:0;border-left:0;
      border-top:var(--hair) solid var(--border)}
    .panel{padding:16px 14px 18px}}

  @media (prefers-reduced-motion:reduce){
    *,*::before,*::after{animation-duration:1ms!important;animation-delay:0ms!important;
      transition-duration:1ms!important;scroll-behavior:auto!important}
    .panel:hover,.dash-dropdown-trigger:hover,.dash-slider-thumb:hover{
      transform:none}}
</style></head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""


# ----------------------------------------------------------------- small pieces
# A responsive figure is sized by the element around it, and dcc.Graph writes
# height:100% inline, which a stylesheet cannot outrank. The height goes here.
CHART = {"height": "430px"}
CHART_HALF = {"height": "380px"}
CHART_BARS = {"height": "330px"}


def panel(children, note=None, title=None):
    # A Dash component counts its children for len(), so an empty one is falsy and
    # a plain truth test would drop the very placeholder a callback fills in.
    head = []
    if title is not None:
        head.append(html.H3(title, className="panel-title"))
    if note is not None:
        head.append(html.P(note, className="panel-note"))
    return html.Div(head + list(children), className="panel")


def stat(value, label):
    return html.Div([html.Span(value, className="stat-num"),
                     html.Span(label, className="stat-label")], className="stat")


def field(label_text, control):
    return html.Div([html.Label(label_text, className="field-label"), control],
                    className="field-group")


def data_table(header, rows, narrow=False, width=""):
    """A hand-built table. The row treatments the design asks for, a weekend tint,
    an inset rule on hover and a magnitude bar behind a figure, are all per-cell."""
    head = html.Thead(html.Tr([
        html.Th(name, className="num" if right else "") for name, right in header]))
    body = html.Tbody([html.Tr(cells, className="we" if weekend else "")
                       for cells, weekend in rows])
    return html.Div(html.Table([head, body],
                               className="data narrow" if narrow else "data"),
                    className=f"table-scroll {width}".strip())


def reason(exc):
    """The HTTP status if there is one, because the exception class alone tells a
    reader nothing about why the weather did not arrive."""
    resp = getattr(exc, "response", None)
    return f"HTTP {resp.status_code}" if resp is not None else type(exc).__name__


def signed(text):
    """A true minus sign, U+2212. A hyphen is narrower than a digit and breaks the
    column's alignment, which is the whole reason the tables are tabular."""
    return text.replace("-", "−")


def magnitude(pct, tone):
    """A tinted bar behind a figure, so relative size reads without a chart."""
    pct = max(0.0, min(100.0, pct))
    return {"backgroundImage": f"linear-gradient(to left, {tone} 0 {pct:.1f}%,"
                               f" transparent {pct:.1f}% 100%)",
            "backgroundRepeat": "no-repeat",
            "backgroundSize": "calc(100% - 24px) 17px",
            "backgroundPosition": "right 12px center"}


# ---------------------------------------------------------------------- layout
app.layout = html.Div(id="root", children=[
    dcc.Store(id="theme", data="light", storage_type="session"),
    html.Div(id="shell"),
])


@app.callback(Output("theme", "data"),
              Input("theme-toggle", "n_clicks"),
              State("theme", "data"), prevent_initial_call=True)
def flip_theme(n_clicks, current):
    """The button is rebuilt whenever the shell redraws, and a component that has
    just appeared fires its callback once with n_clicks at 0. Only a real click
    should flip anything, so anything falsy is left alone."""
    if not n_clicks:
        raise PreventUpdate
    return "dark" if current == "light" else "light"


@app.callback(Output("root", "style"), Output("root", "className"),
              Input("theme", "data"))
def page_style(name):
    """The tokens the stylesheet reads. A Dash component paints its own internals,
    so a colour that is not on this list cannot reach them."""
    p = palette(name)
    style = {"minHeight": "100vh", "background": p["page"], "color": p["ink"],
             "transition": "background-color var(--dur-theme) var(--ease-standard)"}
    style.update(css_vars(p))
    style.update(MOTION)
    return style, f"theme-{name}"


@app.callback(Output("shell", "children"), Input("theme", "data"))
def shell(name):
    p = palette(name)
    return [
        html.Div(className="masthead", children=html.Div(className="mast-inner", children=[
            html.Div([
                html.P("TfL Santander Cycles", className="brandline"),
                html.H1("Daily hires: explore and forecast", className="title"),
                html.P([html.B(f"{len(bikes):,} days"), " from 2014 · adjusted R² ",
                        html.B(FIT["adj_r2"]), " on the ", html.B(FIT["days"]),
                        " a lag term leaves · typical error ",
                        html.B(FIT["resid_se"]),
                        " hires a day"], className="provenance"),
            ]),
            html.Div(className="toggle-wrap", children=[
                html.Span("Theme", className="toggle-legend"),
                html.Button(id="theme-toggle", n_clicks=0, className="theme-toggle",
                            **{"aria-label": "Switch between the light and dark theme"}),
            ]),
        ])),
        html.Div(className="shell", children=[
            # A theme flip redraws everything below, and a redrawn component opens
            # on its default. Persistence hands each one its last value back.
            dcc.Tabs(id="tabs", value="explore", className="dash-tabs",
                     persistence=True, persistence_type="session",
                     children=[dcc.Tab(label="Explore the data", value="explore",
                                       className="dash-tab",
                                       selected_className="dash-tab--selected",
                                       style=tab_style(p), selected_style=tab_sel(p)),
                               dcc.Tab(label="Predict demand", value="predict",
                                       className="dash-tab",
                                       selected_className="dash-tab--selected",
                                       style=tab_style(p), selected_style=tab_sel(p))],
                     style={"height": "48px", "borderBottom": f"1px solid {p['border']}"}),
            html.Div(id="body", style={"marginTop": "22px"}),
        ]),
    ]


def tab_style(p):
    return {"background": "transparent", "color": p["muted"], "border": "none",
            "padding": "15px 18px 13px", "fontWeight": 500, "fontSize": "14px",
            "fontFamily": FONT}


def tab_sel(p):
    return {"background": "transparent", "color": p["ink"], "border": "none",
            "padding": "15px 18px 13px", "fontWeight": 600, "fontSize": "14px",
            "fontFamily": FONT}


@app.callback(Output("body", "children"),
              Input("tabs", "value"), Input("theme", "data"))
def body(tab, name):
    # key changes with the tab, which makes React build a fresh subtree instead of
    # reusing the old one. Without it the entrance animations run once on the first
    # page load and never again, because a reused node keeps its finished animation.
    # A tab is a move between views rather than a change of data, so replaying them
    # here does not break the rule against animating a filter change.
    if tab == "explore":
        return html.Div(key=tab, children=[
            html.Div(className="controls", children=[
                field("Weather variable",
                      dcc.Dropdown(id="weather-var", clearable=False, value="temp",
                                   persistence=True, persistence_type="session",
                                   options=[{"label": f"{n}{f' ({u})' if u else ''}",
                                             "value": k}
                                            for k, (n, u) in WEATHER.items()])),
                field("Colour by",
                      dcc.Dropdown(id="colour-by", clearable=False, value="weekend",
                                   persistence=True, persistence_type="session",
                                   options=[{"label": v, "value": k}
                                            for k, v in COLOUR_BY.items()])),
                html.Div(className="field-group slider-group", children=[
                    html.Label("Year range", className="field-label"),
                    dcc.RangeSlider(id="year-range", min=YEARS[0], max=YEARS[1],
                                    step=1, value=list(YEARS),
                                    persistence=True, persistence_type="session",
                                    tooltip={"placement": "bottom",
                                             "always_visible": False},
                                    marks={y: str(y) for y in
                                           range(YEARS[0], YEARS[1] + 1, 3)}),
                ]),
            ]),
            html.Div(id="summary", className="stats"),
            html.Div(className="stack", children=[
                panel([dcc.Graph(id="scatter", config=SCATTER_CONFIG,
                                 style=CHART)],
                      title=html.Span(id="scatter-title"),
                      note="One point per day, with an OLS trendline per group. "
                           "Drag to zoom, click a legend key to isolate a group."),
                html.Div(className="grid-2", children=[
                    panel([dcc.Graph(id="by-day", config=STATIC_CONFIG,
                                     style=CHART_HALF)],
                          title="Average hires by day of week",
                          note="Weekend bars carry the weekend colour throughout "
                               "the product. Hover a bar for its gap to the "
                               "weekly mean."),
                    panel([dcc.Graph(id="over-time", config=GRAPH_CONFIG,
                                     style=CHART_HALF)],
                          title="Average daily hires by month",
                          note="Drag the window under the chart, or pick a span "
                               "above it. Winter months sit on a band."),
                ]),
            ]),
        ])
    return html.Div(key=tab, className="stack", children=[
        panel([dcc.Loading(html.Div(id="jan"), color=palette(name)["accent"],
                           type="dot")],
              title="First week of January 2026",
              note="Held-out week, predicted from weather the Open-Meteo archive "
                   "has already recorded. The whisker on each bar is the 95% "
                   "range of days with that weather."),
        panel([dcc.Loading(html.Div(id="forecast"), color=palette(name)["accent"],
                           type="dot")],
              title="The next five days",
              note="Live Open-Meteo forecast for London, refreshed on every load. "
                   f"The axis is pinned to {FORECAST_CEILING:,} so this morning "
                   "and tomorrow morning stay comparable. The whisker is the 95% "
                   "range of days with this weather, not the error in the forecast."),
        panel([html.Div(id="coefs")],
              title="The coefficients behind these numbers",
              note="Read from model_coefficients.csv. Monday is the baseline, so "
                   "its effect is zero and every other day is read against it. "
                   "The bars are scaled to the largest term other than the "
                   "intercept, whose own bar is therefore clipped."),
    ])


# -------------------------------------------------------------------- explore
def filtered(years):
    lo, hi = years
    y = bikes["date"].dt.year
    return bikes[(y >= lo) & (y <= hi)]


@app.callback(Output("summary", "children"), Input("year-range", "value"))
def summary(years):
    d = filtered(years)
    wk = d[~d["weekend"]]["bikes_hired"].mean()
    we = d[d["weekend"]]["bikes_hired"].mean()
    return [stat(f"{len(d):,}", "days in view"),
            stat(f"{d['bikes_hired'].mean():,.0f}", "average daily hires"),
            stat(f"{wk:,.0f}", "weekday average"),
            stat(f"{we:,.0f}", "weekend average")]


@app.callback(Output("scatter", "figure"), Output("scatter-title", "children"),
              Input("weather-var", "value"), Input("colour-by", "value"),
              Input("year-range", "value"), Input("theme", "data"))
def scatter(var, colour, years, name):
    p = palette(name)
    d = filtered(years)
    unit = WEATHER[var][1]
    axis = f"{WEATHER[var][0]}{f' ({unit})' if unit else ''}"
    kw = dict(x=var, y="bikes_hired", trendline="ols", opacity=0.5,
              labels={var: axis, "bikes_hired": "Daily hires",
                      "day_label": "", "season_name": ""})
    if colour == "weekend":
        # The weekday and weekend pairing is the product's semantic coding rather
        # than two slots off the palette, so it is named rather than taken in order.
        kw.update(color="day_label",
                  color_discrete_map={"Weekday": p["accent"], "Weekend": p["weekend"]},
                  category_orders={"day_label": ["Weekday", "Weekend"]})
    elif colour == "season_name":
        kw.update(color="season_name", color_discrete_map=p["season"],
                  category_orders={"season_name": ["Winter", "Spring", "Summer",
                                                   "Autumn"]})
    fig = px.scatter(d, **kw)
    fig.update_traces(marker=dict(size=5, line=dict(width=0)),
                      selector=dict(mode="markers"))
    fig.update_traces(line=dict(width=2.5), selector=dict(mode="lines"))
    if colour == "none":
        fig.update_traces(marker_color=p["accent"], selector=dict(mode="markers"))
        fig.update_traces(line_color=p["weekend"], selector=dict(mode="lines"))
    fig.update_layout(**base_layout(p, height=430,
                                    margin=dict(l=64, r=16, t=30, b=48)),
                      hovermode="closest")
    fig.update_xaxes(**spikes(p))
    return fig, f"Hires against {WEATHER[var][0].lower()}"


@app.callback(Output("by-day", "figure"),
              Input("year-range", "value"), Input("theme", "data"))
def by_day(years, name):
    p = palette(name)
    d = (filtered(years).groupby("day_of_week", observed=True)["bikes_hired"]
         .mean().reindex(DAY_ORDER))
    gap = d.values - d.mean()
    fig = go.Figure(go.Bar(
        x=[DAY_FULL[k] for k in d.index], y=d.values,
        marker_color=[p["weekend"] if k in ("Sat", "Sun") else p["accent"]
                      for k in d.index],
        marker_line_width=0, customdata=gap,
        texttemplate="%{y:,.0f}", textposition="outside", cliponaxis=False,
        textfont=dict(family=FONT, size=12.5, color=p["ink"]),
        hovertemplate="<b>%{x}</b><br><b>%{y:,.0f}</b> hires<br>"
                      "%{customdata:+,.0f} vs the weekly mean<extra></extra>"))
    fig.update_layout(**base_layout(p, height=380,
                                    margin=dict(l=12, r=12, t=28, b=44)),
                      showlegend=False, bargap=0.34)
    fig.update_xaxes(fixedrange=True, tickvals=[DAY_FULL[k] for k in d.index],
                     ticktext=list(d.index))
    fig.update_yaxes(visible=False, fixedrange=True, range=[0, d.max() * 1.17])
    return fig


@app.callback(Output("over-time", "figure"),
              Input("year-range", "value"), Input("theme", "data"))
def over_time(years, name):
    """Monthly average hires, where the summer-winter swing dominates everything
    else and the level shifts twice: up through 2022, then down from 2023."""
    p = palette(name)
    d = filtered(years)
    # to_period drops the timezone and says so on every call, so it goes first
    month = d["date"].dt.tz_localize(None).dt.to_period("M")
    m = (d.groupby(month)["bikes_hired"].mean()
         .rename_axis("month").reset_index())
    m["month"] = m["month"].dt.to_timestamp()
    fig = go.Figure(go.Scatter(
        x=m["month"], y=m["bikes_hired"], mode="lines",
        line=dict(color=p["accent"], width=2), fill="tozeroy",
        fillcolor=p["accent_soft"],
        hovertemplate="%{x|%B %Y}<br><b>%{y:,.0f}</b> hires a day<extra></extra>"))
    fig.update_layout(**base_layout(p, height=380,
                                    margin=dict(l=56, r=12, t=52, b=36)),
                      showlegend=False, hovermode="x unified",
                      shapes=winter_bands(p, years[0], years[1]))
    fig.update_xaxes(rangeslider=rangeslider(p), rangeselector=rangeselector(p),
                     **spikes(p))
    fig.update_yaxes(rangemode="tozero")
    return fig


# -------------------------------------------------------------------- predict
def prediction(weather, p, ceiling=None):
    """One chart and one table over the same days.

    The ceiling is the axis top and the scale the table's magnitude bars are read
    against, so bar and chart always agree. January is a fixed week and takes its
    own maximum; the five-day panel is refetched every morning and takes the shared
    one, or its scale would change under the planner overnight.
    """
    pred = predict(weather, coefficients)
    days = pred["day_of_week"].astype(str)
    weekend = days.isin(["Sat", "Sun"])
    dates = pd.to_datetime(pred["date"])
    labels = [f"{d:%a}<br>{d.day} {d:%b}" for d in dates]
    long_dates = [f"{d:%A} {d.day} {d:%B %Y}" for d in dates]

    # A single number hides how wide the model's own error is. The band is the
    # 95% prediction interval, taken as 1.96 residual standard errors: that leaves
    # out the uncertainty in the coefficients, which the notebook measures at 0.2%
    # of the width. Floored at zero for the same reason the point estimate is.
    half = 1.96 * coefficients.get("residual_se", RESIDUAL_SE)
    lo = (pred["predicted_hires"] - half).clip(lower=0)
    hi = pred["predicted_hires"] + half
    ceiling = ceiling or float(hi.max()) * 1.08

    custom = list(zip(long_dates, pred["temp"], pred["precip"], pred["windspeed"],
                      lo.round(0), hi.round(0)))
    fig = go.Figure(go.Bar(
        x=labels, y=pred["predicted_hires"],
        marker_color=[p["weekend"] if w else p["accent"] for w in weekend],
        marker_line_width=0, customdata=custom,
        error_y=dict(type="data", symmetric=False,
                     array=(hi - pred["predicted_hires"]).tolist(),
                     arrayminus=(pred["predicted_hires"] - lo).tolist(),
                     color=p["faint"], thickness=1.2, width=5),
        # Inside the bar rather than above it: the whisker owns the space above,
        # and a label sitting on top of it reads as part of the line.
        texttemplate="%{y:,.0f}", textposition="inside", insidetextanchor="end",
        cliponaxis=False, textfont=dict(family=FONT, size=12.5, color=p["accent_ink"]),
        hovertemplate="<b>%{customdata[0]}</b><br><b>%{y:,.0f}</b> hires<br>"
                      "95%% of days like this fall between "
                      "%{customdata[4]:,.0f} and %{customdata[5]:,.0f}<br>"
                      "%{customdata[1]:.1f} °C · %{customdata[2]:.1f} mm · "
                      "%{customdata[3]:.1f} km/h<extra></extra>"))
    fig.update_layout(**base_layout(p, height=330,
                                    margin=dict(l=12, r=12, t=30, b=52)),
                      showlegend=False, bargap=0.34)
    fig.update_xaxes(type="category", fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True, range=[0, ceiling])

    # The table shows the model's own drivers, so changing the model changes the
    # columns without anything here needing to know which ones they are.
    shown = [t for t in drivers(coefficients) if t != LAG_TERM]
    rows = []
    for (_, r), w, d in zip(pred.iterrows(), weekend, dates):
        value = r["predicted_hires"]
        tone = p["weekend_soft"] if w else p["accent_soft"]
        cells = [html.Td(f"{d.day} {d:%b %Y}"),
                 html.Td(DAY_FULL[str(r["day_of_week"])])]
        cells += [html.Td(signed(f"{r[c]:,.{COLUMNS[c][1]}f}"), className="num")
                  for c in shown]
        cells.append(html.Td(f"{value:,.0f}", className="num out",
                             style=magnitude(value / ceiling * 100, tone)))
        rows.append((cells, bool(w)))
    header = ([("Date", False), ("Day", False)]
              + [(COLUMNS[c][0], True) for c in shown]
              + [("Predicted hires", True)])
    return html.Div([dcc.Graph(figure=fig, config=STATIC_CONFIG,
                               style=CHART_BARS),
                     data_table(header, rows)])


@app.callback(Output("jan", "children"), Input("theme", "data"))
def january(name):
    """January is a week that has already happened, so its weather is fixed and a
    copy of it is kept beside the app. Open-Meteo rate limits by IP and a shared
    host runs into that limit on other people's traffic, which would otherwise
    leave the panel the assignment names showing an error message."""
    p = palette(name)
    try:
        w = open_meteo_history("London", "2026-01-01", "2026-01-07")
        note = None
    except Exception as e:
        w = pd.read_csv(JANUARY_FALLBACK, parse_dates=["date"])
        note = html.P(f"Read from the copy stored with the app, because the live "
                      f"archive answered {reason(e)}. The week is in the past, so "
                      f"the two are the same weather.", className="panel-note")
    seeded = html.P(f"Each day's hires feed into the next as the model's lag term. "
                    f"The week starts from {seed_lag(w['date'].min())[1]}.",
                    className="panel-note") if LAG_TERM in coefficients else None
    out = prediction(w, p)
    return html.Div([x for x in (note, seeded, out) if x is not None])


@app.callback(Output("forecast", "children"), Input("theme", "data"))
def forecast(name):
    """A stale forecast with its date on it beats an error message. Open-Meteo
    limits by IP, and on a shared host the limit is reached on other people's
    traffic, so the newest answer that did arrive is kept on disk and served with
    a note whenever the next one does not."""
    p = palette(name)
    note = None
    try:
        w = open_meteo("London", 5)
        w.to_csv(FORECAST_CACHE, index=False)
    except Exception as e:
        try:
            w = pd.read_csv(FORECAST_CACHE, parse_dates=["date"])
        except Exception:
            return html.Div(f"Open-Meteo forecast unavailable: {e}",
                            style={"color": p["red"], "fontSize": "14px"})
        taken = pd.to_datetime(w["date"]).min()
        note = html.P(f"The live forecast answered {reason(e)}, so this is the "
                      f"last one that arrived, issued {taken:%-d %B %Y}.",
                      className="panel-note")
    seeded = html.P(f"Each day's hires feed into the next as the model's lag term. "
                    f"The run starts from {seed_lag(w['date'].min())[1]}.",
                    className="panel-note") if LAG_TERM in coefficients else None
    out = prediction(w, p, ceiling=FORECAST_CEILING)
    return html.Div([x for x in (note, seeded, out) if x is not None])


@app.callback(Output("coefs", "children"), Input("theme", "data"))
def coefs(name):
    """Scaled to the largest term other than the intercept: against the intercept
    the rest are slivers and the bar stops saying anything.

    The metadata rows are not terms and are left out of both the table and the
    scale, or the residual standard error would be the tallest bar in a chart of
    coefficients.

    Terms under ten carry three decimals rather than one. The lag is the reason:
    at one decimal a slope of 0.4601 prints as 0.5, which is a different model.
    """
    p = palette(name)
    terms = {k: v for k, v in coefficients.items() if k not in METADATA}
    others = [abs(v) for k, v in terms.items() if k != "Intercept"]
    top = max(others) or 1
    rows = []
    for term, value in terms.items():
        pct = min(abs(value) / top * 100, 100)
        tone = p["red_soft"] if value < 0 else p["accent_soft"]
        text = signed(f"{value:,.3f}" if abs(value) < 10 else f"{value:,.1f}")
        rows.append(([html.Td(term),
                      html.Td(text, className="num out neg" if value < 0
                              else "num out", style=magnitude(pct, tone))], False))
    return data_table([("Term", False), ("Coefficient", True)], rows, narrow=True,
                      width="coef")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)

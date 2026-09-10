# London bikes — daily hire dashboard

A Dash app for TfL planners. It explores fifteen years of Santander Cycles hires
against the weather, and forecasts demand from live Open-Meteo data using the
regression model fitted in `bikes_assignment_group8.ipynb`.

AM01 Applied Statistics with Python, Study Group 8.

## What it does

### Explore

Pick a weather variable, colour the points by weekend or by season, and narrow the
year range. The scatter carries an OLS trendline per group; two more charts show
average hires by day of week and the monthly average over the whole series, with
December to February shaded.

A planner never has to read a number off a chart by eye. Hover any mark for the
exact figure, its date and the weather behind it; the day-of-week bars also give
the gap to the weekly mean. Zoom by dragging on the scatter or the monthly chart,
drag the window under the monthly chart, pick 1y / 3y / 5y / All above it, or click
a legend key to isolate one group. The seven day-of-week bars do not zoom, because
there is nothing inside seven bars to zoom into.

### Predict

Applies `model_coefficients.csv` to real weather from Open-Meteo: the first week
of January 2026 from the historical archive, and the next five days from the live
forecast. Each day is shown with the weather that produced it, and each table
figure sits on a bar showing its size against the chart's own ceiling.

The model reads yesterday's hires, so each panel is a chain rather than seven
independent sums: the first day starts from a figure the panel names, and every day
after it starts from the day before. January starts from a real number, the 11,358
hires recorded on 31 December 2025. The five-day panel is past the end of the data,
so it starts from the average of that weekday in that season and says so.

### Both

Light and dark are two presentations of one design rather than an inversion, and
the page remembers which one you chose. The layout holds from 1440 px down to a
phone, and nothing moves for a reader who has asked their system for less motion.

## The model

```
bikes_hired ~ temp + precip + windspeed + visibility + solarradiation
            + sealevelpressure + C(season_name) + C(day_of_week) + bikes_hired_lag1
```

Fitted on 4,382 days from 2 January 2014. The file starts a day earlier; the first
row has no day before it, which is what the lag costs. Adjusted R² 0.728, residual
standard error 4,784 hires a day. Every weather term and the lag clear the 1% level
by a wide margin. Two of the six day contrasts do not clear 5%, Wednesday and
Thursday against Monday at 0.87 and 0.57, while the seven days taken together are
worth keeping (F-test p = 2.0e-128). Across the seven numeric predictors the largest
VIF is 1.79, on the lag.

The predictors are what `open_meteo.py` can hand the app on the morning of the
forecast, and three of them took work before that was true. Solar radiation is
Open-Meteo's shortwave radiation put on the training file's scale by a fitted line;
over 2024 the calibrated series correlates with the file at 0.898 and the two means
sit 1.75 W/m² apart. Visibility needs both a second source and a scale, and is the
weakest link in the whole app; the section below says how weak. Yesterday's hires
exist for no future day, so the app runs the model forward a day at a time, feeding
each prediction back in as the next day's lag and starting from a real figure where
the data reaches.

`model_coefficients.csv` carries one row per model term and nothing else, so the
residual standard error the prediction band is drawn from is not in it. The app
reads it from the file when a row exists and otherwise from `RESIDUAL_SE` in
`app.py`, quoted from the same summary as the two figures above.

The app does not refit anything. It reads the coefficients from
`model_coefficients.csv`, which the notebook writes, and the CSV decides the model:
`predict()` multiplies each term by the column of the same name and picks the day
and season terms off the calendar, so changing the model changes the forecast tables'
columns without a line of app code. The lag is the one term with no column to read,
and the only one the app knows by name. A term the weather helper cannot supply stops
the app with a message naming it rather than a `KeyError` further down.

## Files

| File | What it is |
|---|---|
| `app.py` | the whole application: palettes, motion tokens, layout, callbacks |
| `open_meteo.py` | the course weather helper, with pressure, season, and calibrated solar radiation and visibility added |
| `model_coefficients.csv` | the fitted model, one row per term |
| `january_2026.csv` | that week's weather, kept for when Open-Meteo rate limits the host |
| `forecast_seed.csv` | the last forecast that arrived, for the same reason |
| `pyproject.toml` | the project and its pinned dependencies |
| `uv.lock` | the exact resolve, so the build is reproducible |
| `gunicorn.conf.py` | the bind address, read from `PORT` |
| `render.yaml` | Render service definition |

Everything the app needs is in `app.py`, which is why it is one long file rather
than several short ones: the assignment's repository list names `app.py`,
`open_meteo.py`, `model_coefficients.csv`, `pyproject.toml` and the deployment
files, so a second module would be the one thing nobody remembers to commit.

## Running it locally

```bash
uv sync
uv run python app.py
```

Then open http://127.0.0.1:8050.

## Deploying to Render

1. Push this folder to a public GitHub repository, `uv.lock` included.
2. On Render, create a **New Web Service** and connect that repository.
3. Build command `pip install uv && uv sync`, start command
   `uv run gunicorn app:server`. `render.yaml` carries the same two, so it makes
   no difference whether they are typed in or read from the file.
4. The free plan is enough. The first request after an idle period takes a few
   seconds while the service wakes.

`app.py` exposes `server = app.server` for gunicorn. The port is the one thing the
start command above does not say: gunicorn ignores `PORT` and defaults to
`127.0.0.1:8000` whatever Render sets, so a bare `gunicorn app:server` binds where
nothing is listening and the deploy fails its health check. `gunicorn.conf.py` is
loaded automatically from the working directory and binds `0.0.0.0:$PORT`, which
is why the start command can stay exactly as the assignment writes it.

The pins are pins rather than lower bounds, and they matter more than usual here.
Dash 4 renamed the classes its dropdown and slider paint themselves with, so an
older resolve installs an app that runs correctly wearing none of its styling.

## What to know about the numbers

The training weather and the serving weather are not the same measurement. The model
is fitted on the columns in `london_bikes.csv` and applied to Open-Meteo. Over the
whole of 2024 the six agree to very different degrees:

| | r | file mean | Open-Meteo mean |
|---|---|---|---|
| `sealevelpressure` | 0.999 | 1,014.00 | 1,014.19 |
| `temp` | 0.994 | 12.56 | 11.75 |
| `solarradiation` | 0.898 | 79.26 | 77.51 |
| `windspeed` | 0.885 | 21.59 | 14.62 |
| `precip` | 0.817 | 1.82 | 2.44 |
| `visibility` | 0.720 | 14.90 | 20.63 |

Pressure and temperature are effectively the same reading; wind is the one that is
plainly not, running seven units high in the file. Solar radiation sits where it does
because it was calibrated onto the file's scale rather than taken as it came. Hold the
day and the lag fixed and swap only the weather source, and 2024's predictions move by
1,052 hires on average, with a mean absolute difference of 1,235, which is 26% of the
model's own residual standard error.

Visibility is worth its own paragraph, because it is the one column that will not
transfer honestly. Open-Meteo's ERA5 archive has no visibility at all: ask it for the
first week of January and all 168 hours come back null, which is why the app reads that
field from the historical forecast archive instead. Scale is the harder half. The
training column is not a stable measurement of London: its annual mean runs 20.0 km in
2014, climbs to 32.1 in 2022, drops to 14.9 in 2024 and returns to 25.3 in 2025, and
its standard deviation moves with it, from 4.3 to 14.1. Open-Meteo's own series over
the same years sits at 13.2 to 15.1 with a standard deviation near 5. London's air did
not do that; the file's provider changed what it was reporting. So no single line can
map one onto the other for every year. The app uses one fitted over the 1,096 days the
two sources share, `file = 6.5637 + 0.9826 × open-meteo km`, which correlates at 0.480
and carries a mean absolute error of 6.95 km. At 115 hires per kilometre that is worth
about 800 hires a day of noise on its own, and it is most of the 26% above.

Predictions are floored at zero. A linear model has nothing stopping it going
negative, and on the two wettest days in the training file it does, reaching −10,953.
Those are the 9th and 10th of February 2021, carrying 168 mm and 151 mm of recorded
precipitation against a median of 0.14 mm; Open-Meteo reports 1.0 mm and 0.0 mm for
the same dates. See the Part 4 write-up in the notebook.

Each forecast carries a 95% band, drawn as the estimate plus or minus 1.96 residual
standard errors. That is the interval for a single day with that weather, not the
error in the point estimate, and it is wide: ±9,377 hires. It leaves out the
uncertainty in the coefficients themselves, which measures 0.2% of
the width, and the compounding a recursive forecast adds as it runs.

The five-day chart's axis is pinned to 60,000 while January takes its own maximum.
That panel is refetched every morning, and a free axis would rescale overnight, so
Tuesday's chart and Wednesday's could not be compared at a glance. The ceiling sits
above 49,300, the highest this panel's own arithmetic reaches on the training
weather, plus the band on top of it.

## Notes

Open-Meteo is free and needs no API key. If either call fails the app says so on
the page rather than crashing, so a network problem during a demo shows a message
instead of a stack trace.

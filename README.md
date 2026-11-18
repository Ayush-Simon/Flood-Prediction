# NYC Flood Watch: urban flood depth prediction and early warning

Final-year B.Tech project. Predicts street flooding in New York City from rainfall, using the two-stage
method of Huang, Hong & Sun (2025), *Urban flood depth prediction using an improved LSTM model incorporating
precipitation forecasting* (Natural Hazards), and shows it on a live, map-based dashboard.

- **Stage 1, rain forecasting:** ARIMA, XGBoost and ANN (hourly), and an ANN that forecasts the next 40 minutes of rain (5-minute steps).
- **Stage 2, flood depth:** persistence, XGBoost, LSTM, attention-LSTM, and the project's own **ANN-LSTM** and **CNN-LSTM**, predicting depth 5–40 minutes ahead at 27 FloodNet street sensors.
- **Street risk:** an XGBoost model scores all 99,323 NYC streets for a storm, using drains, street size, the city's stormwater flood maps and past 311 flooding reports.
- **Dashboard:** live weather, rain and flood map with a route finder that avoids likely flooded streets, plus a replay of 58 real 2026 storms.

The study city changed from Bhubaneswar to New York because no public, timestamped street flood-depth data exists for Bhubaneswar.

## Main results (test storms, July–October 2026)

| Model | R² 5 min | R² 20 min | R² 40 min | Floods caught early |
| --- | --- | --- | --- | --- |
| Persistence (baseline) | 0.93 | 0.44 | −0.10 | 0% |
| **LSTM (standard)** | **0.94** | **0.52** | **0.11** | 0% |
| ANN-LSTM | 0.93 | 0.49 | 0.05 | 3% |
| CNN-LSTM | 0.93 | 0.48 | −0.01 | 1% |
| CNN-LSTM + ANN rain (live model, average of 3 runs) | 0.90 | 0.47 | 0.07 | 3% |
| LSTM + perfect rain forecast (upper bound) | 0.94 | 0.53 | 0.19 | 33% |

The standard LSTM has the lowest error of the neural models (significantly better than ANN-LSTM and CNN-LSTM, p ≤ 0.01), but a three-parameter physical bucket model is just as accurate (p = 0.56). Counting false alarms, warnings catch about a third of floods 10–15 minutes ahead at about four false alarms per real flood. Checked against measured sensor water, the street model barely beats rain alone and over-ranks complaint-heavy areas. Early warning
depends on the rain forecast: with perfect rain the LSTM catches a third of floods 40 minutes ahead.
Full results: `reports/` and the results chapter.

## Run the dashboard

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy xgboost requests
python3 backend/build_store.py     # first time only: loads backend/data/ into the secure store
python3 backend/app.py             # website + live updates every 15 minutes
```

Open http://localhost:8000. Use **Past storms** in the header to replay 2026 storms.
Without live weather: `python3 backend/app.py --no-live`.
Record of the live run for the report: `python3 backend/export_history.py --days 3`.

## Hosting on Netlify (live forecast every 30 minutes)

Netlify's free plan allows only about 20 production deploys a month, so the site is not redeployed for every forecast.
`.github/workflows/netlify.yml` has two jobs:
- **live** (every 30 minutes): runs one live update with `backend/build_live_feed.py` and force-pushes `live.json` and
  `history.json` to the `live-data` branch (a single commit, replaced each time, so the repository does not grow).
- **site** (when the code changes, or "Run workflow"): builds the site with `backend/build_static.py --live-url ...`
  and deploys it to Netlify. The page reads the live forecast from the `live-data` branch and falls back to the copy
  built into the site if GitHub cannot be reached.

One-time setup (the repository must be public, because the page reads `live.json` without signing in):
1. Create a free Netlify account and a personal access token at https://app.netlify.com/user/applications.
2. GitHub repo > Settings > Secrets and variables > Actions > New repository secret: `NETLIFY_AUTH_TOKEN`.
3. GitHub repo > Settings > Actions > General > Workflow permissions: choose "Read and write permissions", Save.
4. Actions tab > "Netlify site and live forecast" > Run workflow. The site job's summary shows the address
   (default name `nyc-flood-watch-<your GitHub name>`; set the variable `NETLIFY_SITE_NAME` to choose another).

## Hosting on Vercel instead (switched off)

`.github/workflows/deploy.yml` is kept for reference and runs only when started by hand. To use it, restore its schedule and push triggers and turn off `netlify.yml`.

Vercel only serves files, so a GitHub Action does the live work and deploys the result:
`.github/workflows/deploy.yml` runs every 30 minutes (and on every push to `main`). It builds the data store,
runs one live update, writes the site as plain files with `backend/build_static.py`, and deploys `site/` to Vercel.
Nothing is committed back, so the repository stays small. Past storms and the Model Lab work the same as locally;
the live forecast refreshes every 30 minutes instead of 15, and the page says so if an update is late.

One-time setup:
1. Create a free Vercel account and a token at https://vercel.com/account/tokens.
2. In the GitHub repo: Settings > Secrets and variables > Actions > New repository secret: `VERCEL_TOKEN`.
3. Actions tab > "Live forecast and deploy" > Run workflow. The run summary shows the site's address.
4. Add that address as a repository variable `SITE_URL` (same settings page, Variables tab), so later runs
   keep the 30-day history and reuse the last forecast if a live update fails.

Build the static site locally: `python3 backend/build_static.py` (add `--update` for a fresh live forecast).

## Folder structure

| Folder | Contents |
| --- | --- |
| `frontend/` | The website: `index.html`, `css/` (look and layout), `js/app.js` (map, charts, routing), `assets/` (font, icon) |
| `backend/` | Web server and API (`server.py`), data store (`store.py`, `store/flood.db`), live updater (`updater.py`), model inference (`flood_models.py`) |
| `backend/data/` | Published datasets (street network, storm replays, results) that the store is built from |
| `scripts/` | The pipeline, numbered in run order (see below) |
| `models/` | Trained road model (XGBoost) and CNN-LSTM + ANN weights |
| `reports/` | Result tables (CSV) and figures (PNG) |
| `data/processed/` | Cleaned data small enough for GitHub |
| `data/raw/` | Sensor list only; download the rest as below |

## How the data is kept safe

- All website data lives in one SQLite database, `backend/store/flood.db`, private to your user account (folder 700, files 600) and kept out of Git.
- The server opens it read-only. Only `build_store.py` and the live updater can write.
- Every dataset has a SHA-256 checksum that is checked when it is read; a changed row is refused. `python3 backend/build_store.py --verify` checks them all.
- The website can only ask for named datasets through `/api/...`; it cannot read files. Static files come only from `frontend/`.
- The server listens on this computer only (127.0.0.1), sends strict browser security headers (Content-Security-Policy and others), limits each address to 300 requests a minute, and hides error details.
- The one write action, `POST /api/refresh`, needs the admin token in `backend/store/.admin_token`.
- Every live update is also kept for 30 days (`/api/history`), with an audit log of imports and updates.

## Pipeline

| Script | Step |
| --- | --- |
| `02c_merge_rain_depth.py` | Hourly depth + rain, storm windows, offline sensor check |
| `03_rain_forecast.py` | Rain forecasting: persistence, ARIMA, XGBoost, ANN |
| `04_depth_models.py` | Hourly depth models |
| `05a`, `05b`, `05c` | Street table, street features, road risk model |
| `06a_live_tables.py` | Tables the live app needs |
| `07a`, `07b` | 5-minute gauge rain and 5-minute depth |
| `07c_depth_models_5min.py` | 5-minute depth models (LSTM variants, XGBoost) |
| `07d_ann_cnn_lstm.py` | ANN-LSTM and CNN-LSTM |
| `08_build_dashboard_data.py` | Storm replay files for the dashboard (`backend/data/`) |
| `09a_export_live_models.py` | Exports CNN-LSTM + ANN to JSON for the live app |
| `07e_recursive_and_future_work.py` | The paper's recursive ANN-LSTM and future-work experiments |
| `10a`, `10b`, `10c` | Confidence intervals, significance tests, gauge distance, honest warning scores, physics (bucket) baseline |
| `10d_street_model_vs_water.py` | Street model checked against measured sensor water; complaint bias |
| `13a_export_lab_models.py` | Models for the Model Lab (run in the browser) |
| `14a`–`14d` | Perfect-rain (oracle) experiment, per-flood warnings, 311 exposure correction |
| `15a_notebook_extensions_nyc.py` | Improved LSTM vs Feature-enhanced LSTM, accuracy by rainfall condition, risk levels |
| `16a_rain_now_past.py` | Main model's 5/15/30/40-minute forecasts on past storms ("Raining now") |
| `16b_feature_lstm_export.py` | Improved / Feature-enhanced LSTM for the website and Model Lab |
| `17a_ensemble_live_model.py` | Live model as the average of 3 training runs (matches the reported R² 0.90 at 5 min) |
| `17b_floodnet_live_depth.py` | Current sensor depths from FloodNet (api.fieldkit.org, project 174) for live forecasts |
| `17c_tree_model_tuning.py` | XGBoost, LightGBM and Random Forest tuned with time-based cross-validation |

## Data sources (all free and public)

| Data | Source |
| --- | --- |
| Street flood depth | FloodNet NYC on NYC Open Data (datasets `aq7i-eu5q`, `kb2e-tjy3`); live readings from the FloodNet dashboard data (api.fieldkit.org) |
| 1-minute rain | NOAA airport gauges (Central Park, LaGuardia, JFK) via Iowa Environmental Mesonet |
| Hourly rain, live weather | Open-Meteo |
| Streets, storm drains, stormwater flood maps, 311 reports | NYC Open Data (`inkn-q76z`, `2w2g-fk3i`, `9i7c-xyvv`, `erm2-nwe9`) |

Large raw and intermediate files (over about 20 MB) are not in the repository; the notebook
`01_data_exploration.ipynb` and the scripts download or rebuild them.

## Limitations

Tested on four months of storms; airport gauges are 4–25 km from most sensors; the road model learns from
311 reports, not measured water; live forecasts start from FloodNet's current readings when its public data
(api.fieldkit.org) answers, and from a dry street otherwise; live mode has not been validated on real storms. This is a student research prototype, not an official warning service.

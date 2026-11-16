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

## Data sources (all free and public)

| Data | Source |
| --- | --- |
| Street flood depth | FloodNet NYC on NYC Open Data (datasets `aq7i-eu5q`, `kb2e-tjy3`) |
| 1-minute rain | NOAA airport gauges (Central Park, LaGuardia, JFK) via Iowa Environmental Mesonet |
| Hourly rain, live weather | Open-Meteo |
| Streets, storm drains, stormwater flood maps, 311 reports | NYC Open Data (`inkn-q76z`, `2w2g-fk3i`, `9i7c-xyvv`, `erm2-nwe9`) |

Large raw and intermediate files (over about 20 MB) are not in the repository; the notebook
`01_data_exploration.ipynb` and the scripts download or rebuild them.

## Limitations

Tested on four months of storms; airport gauges are 4–25 km from most sensors; the road model learns from
311 reports, not measured water; live mode has no live sensor depths and has not been validated on real
storms. This is a student research prototype, not an official warning service.

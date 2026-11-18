# NYC Flood Watch: context for a new chat

Paste this whole file as the first message of a new chat, then say what you want to do next.

## Who and what
- Final-year project (IIT-style viva coming up). Reproduces Huang, Hong & Sun (2025, *Natural Hazards*): two-stage
  rain forecast then LSTM water-depth forecast. Study city switched from Bhubaneswar to New York City because NYC
  has public measured street-flood depth (FloodNet); Bhubaneswar has none.
- Project folder on my Mac: `~/Bhubaneswar-Flood-Prediction` (Python venv in `.venv`).
- GitHub: https://github.com/Ayush-Simon/Flood-Prediction (public). GitHub is not linked to Claude, so I push myself:
  `git add -A && git commit -m "..." && git push`.
- I prefer concise, casual answers and iterate with short prompts.

## Data
- FloodNet: 27 street sensors, water depth every minute, resampled to 5 minutes. Airport rain gauges (Central Park,
  LaGuardia, JFK) via IEM. Open-Meteo weather. NYC Open Data: about 99,000 streets, drains, flood maps, 311 complaints.
- Split by time: train up to 2025-12-31, validation Jan–Jun 2026, test Jul–Oct 2026.
- Training data export: `data/processed/nyc_flood_training_data.zip` (train / validation / test CSVs, README).

## Models (window = last 12 five-minute readings: depth, rain, rain-missing flag; output = depth 5…40 min ahead)
- LSTM (64 units), ANN-LSTM (an MLP first forecasts 40 min of rain), CNN-LSTM (Conv1d before the LSTM).
- Street risk: XGBoost on 311 flooding complaints. Physics baseline: bucket model depth(t+1) = r·d + k·max(0, rain − c).
- Results (test set): plain LSTM significantly beats ANN-LSTM (p = 0.01), CNN-LSTM (p < 0.001) and persistence
  (p = 0.002); bucket model ties the LSTM (p = 0.56). R² about 0.93 at 5 min, 0.46 at 20 min, about 0 at 40 min.
  Warnings catch about 1/3 of floods 10–15 min ahead with about 80% false alarms. Street model vs measured water:
  AUC 0.67 vs 0.64 for rain alone; complaint bias Spearman 0.57 (p = 0.002). Gauge distance doesn't explain the gap.
- Honest limits: live sensor depths not connected (live forecasts start dry = lower bound); never run through a real
  storm yet (next rain forecast 22 Nov 2026); street map learns from complaints, not measured water.

## System
- `backend/`: `app.py` (start: `python3 backend/app.py`, add `--host 0.0.0.0` in Codespaces, `--no-live` for no
  updates), `server.py` (read-only, security headers, CSP, rate limit, admin token), `store.py` (SQLite with
  checksums), `build_store.py` (run once), `updater.py` (live update every 15 min), `build_static.py` (site as plain
  files), `build_live_feed.py` (live forecast only), `data/` (storms, models, lab samples).
- `frontend/`: `index.html` (live and past-storm views, flood map with Esri satellite imagery, routing),
  `lab.html` (Model Lab: upload a CSV with time_utc, rain_mm, depth_in; press Run; ANN-LSTM / CNN-LSTM / Both;
  step-through panel, comparison charts). Models run in the browser (`js/models.js`, matches PyTorch to 3.4e-7 in).
- `requirements.txt` (website), `requirements-train.txt` (training scripts in `scripts/`).

## Hosting
- Netlify via `.github/workflows/netlify.yml`: the live job runs every 30 min and force-pushes the forecast to the
  `live-data` branch; the site job deploys to Netlify only when code changes (free plan: about 20 deploys a month).
  Needs the `NETLIFY_AUTH_TOKEN` secret and Settings > Actions > General > "Read and write permissions".
  Default address: https://nyc-flood-watch-ayush-simon.netlify.app
- Status: first run failed; a fix (build without a forecast) is on my Mac and needs pushing, then "Run workflow".
- Vercel workflow (`deploy.yml`) is kept but switched to manual only.
- Note for Claude: the Mac bridge can't write into `.github/`; write to `_to_delete/` and `cp` it in with the shell.
  Copy files to a freshly named staging path before committing, then check checksums on the Mac.

## Still to do
- Push the Netlify fix and get a green run.
- Live run through the 22 Nov 2026 rain; export with `backend/export_history.py`; add to the report.
- Rename the repo / folder (it still says Bhubaneswar) and explain the city switch in one sentence.
- Mock viva practice: input window, time split, R² (and RMSE, persistence baseline).
- Deck cover placeholders ([Your name], [Guide's name], [Department, Institute]).
- Optional ideas: Bhubaneswar news-report flood events, GRU/BiLSTM, more weather inputs, 4 risk levels with F1.

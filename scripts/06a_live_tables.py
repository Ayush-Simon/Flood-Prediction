"""Step 3a: small tables the live dashboard needs.
- data/processed/nyc_street_live_features.csv : per-street model inputs that don't depend on today's rain
  (road + drain + flood-map features, and complaint history up to the end of the complaint data)
- data/processed/nyc_rain_cells.csv : one location per rainfall grid cell, used to ask for the live forecast
Run from anywhere: python scripts/06a_live_tables.py"""
import os, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P, R = f"{ROOT}/data/processed", f"{ROOT}/data/raw"
info = json.load(open(f"{ROOT}/models/road_risk_model_info.json"))
T0 = pd.Timestamp(info["history_origin"])

st = pd.read_csv(f"{P}/nyc_street_features.csv")
cm = pd.read_csv(f"{P}/nyc_complaints_matched.csv", parse_dates=["date"])
asof = cm["date"].max() + pd.Timedelta(days=1)
yrs = (asof - T0).days / 365.25
cnt = cm.groupby(["street_id", "kind"]).size().unstack(fill_value=0)
st["hist_flood"] = st["street_id"].map(cnt["street_flood"]).fillna(0) / yrs
st["hist_drain"] = st["street_id"].map(cnt["drain_clogged"]).fillna(0) / yrs
st["grid"] = (st["x"] // 1500).astype(int) * 100000 + (st["y"] // 1500).astype(int)     # same 1,500 ft grid as training
g = st.groupby("grid").agg(n=("street_id", "size"), f=("hist_flood", "sum"))
st["area_flood"] = st["grid"].map(g["f"] / g["n"])

static = [f for f in info["features"] if not f.startswith(("rain_", "month_", "wet_"))]
keep = ["street_id", "name", "boro", "mid_lat", "mid_lon", "rain_cell"] + [f for f in static if f != "boro"]
st[keep].to_csv(f"{P}/nyc_street_live_features.csv", index=False)
print("Streets:", len(st), "| history as of", asof.date(), "| static inputs:", len(static))

# one representative point per rain cell = centre of the streets that use it
cells = st.groupby("rain_cell").agg(lat=("mid_lat", "mean"), lon=("mid_lon", "mean"), streets=("street_id", "size")).round(4)
cells.to_csv(f"{P}/nyc_rain_cells.csv"); print(cells.to_string())

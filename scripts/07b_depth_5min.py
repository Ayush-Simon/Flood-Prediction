"""Stage 7: 5-minute flood depth per sensor, paired with the nearest working rain gauge.
Usage: python 07b_depth_5min.py <raw_dir> <processed_dir> <out_dir>
Inputs : raw/floodnet_events.csv, raw/floodnet_sensors.csv,
         processed/nyc_storm_windows.csv (for the 27 sensors, terrain and offline sensor-years),
         out/nyc_rain_5min_stations.csv (from 07a)
Output : nyc_depth_rain_5min.csv  (only storm windows: sensor, time, depth, rain, flags, terrain)"""
import sys, ast
import numpy as np, pandas as pd
RAW, PROC, OUT = sys.argv[1:4]
STATIONS = {"NYC": (40.7790, -73.9692), "LGA": (40.7794, -73.8803), "JFK": (40.6386, -73.7622)}
MIN_COV = 0.6                      # a 5-min rain value is used only if >= 3 of its 5 minutes were reported
WET_MM = 0.25                      # one gauge tip (0.01 inch)
BEFORE, AFTER = 12, 36             # storm window: 1 h before the first rain/flood .. 3 h after the last

sw = pd.read_csv(f"{PROC}/nyc_storm_windows.csv", usecols=["Sensor ID", "time", "suspect_offline", "elevation_m",
                                                         "slope_deg", "height_above_low_in", "Latitude", "Longitude"])
sw["year"] = pd.to_datetime(sw["time"]).dt.year
sensors = sw.drop_duplicates("Sensor ID").set_index("Sensor ID")[["elevation_m", "slope_deg", "height_above_low_in", "Latitude", "Longitude"]]
offline = set(map(tuple, sw.loc[sw["suspect_offline"], ["Sensor ID", "year"]].drop_duplicates().values))
se = pd.read_csv(f"{RAW}/floodnet_sensors.csv").drop_duplicates("Sensor ID").set_index("Sensor ID")

ev = pd.read_csv(f"{RAW}/floodnet_events.csv")
ev = ev[ev["Sensor ID"].isin(sensors.index)].copy()
fmt = "%m/%d/%Y %I:%M:%S %p"
ev["start"] = pd.to_datetime(ev["Flood Start Datetime (GMT)"], format=fmt)
ev["end"] = pd.to_datetime(ev["Flood End Datetime (GMT)"], format=fmt)

rain = pd.read_csv(f"{OUT}/nyc_rain_5min_stations.csv", parse_dates=["time_utc"])
R = {s: g.set_index("time_utc") for s, g in rain.groupby("station")}
full = R["NYC"].index

def dist(a, b): return np.hypot((a[0] - b[0]) * 111.3, (a[1] - b[1]) * 111.3 * np.cos(np.radians(40.7)))

rows = []
for sid, g in ev.groupby("Sensor ID"):
    # minute depth inside events (straight line between readings; overlaps merged by max), then 5-min max
    parts = []
    for _, e in g.iterrows():
        secs = np.array(ast.literal_eval(e["Time Series Depth Timestamps (seconds)"]))
        dep = np.array(ast.literal_eval(e["Time Series Depth Values (inches)"]))
        mins = pd.date_range(e["start"].ceil("min"), e["end"].floor("min"), freq="min")
        parts.append(pd.Series(np.interp((mins - e["start"]).total_seconds(), secs, dep), index=mins))
        parts.append(pd.Series(dep, index=(e["start"] + pd.to_timedelta(secs, unit="s")).floor("min")))
    s = pd.concat(parts); s = s.groupby(s.index).max()
    depth = s.resample("5min", label="right", closed="right").max().reindex(full).fillna(0)

    installed = min(pd.Timestamp(se.loc[sid, "Date Installed"]), g["start"].min().floor("D"))
    depth[full < installed] = np.nan
    if pd.notna(se.loc[sid, "Date Removed"]):
        depth[full > pd.Timestamp(se.loc[sid, "Date Removed"]) + pd.Timedelta(days=1)] = np.nan
    yrs = pd.Series(full.year, index=full)
    depth[[(sid, y) in offline for y in yrs.values]] = np.nan

    # rain: nearest gauge whose 5-min value is well covered, otherwise the next nearest
    loc = (sensors.loc[sid, "Latitude"], sensors.loc[sid, "Longitude"])
    order = sorted(STATIONS, key=lambda k: dist(loc, STATIONS[k]))
    rain_mm = pd.Series(np.nan, index=full); src = pd.Series("", index=full)
    for k in order:
        ok = (R[k]["coverage"] >= MIN_COV) & rain_mm.isna()
        rain_mm[ok] = R[k].loc[ok, "rain_mm"]; src[ok] = k

    df = pd.DataFrame({"time": full, "depth_in": depth.values, "rain_mm": rain_mm.values, "gauge": src.values})
    trig = ((df["rain_mm"] >= WET_MM) | (df["depth_in"] > 0)).astype(int)
    win = (trig.rolling(AFTER + 1, min_periods=1).max() + trig[::-1].rolling(BEFORE + 1, min_periods=1).max()[::-1]) > 0
    df = df[win & df["depth_in"].notna()].copy()
    df["storm"] = (df["time"].diff() != pd.Timedelta(minutes=5)).cumsum()
    df.insert(0, "Sensor ID", sid)
    df["gauge_km"] = round(dist(loc, STATIONS[order[0]]), 1)
    for c in ["elevation_m", "slope_deg", "height_above_low_in"]: df[c] = sensors.loc[sid, c]
    rows.append(df)
    print(f"{sid[:40]:40s} windows {df['storm'].nunique():4d} | 5-min rows {len(df):6d} | flooded {int((df['depth_in']>0).sum()):5d} | "
          f"rain missing {df['rain_mm'].isna().mean():.1%} | nearest gauge {order[0]} {df['gauge_km'].iloc[0]} km")

out = pd.concat(rows, ignore_index=True)
out["storm_id"] = out["Sensor ID"] + "_" + out["storm"].astype(str); out = out.drop(columns="storm")
out.to_csv(f"{OUT}/nyc_depth_rain_5min.csv", index=False)
print("\nRows:", len(out), "| flooded 5-min steps:", int((out["depth_in"] > 0).sum()), "| storms:", out["storm_id"].nunique())
print("Flooded steps with gauge rain missing:", f"{out.loc[out['depth_in']>0, 'rain_mm'].isna().mean():.1%}")

"""Phase 2, Step 2.3: join hourly depth + rainfall + terrain, build storm windows,
flag sensor-years that look offline. Reads data/raw + data/processed/nyc_depth_hourly.csv,
writes only to data/processed/."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW, PROC = f"{ROOT}/data/raw", f"{ROOT}/data/processed"

WET_MM = 0.5          # an hour counts as rainy at >= 0.5 mm
BEFORE, AFTER = 12, 6 # window: 6 h before first rain .. 12 h after last rain
                      # (hour t is in a window if a rainy hour lies in [t-12, t+6])

# --- load ---
depth = pd.read_csv(f"{PROC}/nyc_depth_hourly.csv")
depth["time"] = pd.to_datetime(depth["time"], utc=True).dt.tz_convert(None)
rain = pd.read_csv(f"{RAW}/nyc_rain_hourly_openmeteo.csv")
rain["time"] = pd.to_datetime(rain["time"])
rain = rain.rename(columns={"precipitation": "rain_mm"})
terr = pd.read_csv(f"{RAW}/nyc_sensor_terrain.csv")
se = (pd.read_csv(f"{RAW}/floodnet_sensors.csv").drop_duplicates("Sensor ID")
      [["Sensor ID", "Borough", "Latitude", "Longitude",
        "Sensor Ground Height Above Local Low Point (inches)"]]
      .rename(columns={"Sensor Ground Height Above Local Low Point (inches)": "height_above_low_in"}))

df = (depth.merge(rain, on=["Sensor ID", "time"], how="left")
           .merge(terr, on="Sensor ID", how="left")
           .merge(se, on="Sensor ID", how="left")
           .sort_values(["Sensor ID", "time"]).reset_index(drop=True))
assert df["rain_mm"].notna().all(), "rain missing for some hours"

# --- storm windows (per sensor) ---
def windows(g):
    wet = (g["rain_mm"] >= WET_MM).astype(int)
    flood = (g["depth_in"] > 0).astype(int)
    trig = (wet | flood)                          # never drop a flooded hour
    back = trig.rolling(BEFORE + 1, min_periods=1).max()                # trigger in [t-12, t]
    fwd = trig[::-1].rolling(AFTER + 1, min_periods=1).max()[::-1]      # trigger in [t, t+6]
    inwin = ((back + fwd) > 0) & g["depth_in"].notna()                  # only observed hours
    start = inwin & ~inwin.shift(fill_value=False)
    sid = start.cumsum().where(inwin)
    return pd.DataFrame({"in_storm": inwin, "storm_n": sid}, index=g.index)

w = df.groupby("Sensor ID", group_keys=False).apply(windows)
df = df.join(w)
df["storm_id"] = np.where(df["in_storm"],
                          df["Sensor ID"] + "_" + df["storm_n"].fillna(0).astype(int).astype(str), None)
df = df.drop(columns="storm_n")

# --- storm-level table ---
st = (df[df["in_storm"]].groupby(["Sensor ID", "storm_id"])
        .agg(start=("time", "min"), end=("time", "max"), hours=("time", "size"),
             peak_rain_mm=("rain_mm", "max"), total_rain_mm=("rain_mm", "sum"),
             peak_depth_in=("depth_in", "max")).reset_index())
st["flooded"] = st["peak_depth_in"] > 0
st["year"] = st["start"].dt.year

# flood with (almost) no rain in our data = rain/depth mismatch
st["flood_without_rain"] = st["flooded"] & (st["peak_rain_mm"] < WET_MM)

# --- suspected offline sensor-years ---
# threshold = median peak rain of storms that DID flood at that sensor;
# a sensor-year with >=3 storms above it and zero floods looks offline.
thr = st[st["flooded"] & ~st["flood_without_rain"]].groupby("Sensor ID")["peak_rain_mm"].median()
st["above_thr"] = st["peak_rain_mm"] >= st["Sensor ID"].map(thr)
sy = st.groupby(["Sensor ID", "year"]).agg(big_storms=("above_thr", "sum"),
                                            floods=("flooded", "sum")).reset_index()
sy["suspect_offline"] = (sy["big_storms"] >= 3) & (sy["floods"] == 0)
df["year"] = df["time"].dt.year
df = df.merge(sy[["Sensor ID", "year", "suspect_offline"]], on=["Sensor ID", "year"], how="left")
df["suspect_offline"] = df["suspect_offline"].fillna(False).astype(bool)
st = st.merge(sy[["Sensor ID", "year", "suspect_offline"]], on=["Sensor ID", "year"], how="left")

# --- save ---
cols = ["Sensor ID", "time", "rain_mm", "depth_in", "gap_filled", "in_storm", "storm_id",
        "suspect_offline", "elevation_m", "slope_deg", "height_above_low_in",
        "Borough", "Latitude", "Longitude"]
df[cols].to_csv(f"{PROC}/nyc_hourly_full.csv", index=False)
df.loc[df["in_storm"], cols].to_csv(f"{PROC}/nyc_storm_windows.csv", index=False)
st.to_csv(f"{PROC}/nyc_storms_summary.csv", index=False)
sy.to_csv(f"{PROC}/nyc_sensor_year_check.csv", index=False)

# --- report ---
obs = df["depth_in"].notna()
sw = df[df["in_storm"]]
print("Full table rows:", len(df), "| observed hours:", int(obs.sum()))
print("Storm-window rows:", len(sw), f"({len(sw)/obs.sum():.1%} of observed hours)")
print("Storms:", len(st), "| flooded storms:", int(st["flooded"].sum()))
print("Flooded hours kept in windows:", int((sw["depth_in"] > 0).sum()), "of", int((df["depth_in"] > 0).sum()))
print(f"Share of flooded hours inside storm windows: {(sw['depth_in']>0).mean():.1%}")
print("Flooded storms with < 0.5 mm rain in our data:", int(st["flood_without_rain"].sum()),
      "of", int(st["flooded"].sum()))
print("\nSuspected offline sensor-years:")
print(sy[sy["suspect_offline"]].to_string(index=False))

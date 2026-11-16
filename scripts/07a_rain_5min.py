"""Stage 7: clean the 1-minute airport rain gauges and make 5-minute rainfall.
Usage: python 07a_rain_5min.py <raw_dir> <out_dir>
Inputs : raw/asos_1min/*.csv.gz (NYC = Central Park, JFK, LGA; precip in inches per minute, 'M' = missing)
         raw/nyc_rain_hourly_openmeteo.csv (only to spot broken winter days)
Output : nyc_rain_5min_stations.csv  (time_utc, station, rain_mm, coverage)
         nyc_rain_bad_days.csv       (station-days removed, with reason)"""
import sys, glob
import numpy as np, pandas as pd
RAW, OUT = sys.argv[1:3]

d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RAW}/asos_1min/*.csv.gz"))], ignore_index=True)
d["time"] = pd.to_datetime(d["valid(UTC)"])
d["mm"] = pd.to_numeric(d["precip"], errors="coerce") * 25.4                 # inches -> mm
d = d.drop_duplicates(["station", "time"])
print("1-min rows:", len(d), "| missing values ('M'):", int(d["mm"].isna().sum()))

# --- broken winter days: snow/ice makes heated tipping buckets report impossible totals ---
d["day"] = d["time"].dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.date
day = d.groupby(["station", "day"])["mm"].sum().rename("gauge_mm").reset_index()
om = pd.read_csv(f"{RAW}/nyc_rain_hourly_openmeteo.csv")
om["time"] = pd.to_datetime(om["time"], utc=True).dt.tz_convert("America/New_York")
om = om.groupby("time")["precipitation"].mean()
om = om.groupby(om.index.date).sum().rename("reanalysis_mm")
day = day.join(om, on="day")
cold = pd.to_datetime(day["day"]).dt.month.isin([11, 12, 1, 2, 3, 4])
bad = day[cold & (day["gauge_mm"] > 3 * day["reanalysis_mm"] + 10)].assign(reason="cold-season total far above reanalysis (likely snow/ice)")
bad.to_csv(f"{OUT}/nyc_rain_bad_days.csv", index=False)
print("Station-days removed:", len(bad))
d = d.merge(bad[["station", "day"]].assign(drop=1), on=["station", "day"], how="left")
d.loc[d["drop"] == 1, "mm"] = np.nan

# --- 5-minute totals; coverage = share of the 5 minutes that have a reading ---
d["t5"] = d["time"].dt.ceil("5min")            # label = end of the 5-minute bin, like FloodNet depth below
r5 = d.groupby(["station", "t5"]).agg(rain_mm=("mm", "sum"), n=("mm", "count")).reset_index()
full = pd.date_range("2020-10-01 00:05", "2026-11-01 00:00", freq="5min")
out = []
for s, g in r5.groupby("station"):
    g = g.set_index("t5").reindex(full)
    out.append(pd.DataFrame({"time_utc": full, "station": s, "rain_mm": g["rain_mm"].fillna(0).round(3).values,
                             "coverage": (g["n"].fillna(0) / 5).round(2).values}))
out = pd.concat(out, ignore_index=True)
out.to_csv(f"{OUT}/nyc_rain_5min_stations.csv", index=False)
print("5-min rows:", len(out))
print(out.groupby("station").agg(coverage=("coverage", "mean"), max_5min_mm=("rain_mm", "max")).round(2).to_string())

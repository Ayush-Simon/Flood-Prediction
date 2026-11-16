"""Step 2a: build the table of ordinary streets (one row per road segment) with a midpoint.
Reads data/raw/nyc_street_centerline.csv, writes data/processed/nyc_streets.csv."""
import os, re
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cols = ["the_geom", "PHYSICALID", "STATUS", "RW_TYPE", "Borough Code", "Street Width",
        "Number Travel Lanes", "Number Total Lane", "Posted Speed", "TRAFDIR", "Full Street Name", "L_ZIP"]
s = pd.read_csv(f"{ROOT}/data/raw/nyc_street_centerline.csv", usecols=cols, low_memory=False)
print("All segments:", len(s)); print("STATUS values:", s["STATUS"].value_counts().to_dict())
s = s[(s["RW_TYPE"] == 1) & (s["STATUS"] == 2)].copy()      # ordinary, built streets

num = re.compile(r"-?\d+\.\d+|-?\d+")
def geom(w):
    v = np.array(num.findall(w), dtype=float).reshape(-1, 2)       # lon, lat pairs
    lon, lat = v[:, 0], v[:, 1]
    k = np.cos(np.radians(lat.mean()))
    d = np.hypot(np.diff(lon) * k, np.diff(lat)) * 111320           # metres per piece
    c = np.concatenate([[0], np.cumsum(d)])
    if c[-1] == 0: return lat[0], lon[0], 0.0
    h = c[-1] / 2; i = min(np.searchsorted(c, h, side="right") - 1, len(d) - 1)
    f = (h - c[i]) / d[i] if d[i] > 0 else 0
    return lat[i] + f * (lat[i+1] - lat[i]), lon[i] + f * (lon[i+1] - lon[i]), c[-1]
g = np.array([geom(w) for w in s["the_geom"]])
s["mid_lat"], s["mid_lon"], s["length_m"] = g[:, 0].round(6), g[:, 1].round(6), g[:, 2].round(1)
s = s.drop(columns=["the_geom", "STATUS", "RW_TYPE"]).rename(columns={
    "PHYSICALID": "street_id", "Borough Code": "boro", "Street Width": "width_ft",
    "Number Travel Lanes": "travel_lanes", "Number Total Lane": "total_lanes",
    "Posted Speed": "speed_mph", "Full Street Name": "name", "L_ZIP": "zip"})
s = s.drop_duplicates("street_id")
s = s[s["mid_lat"].between(40.4, 41.0) & s["mid_lon"].between(-74.3, -73.6)]
s.to_csv(f"{ROOT}/data/processed/nyc_streets.csv", index=False)
print("Streets kept:", len(s)); print(s["boro"].value_counts().sort_index().to_dict())
print(s[["length_m", "width_ft"]].describe().round(1).to_string()); print(s.head(3).to_string())

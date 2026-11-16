"""Step 2b: street features + complaint-to-street matching + daily rain table.
Usage: python 05b_street_features.py <raw_dir> <flood_maps_dir> <out_dir>
Needs: geopandas, pyogrio, scipy."""
import sys, glob, warnings
import numpy as np, pandas as pd, geopandas as gpd, pyogrio
from shapely import wkt
from scipy.spatial import cKDTree
warnings.filterwarnings("ignore")
RAW, FM, OUT = sys.argv[1:4]
CRS = 2263                                   # NY State Plane, units = feet

# ---------- streets ----------
cols = ["the_geom", "PHYSICALID", "STATUS", "RW_TYPE", "Borough Code", "Street Width",
        "Number Travel Lanes", "Number Total Lane", "Posted Speed", "TRAFDIR", "Full Street Name"]
s = pd.read_csv(f"{RAW}/nyc_street_centerline.csv", usecols=cols, low_memory=False)
s = s[(s["RW_TYPE"] == 1) & (s["STATUS"] == 2)].drop_duplicates("PHYSICALID")
st = gpd.GeoDataFrame(s.drop(columns=["the_geom", "STATUS", "RW_TYPE"]),
                      geometry=s["the_geom"].map(wkt.loads), crs=4326).to_crs(CRS).reset_index(drop=True)
st = st.rename(columns={"PHYSICALID": "street_id", "Borough Code": "boro", "Street Width": "width_ft",
                        "Number Travel Lanes": "travel_lanes", "Number Total Lane": "total_lanes",
                        "Posted Speed": "speed_mph", "Full Street Name": "name"})
st["length_ft"] = st.length
mid = st.geometry.interpolate(0.5, normalized=True)
st["x"], st["y"] = mid.x, mid.y
ll = gpd.GeoSeries(mid, crs=CRS).to_crs(4326)
st["mid_lat"], st["mid_lon"] = ll.y.round(6), ll.x.round(6)
st = st[st["mid_lat"].between(40.4, 41.0) & st["mid_lon"].between(-74.3, -73.6)].reset_index(drop=True)
st["one_way"] = (st["TRAFDIR"] != "TW").astype(int)
print("Streets:", len(st))

# ---------- city flood maps: share of each street inside each modelled flood zone ----------
def load(key):
    g = [p for p in glob.glob(f"{FM}/**/*.gdb", recursive=True) if key in p][0]
    d = pyogrio.read_dataframe(g, layer=pyogrio.list_layers(g)[0][0]).to_crs(CRS)
    d["geometry"] = d.geometry.force_2d().make_valid()
    return d.explode(index_parts=False).reset_index(drop=True)
maps = {"mod": load("Moderate_Flood_2_13_inches_per_hr_with_Current"), "ext": load("Extreme_Flood")}
fr = [0.1, 0.3, 0.5, 0.7, 0.9]
pts = gpd.GeoDataFrame({"i": np.repeat(st.index.values, len(fr))},
                       geometry=np.concatenate([st.geometry.interpolate(f, normalized=True).values for f in fr]).reshape(len(fr), -1).T.ravel(),
                       crs=CRS)
for k, m in maps.items():
    for cat, nm in ((1, "nuis"), (2, "deep")):          # 1 = 4 in to 1 ft, 2 = deeper than 1 ft
        poly = m[m["Flooding_Category"] == cat]
        hit = gpd.sjoin(pts, poly[["geometry"]], predicate="within", how="inner")
        cnt = hit.groupby("i").size() / len(fr)
        st[f"{k}_{nm}"] = cnt.reindex(st.index, fill_value=0).values
        print(f"  {k}_{nm}: {(st[f'{k}_{nm}'] > 0).mean():.1%} of streets touch the zone")

# share of streets within ~500 ft that touch a moderate flood zone (low-lying neighbourhood)
tree = cKDTree(st[["x", "y"]].values)
nb = tree.query_ball_point(st[["x", "y"]].values, 500)
flag = ((st["mod_nuis"] + st["mod_deep"]) > 0).values
st["mod_nbr_share"] = [flag[i].mean() for i in nb]
st["nbr_streets"] = [len(i) for i in nb]                  # street density (how built-up)

# ---------- storm drains ----------
cb = pd.read_csv(f"{RAW}/nyc_catch_basins.csv", usecols=["LATITUDE", "LONGITUDE"]).dropna()
cb = gpd.GeoDataFrame(geometry=gpd.points_from_xy(cb["LONGITUDE"], cb["LATITUDE"]), crs=4326).to_crs(CRS)
j = gpd.sjoin_nearest(cb, st[["geometry"]], max_distance=100, how="inner")
j = j[~j.index.duplicated()]
st["drains"] = j.groupby("index_right").size().reindex(st.index, fill_value=0).values
st["drains_per_100ft"] = st["drains"] / (st["length_ft"] / 100).clip(lower=0.5)
ct = cKDTree(np.column_stack([cb.geometry.x, cb.geometry.y]))
st["drain_dist_ft"] = ct.query(st[["x", "y"]].values)[0].clip(max=2000)
st["drains_500ft"] = [len(i) for i in ct.query_ball_point(st[["x", "y"]].values, 500)]
print("Drains matched to a street:", len(j), "of", len(cb))

# ---------- complaints -> nearest street ----------
c = pd.read_csv(f"{RAW}/nyc_311_flooding.csv", parse_dates=["created_date"]).dropna(subset=["latitude", "longitude"])
c["kind"] = np.where(c["descriptor"].str.startswith("Street"), "street_flood", "drain_clogged")
cg = gpd.GeoDataFrame(c[["unique_key", "created_date", "kind"]],
                      geometry=gpd.points_from_xy(c["longitude"], c["latitude"]), crs=4326).to_crs(CRS)
m = gpd.sjoin_nearest(cg, st[["street_id", "geometry"]], max_distance=200, how="inner", distance_col="dist_ft")
m = m[~m.index.duplicated()]
print(f"Complaints matched to a street: {len(m)} of {len(c)} (median distance {m['dist_ft'].median():.0f} ft)")
m["date"] = m["created_date"].dt.date
m[["unique_key", "date", "kind", "street_id", "dist_ft"]].to_csv(f"{OUT}/nyc_complaints_matched.csv", index=False)

# ---------- rain: each street uses its nearest rainfall point; daily features in NY local time ----------
se = pd.read_csv(f"{RAW}/floodnet_sensors.csv").drop_duplicates("Sensor ID")
r = pd.read_csv(f"{RAW}/nyc_rain_hourly_openmeteo.csv")
se = se[se["Sensor ID"].isin(r["Sensor ID"].unique())]
sg = gpd.GeoSeries(gpd.points_from_xy(se["Longitude"], se["Latitude"]), crs=4326).to_crs(CRS)
piv = r.pivot(index="time", columns="Sensor ID", values="precipitation")
piv.index = pd.to_datetime(piv.index, utc=True).tz_convert("America/New_York")
sig = piv.round(3).T.apply(lambda row: hash(tuple(row)), axis=1)          # identical series = same grid cell
cell_of = {sid: f"cell_{k+1}" for k, h in enumerate(pd.unique(sig)) for sid in sig.index[sig == h]}
st["rain_cell"] = [cell_of[se["Sensor ID"].values[i]] for i in cKDTree(np.column_stack([sg.x, sg.y])).query(st[["x", "y"]].values)[1]]
first = {}
for sid, cl in cell_of.items(): first.setdefault(cl, sid)
days = []
for cl, sid in first.items():
    h = piv[sid]
    g = pd.DataFrame({"rain_total": h, "rain_max1h": h, "rain_max3h": h.rolling(3, min_periods=1).sum(),
                      "rain_max6h": h.rolling(6, min_periods=1).sum(), "wet_hours": (h >= 0.5).astype(int)})
    d = g.groupby(g.index.date).agg({"rain_total": "sum", "rain_max1h": "max", "rain_max3h": "max",
                                     "rain_max6h": "max", "wet_hours": "sum"})
    d["rain_prev1"] = d["rain_total"].shift(1).fillna(0)
    d["rain_prev3"] = d["rain_total"].shift(1).rolling(3, min_periods=1).sum().fillna(0)
    d["rain_next1"] = d["rain_total"].shift(-1).fillna(0)       # only used to define storms, never as a feature
    d["rain_cell"] = cl; d.index.name = "date"; days.append(d.reset_index())
days = pd.concat(days, ignore_index=True)
days.to_csv(f"{OUT}/nyc_rain_daily_by_cell.csv", index=False)
print("Rain cells:", st["rain_cell"].value_counts().to_dict())

st.drop(columns=["geometry", "TRAFDIR"]).to_csv(f"{OUT}/nyc_street_features.csv", index=False)
print(st[["length_ft", "width_ft", "drains", "drain_dist_ft", "drains_500ft", "mod_nbr_share"]].describe().round(2).to_string())

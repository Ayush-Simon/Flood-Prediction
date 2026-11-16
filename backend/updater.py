"""Live updater: fetches live weather and rain, runs the flood models and saves the result in the data store.

Runs inside backend/app.py every 15 minutes. What it does:
  - Weather now and the next 24 h: Open-Meteo forecast API (free, no key)
  - Rain measured in the last hours: airport gauges (Central Park, LaGuardia, JFK) via Iowa Environmental Mesonet
  - Road model (XGBoost): flood risk of every street for the next 24 h of forecast rain
  - Depth model (ANN rain forecaster + CNN-LSTM): water depth 5-40 min ahead at each FloodNet sensor
    Live sensor depths are not connected, so the model starts from "street dry now" unless
    backend/store/sensor_depth_now.json gives current depths ({"Sensor ID": inches, ...}).
Needs: pandas, numpy, xgboost, requests."""
import io, json, os, time
from datetime import datetime, timezone
import numpy as np, pandas as pd, requests, xgboost as xgb
from flood_models import DepthModel
import store

HERE = store.HERE; ROOT = os.path.dirname(HERE); P = f"{ROOT}/data/processed"
NY = "America/New_York"
LON0, LAT0, Q = -74.27, 40.48, 1e5                       # same map coordinates as roads.json
RAINF = ["rain_total", "rain_max1h", "rain_max3h", "rain_max6h", "wet_hours", "rain_prev1", "rain_prev3"]
LEVELS = [0.010, 0.003, 0.001]
STORM_MM = 5.0                                          # the road model was trained on days with >= 5 mm
WHAT_IF = dict(rain_total=21.8, rain_max1h=5.5, rain_max3h=11.0, rain_max6h=15.6, wet_hours=9, rain_prev1=1.2, rain_prev3=7.1)
BORO = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"}

info = json.load(open(f"{ROOT}/models/road_risk_model_info.json")); FEATS = info["features"]
road = xgb.Booster(); road.load_model(f"{ROOT}/models/road_risk_xgb.json")
BEST = (0, int(road.attributes().get("best_iteration", road.num_boosted_rounds() - 1)) + 1)
depth_model = DepthModel(json.load(open(f"{ROOT}/models/live_cnn_lstm_ann.json")))
st = pd.read_csv(f"{P}/nyc_street_live_features.csv")
cells = pd.read_csv(f"{P}/nyc_rain_cells.csv")
share = st["rain_cell"].value_counts(normalize=True)
sensors = pd.read_csv(f"{ROOT}/data/raw/floodnet_sensors.csv").drop_duplicates("Sensor ID").set_index("Sensor ID")
sensor_cell = pd.read_csv(f"{P}/sensor_rain_cell.csv").set_index("Sensor ID")["cell"]
WMO = {0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Freezing fog", 51: "Light drizzle", 53: "Drizzle",
       55: "Heavy drizzle", 56: "Freezing drizzle", 57: "Freezing drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain",
       67: "Freezing rain", 71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains", 80: "Light showers", 81: "Showers",
       82: "Violent showers", 85: "Snow showers", 86: "Heavy snow showers", 95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with hail"}


SCEN_MM = [5, 10, 15, 20, 30, 40, 60, 80, 100]          # 24-hour city rain totals for the "what rain does" curve
SCALED = ["rain_total", "rain_max1h", "rain_max3h", "rain_max6h"]


def road_predict(now, cell_rain):
    """street flood chance from the road model; cell_rain = {rain cell: {feature: value}} (None: a typical storm)"""
    X = pd.DataFrame(index=st.index)
    for f in FEATS:
        if f in RAINF:
            X[f] = st["rain_cell"].map({c: v[f] for c, v in cell_rain.items()}) if cell_rain else WHAT_IF[f]
        elif f == "month_sin": X[f] = np.sin(2 * np.pi * now.month / 12)
        elif f == "month_cos": X[f] = np.cos(2 * np.pi * now.month / 12)
        else: X[f] = st[f].fillna(-1)
    return road.predict(xgb.DMatrix(X[FEATS].values.astype(np.float32)), iteration_range=BEST)


def rain_scenarios(now, rain_feats, city_total):
    """run the road model again for 5 to 100 mm of rain in the next 24 hours.
    Shape of the storm: today's forecast if it has real rain, otherwise a typical NYC storm.
    The last 3 days of rain (how wet the ground already is) stay as they really are."""
    out = []
    for mm in SCEN_MM:
        cells = {}
        for c, v in rain_feats.items():
            if city_total >= 1:
                k = mm / city_total; base = v
            else:
                k = mm / WHAT_IF["rain_total"]; base = WHAT_IF
            cells[c] = {f: (base[f] * k if f in SCALED else min(24, round(base["wet_hours"] * k ** 0.5)) if f == "wet_hours" else v[f]) for f in RAINF}
            cells[c]["rain_max1h"] = min(cells[c]["rain_max1h"], cells[c]["rain_total"])
        p = road_predict(now, cells); lvl = np.select([p >= t for t in LEVELS], [3, 2, 1], 0)
        out.append({"mm": mm, "counts": [int((lvl == k).sum()) for k in (3, 2, 1)], "reports": round(float(p.sum()), 1)})
    return out


def open_meteo():
    base = {"latitude": ",".join(cells["lat"].astype(str)), "longitude": ",".join(cells["lon"].astype(str)),
            "hourly": "precipitation,precipitation_probability,temperature_2m", "past_days": 3, "forecast_days": 2, "timezone": NY,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,cloud_cover,pressure_msl,wind_speed_10m,wind_gusts_10m,wind_direction_10m"}
    try:                                                 # 15-minute rain, if the API accepts it
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=60, params={**base, "minutely_15": "precipitation", "past_minutely_15": 8, "forecast_minutely_15": 16})
        r.raise_for_status()
    except requests.RequestException:
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=60, params=base); r.raise_for_status()
    d = r.json(); return d if isinstance(d, list) else [d]


def gauge_rain():
    """last 6 hours of hourly rain at the three airport gauges (METAR reports; inches -> mm)"""
    try:
        r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", timeout=60, params={
            "station": ["NYC", "LGA", "JFK"], "data": "p01i", "hours": 7, "format": "onlycomma", "tz": "Etc/UTC", "missing": "empty", "latlon": "no"})
        r.raise_for_status()
        g = pd.read_csv(io.StringIO(r.text))
        g["valid"] = pd.to_datetime(g["valid"], utc=True).dt.tz_convert(NY); g["p01i"] = pd.to_numeric(g["p01i"], errors="coerce")
        g["hour"] = (g["valid"] + pd.Timedelta(minutes=9)).dt.floor("h")        # METAR hour ends at :51 - :59
        h = g.groupby(["station", "hour"])["p01i"].max().mul(25.4).round(1)
        return {s: [[t.strftime("%H:%M"), None if pd.isna(v) else float(v)] for (_, t), v in x.items()] for s, x in h.groupby(level=0)}
    except Exception as e:
        print("Gauge rain unavailable:", e); return {}


def build():
    t0 = time.time()
    data = open_meteo(); now = pd.Timestamp.now(tz=NY).tz_localize(None)
    rain_feats, nxt_city, past_city, prob_city, temp_city, q15, lasth = {}, 0, 0, 0, 0, {}, {}
    for c, d in zip(cells["rain_cell"], data):
        h = pd.DataFrame(d["hourly"]); h["time"] = pd.to_datetime(h["time"]); h = h.set_index("time")
        nxt = h[(h.index > now.floor("h")) & (h.index <= now.floor("h") + pd.Timedelta(hours=24))]
        past = h[(h.index > now.floor("h") - pd.Timedelta(hours=72)) & (h.index <= now.floor("h"))]["precipitation"].fillna(0)
        n = nxt["precipitation"].fillna(0); lasth[c] = float(past.iloc[-1]) if len(past) else 0.0
        rain_feats[c] = dict(rain_total=n.sum(), rain_max1h=n.max(), rain_max3h=n.rolling(3, min_periods=1).sum().max(),
                             rain_max6h=n.rolling(6, min_periods=1).sum().max(), wet_hours=int((n >= 0.5).sum()),
                             rain_prev1=past.tail(24).sum(), rain_prev3=past.sum())
        w = share.get(c, 0); nxt_city = nxt_city + nxt["precipitation"].fillna(0) * w
        prob_city = prob_city + nxt["precipitation_probability"].fillna(0) * w; temp_city = temp_city + nxt["temperature_2m"] * w
        past_city = past_city + h[(h.index > now.floor("h") - pd.Timedelta(hours=12)) & (h.index <= now.floor("h"))]["precipitation"].fillna(0) * w
        if "minutely_15" in d:
            m = pd.Series(d["minutely_15"]["precipitation"], index=pd.to_datetime(d["minutely_15"]["time"]), dtype=float).fillna(0)
            q15[c] = m
    cur = data[0].get("current", {})

    # ---------- road model ----------
    city_total = sum(rain_feats[c]["rain_total"] * share.get(c, 0) for c in rain_feats)
    what_if = city_total < STORM_MM
    p = road_predict(now, None if what_if else {c: {f: v[f] for f in RAINF} for c, v in rain_feats.items()})
    if what_if:                                         # model only valid for storm days: below 5 mm nothing is flagged
        p = np.zeros_like(p)
    scenarios = rain_scenarios(now, rain_feats, city_total)
    lvl = np.select([p >= t for t in LEVELS], [3, 2, 1], 0)
    flag = np.where(lvl > 0)[0]; flag = flag[np.argsort(-p[flag])]

    # ---------- depth model at each sensor ----------
    depth_now = {}
    fp = f"{store.DIR}/sensor_depth_now.json"
    if os.path.exists(fp): depth_now = json.load(open(fp))
    sens, seqs = [], []
    t_utc = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("5min")
    for sid, cl in sensor_cell.items():
        if sid not in sensors.index: continue
        m = q15.get(cl)
        if m is not None:                                 # last hour of rain, 15-min values split into 5-min steps
            hist = m[(m.index > now - pd.Timedelta(hours=1)) & (m.index <= now)].values
            hist = np.repeat(np.pad(hist, (max(0, 4 - len(hist)), 0))[-4:] / 3, 3)[-12:]
        else:
            hist = np.full(12, lasth[cl] / 12)                # no 15-min data: spread the last hour evenly
        d0 = float(depth_now.get(sid, 0.0))
        seqs.append(np.column_stack([np.full(12, d0), hist, np.zeros(12)]))
        sens.append({"id": sid, "name": str(sensors.loc[sid, "Sensor Name"]), "x": int(round((sensors.loc[sid, "Longitude"] - LON0) * Q)),
                     "y": int(round((sensors.loc[sid, "Latitude"] - LAT0) * Q)), "now": d0 if sid in depth_now else None, "observed": True})
    fc, rfc = depth_model.predict(np.asarray(seqs), np.full(len(seqs), t_utc.to_datetime64()), return_rain=True)
    for s, f, r in zip(sens, fc, rfc): s["fc"] = [round(float(v), 2) for v in f]; s["rain_fc"] = [round(float(v), 2) for v in r]

    # ---------- 15-min rain, city average ----------
    rain15 = []
    if q15:
        m = sum(q15[c] * share.get(c, 0) for c in q15)
        m = m[(m.index > now - pd.Timedelta(hours=2)) & (m.index <= now + pd.Timedelta(hours=4))]
        rain15 = [[t.strftime("%H:%M"), round(float(v), 2), bool(t <= now)] for t, v in m.items()]

    out = {
        "mode": "live", "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "now_local": now.strftime("%A %d %B, %H:%M"),
        "weather": {"desc": WMO.get(cur.get("weather_code"), "—"), "temp": cur.get("temperature_2m"), "feels": cur.get("apparent_temperature"),
                    "humidity": cur.get("relative_humidity_2m"), "rain_now": cur.get("precipitation"), "cloud": cur.get("cloud_cover"),
                    "pressure": cur.get("pressure_msl"), "wind": cur.get("wind_speed_10m"), "gust": cur.get("wind_gusts_10m"), "wind_dir": cur.get("wind_direction_10m")},
        "next24": {"time": [t.strftime("%H:%M") for t in nxt_city.index], "rain": [round(float(v), 2) for v in nxt_city.values],
                   "prob": [round(float(v)) for v in prob_city.values], "temp": [round(float(v), 1) for v in temp_city.values]},
        "past12_mm": round(float(past_city.sum()), 1), "rain15": rain15, "gauges": gauge_rain(),
        "city_rain_mm": round(float(city_total), 1), "peak_1h_mm": round(float(max(v["rain_max1h"] for v in rain_feats.values())), 1),
        "what_if": bool(what_if), "expected_reports": round(float(p.sum()), 1),
        "counts": [int((lvl == k).sum()) for k in (3, 2, 1)], "risk_ix": flag.tolist(), "risk_p": [int(round(v * 1e4)) for v in p[flag]],
        "boro": {BORO[b]: [int(((st["boro"] == b) & (lvl == k)).sum()) for k in (3, 2, 1)] for b in BORO},
        "has_reports": False, "reported": [], "sensors": sens, "depth_connected": bool(depth_now),
        "scenarios": scenarios, "storm_shape": "today's forecast" if city_total >= 1 else "a typical NYC storm",
        "model_run": {"road_streets": int(len(p)), "depth_sensors": len(sens), "seconds": round(time.time() - t0, 1)},
    }
    con = store.connect_rw()
    try:
        with con:
            store.add_live(con, out)
    finally:
        con.close()
    top = max((max(s["fc"]) for s in sens), default=0.0)
    print(f"[{now:%H:%M}] {out['weather']['desc']}, {out['weather']['temp']} °C | next 24 h rain {out['city_rain_mm']} mm | "
          f"streets flagged {out['counts']}{' (what-if map)' if what_if else ''} | deepest sensor forecast {top:.1f} in")
    return out

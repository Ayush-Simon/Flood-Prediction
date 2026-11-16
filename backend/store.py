"""The data store: one SQLite database at backend/store/flood.db.

What it holds
  datasets      every dataset the website shows (street map, storm replays, model results, the latest live
                update), saved as compressed JSON with a SHA-256 checksum
  live_history  a summary and full copy of every live update, kept for 30 days
  audit         when data was imported, updated or refused

How it is protected
  - the store folder is private to your user account (folder 700, files 600)
  - the web server opens the database read-only; only the updater and the import script can write
  - every dataset is checked against its checksum when it is read, so a changed or damaged row is refused
  - the website never reads files directly: it can only ask the server for named datasets
"""
import gzip, hashlib, json, os, sqlite3, threading
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "store")
DB = os.path.join(DIR, "flood.db")
KEEP_DAYS = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
  name TEXT PRIMARY KEY, body BLOB NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, updated TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS live_history (
  ts TEXT PRIMARY KEY, city_rain_mm REAL, peak_1h_mm REAL, streets_flagged INTEGER, deepest_sensor_in REAL,
  body BLOB NOT NULL, sha256 TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit (ts TEXT NOT NULL, event TEXT NOT NULL, detail TEXT);
"""


class IntegrityError(Exception):
    pass


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def lock_down():
    """make the store folder and its files private to the current user"""
    os.makedirs(DIR, exist_ok=True)
    os.chmod(DIR, 0o700)
    for f in os.listdir(DIR):
        p = os.path.join(DIR, f)
        if os.path.isfile(p):
            os.chmod(p, 0o600)


def connect_rw():
    old = os.umask(0o077)                       # new files are created private
    try:
        os.makedirs(DIR, mode=0o700, exist_ok=True)
        con = sqlite3.connect(DB, timeout=15)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.executescript(SCHEMA)
    finally:
        os.umask(old)
    lock_down()
    return con


def connect_ro():
    if not os.path.exists(DB):
        raise FileNotFoundError("The data store is empty. Run: python3 backend/build_store.py")
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=15, check_same_thread=False)


def _pack(obj):
    raw = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode()
    return gzip.compress(raw, 6), hashlib.sha256(raw).hexdigest(), len(raw)


def audit(con, event, detail=""):
    con.execute("INSERT INTO audit VALUES (?,?,?)", (_now(), event, detail[:500]))


def put(con, name, obj):
    body, sha, n = _pack(obj)
    con.execute("INSERT INTO datasets VALUES (?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
                "body=excluded.body, sha256=excluded.sha256, bytes=excluded.bytes, updated=excluded.updated",
                (name, body, sha, n, _now()))


def add_live(con, live):
    """save a live update as the current one and add it to the history"""
    put(con, "live", live)
    body, sha, _ = _pack(live)
    deepest = max((max(s["fc"]) for s in live.get("sensors", []) if s.get("fc")), default=0.0)
    con.execute("INSERT OR REPLACE INTO live_history VALUES (?,?,?,?,?,?,?)",
                (_now(), live.get("city_rain_mm"), live.get("peak_1h_mm"), (live.get("counts") or [0])[0], round(deepest, 2), body, sha))
    cut = (datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    con.execute("DELETE FROM live_history WHERE ts < ?", (cut,))
    audit(con, "live_update", f"rain {live.get('city_rain_mm')} mm, streets {live.get('counts')}")


# ------------------------------------------------------------------ reading (used by the web server)
_verified = {}                                   # (name, sha) -> True, so big datasets are checked once
_vlock = threading.Lock()


def get_gz(con, name):
    """compressed JSON bytes of a dataset, after checking its checksum; None if it does not exist"""
    row = con.execute("SELECT body, sha256 FROM datasets WHERE name=?", (name,)).fetchone()
    if row is None:
        return None
    body, sha = row
    key = (name, sha, hashlib.sha256(body).hexdigest())
    with _vlock:
        ok = _verified.get(key)
    if not ok:
        if hashlib.sha256(gzip.decompress(body)).hexdigest() != sha:
            raise IntegrityError(f"dataset '{name}' failed its checksum")
        with _vlock:
            _verified[key] = True
    return body


def history(con, days=7):
    cut = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = con.execute("SELECT ts, city_rain_mm, peak_1h_mm, streets_flagged, deepest_sensor_in FROM live_history "
                       "WHERE ts >= ? ORDER BY ts", (cut,)).fetchall()
    return [dict(zip(["time", "rain_24h_mm", "peak_1h_mm", "streets_flagged", "deepest_sensor_in"], r)) for r in rows]


def verify_all(con):
    """check every dataset; returns a list of the names that failed"""
    bad = []
    for name, body, sha in con.execute("SELECT name, body, sha256 FROM datasets"):
        if hashlib.sha256(gzip.decompress(body)).hexdigest() != sha:
            bad.append(name)
    return bad

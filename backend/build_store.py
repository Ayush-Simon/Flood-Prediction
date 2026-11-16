"""Build (or rebuild) the data store from the files in backend/data/.

  python3 backend/build_store.py            import everything
  python3 backend/build_store.py --verify   only check every dataset against its checksum

backend/data/ holds the published datasets made by scripts/08_build_dashboard_data.py:
  index.json      storm list and model results
  roads.json      the street network used by the map and the router
  storm_*.json    one replay per past storm
  model_*.json    trained ANN-LSTM and CNN-LSTM for the Model Lab (scripts/13a)
  lab_samples.json real test storms to try in the Model Lab
  weather.json    (optional) daily weather for the past storms
"""
import argparse, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import store

SRC = os.path.join(store.HERE, "data")
NAME = re.compile(r"^(index|roads|weather|lab_samples)\.json$|^storm_(\d{4}-\d{2}-\d{2})\.json$|^model_([a-z_]+)\.json$")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    con = store.connect_rw()
    if a.verify:
        bad = store.verify_all(con)
        n = con.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        print(f"{n} datasets checked, {len(bad)} failed" + (": " + ", ".join(bad) if bad else ""))
        sys.exit(1 if bad else 0)
    n = 0
    with con:
        for f in sorted(os.listdir(SRC)):
            m = NAME.match(f)
            if not m:
                continue
            name = m.group(1) or (f"storm:{m.group(2)}" if m.group(2) else f"model:{m.group(3)}")
            with open(os.path.join(SRC, f), encoding="utf-8") as fh:
                store.put(con, name, json.load(fh))
            n += 1
        store.audit(con, "import", f"{n} datasets from backend/data")
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    store.lock_down()
    size = os.path.getsize(store.DB) / 1e6
    print(f"Imported {n} datasets into {os.path.relpath(store.DB)} ({size:.1f} MB). All checksums stored.")


if __name__ == "__main__":
    main()

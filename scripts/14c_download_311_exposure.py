"""Stage 14c: download how much people complain to 311 in each part of the city (any topic except sewers/flooding).

Flooding complaints are partly about water and partly about how likely people are to call 311 at all. To separate
the two, we need each area's general habit of complaining ("exposure"). Downloading all 311 requests since 2020
(about 18 million) is not needed: a fair sample is enough. This takes every request made on a Tuesday between
12:00 and 12:59, which spreads evenly over years, seasons and boroughs.

Run on a computer with internet, from the project folder:
    python3 scripts/14c_download_311_exposure.py
Writes data/raw/nyc_311_exposure_sample.csv (about 150,000 rows, a few MB): created_date, complaint_type, x, y
(x and y are NY State Plane feet, the same as the street table).
Source: NYC Open Data, "311 Service Requests from 2010 to Present" (erm2-nwe9)."""
import csv, json, os, sys, time, urllib.parse, urllib.request

URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
WHERE = ("created_date between '2020-10-01T00:00:00' and '2026-07-28T23:59:59' "
         "AND date_extract_dow(created_date) = 2 AND date_extract_hh(created_date) = 12 "
         "AND complaint_type != 'Sewer' AND x_coordinate_state_plane IS NOT NULL")
PAGE = 50000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "nyc_311_exposure_sample.csv")


def get(offset):
    q = urllib.parse.urlencode({"$select": "created_date,complaint_type,x_coordinate_state_plane,y_coordinate_state_plane",
                                "$where": WHERE, "$order": "created_date,unique_key", "$limit": PAGE, "$offset": offset})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(URL + "?" + q, headers={"Accept": "application/json"}), timeout=120) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print(f"  page at {offset} failed ({e}); retrying in {5 * (attempt + 1)} s"); time.sleep(5 * (attempt + 1))
    raise SystemExit("NYC Open Data did not answer. Check your internet connection and try again later.")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n, offset = 0, 0
    tmp = OUT + ".part"                                                  # only renamed to the real name once complete
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["created_date", "complaint_type", "x", "y"])
        while True:
            rows = get(offset)
            for r in rows:
                try:
                    w.writerow([r["created_date"][:19], r.get("complaint_type", ""),
                                round(float(r["x_coordinate_state_plane"])), round(float(r["y_coordinate_state_plane"]))])
                    n += 1
                except (KeyError, ValueError):
                    pass
            print(f"  {n:,} requests so far", flush=True)
            if len(rows) < PAGE:
                break
            offset += PAGE
    if n < 10000:
        os.remove(tmp)
        sys.exit(f"Only {n} rows came back; something is wrong with the query. Send the output to Claude.")
    os.replace(tmp, OUT)
    print(f"Saved {n:,} requests to {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()

"""NYC Flood Watch: start the website and the live updates.

  python3 backend/app.py              serve the site at http://localhost:8000 and update live data every 15 minutes
  python3 backend/app.py --once       run one live update, save it in the store, and exit
  python3 backend/app.py --no-live    serve the site only (past storms and the last saved live update)

First time only:  python3 backend/build_store.py   (loads backend/data/ into the secure store)
Run a live update on demand (the token is in backend/store/.admin_token):
  curl -X POST -H "Authorization: Bearer $(cat backend/store/.admin_token)" http://localhost:8000/api/refresh
"""
import argparse, os, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server, store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="one live update, then exit")
    ap.add_argument("--no-live", action="store_true", help="serve without fetching live weather")
    ap.add_argument("--host", default="127.0.0.1", help="address to listen on (default: this computer only)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--every", type=int, default=15, help="minutes between live updates")
    a = ap.parse_args()

    if not os.path.exists(store.DB):
        print("The data store is empty, building it from backend/data ...")
        import build_store; sys.argv = [sys.argv[0]]; build_store.main()

    update = None
    if not a.no_live:
        import updater                               # loads the models (needs numpy, pandas, xgboost, requests)
        lock = threading.Lock()

        def update():
            if not lock.acquire(blocking=False):
                return False                         # an update is already running
            try:
                updater.build(); return True
            except Exception as e:
                print("Live update failed, keeping the last one:", e); return False
            finally:
                lock.release()

        update()
        if a.once:
            return

    if a.host not in ("127.0.0.1", "localhost"):
        print(f"Warning: listening on {a.host}, so other devices on the network can open the site.")
    server.serve(a.host, a.port, refresh=update)
    print(f"Open http://localhost:{a.port}   (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(a.every * 60)
            if update:
                update()
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()

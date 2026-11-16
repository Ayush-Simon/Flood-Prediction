"""The web server: serves the website from frontend/ and the data from the store through a small API.

API (read-only, same origin only)
  GET  /api/index               storm list and model results
  GET  /api/roads               street network
  GET  /api/storm/YYYY-MM-DD    one past storm
  GET  /api/model/ann_lstm      trained model weights for the Model Lab (also cnn_lstm)
  GET  /api/lab_samples         real test storms to try in the Model Lab
  GET  /api/live                the latest live update
  GET  /api/weather             daily weather for past storms (if imported)
  GET  /api/history?days=7      summaries of past live updates (up to 30 days)
  GET  /api/health              is the store reachable and when was the last live update
  POST /api/refresh             run a live update now; needs  Authorization: Bearer <admin token>

Security
  - listens on 127.0.0.1 (this computer only) unless you choose another address
  - the database is opened read-only; the website can only ask for named datasets
  - static files: only from frontend/, only known file types, no hidden files, no '..' tricks
  - strict browser security headers (Content-Security-Policy and others), no cross-site access
  - each address may make at most 300 requests a minute; slow or oversized requests are dropped
  - errors never show file paths or stack traces
"""
import gzip, hmac, json, os, re, secrets, threading, time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import store

ROOT = os.path.dirname(store.HERE)
FRONTEND = os.path.realpath(os.path.join(ROOT, "frontend"))
TOKEN_FILE = os.path.join(store.DIR, ".admin_token")
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon", ".woff2": "font/woff2", ".txt": "text/plain; charset=utf-8"}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
NAMED = {"index", "roads", "live", "weather", "lab_samples", "source"}
OPTIONAL = {"weather", "source"}           # may be missing; "source" exists only on the hosted site (where its live forecast comes from)
MODEL = re.compile(r"^[a-z_]{1,32}$")
RATE, WINDOW = 300, 60
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https://server.arcgisonline.com; font-src 'self'; "
       "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer", "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "geolocation=(), camera=(), microphone=(), payment=(), usb=()",
}


def admin_token():
    """the admin token: FLOOD_ADMIN_TOKEN if set, else a random one kept in backend/store/.admin_token"""
    t = os.environ.get("FLOOD_ADMIN_TOKEN")
    if t:
        return t
    if not os.path.exists(TOKEN_FILE):
        old = os.umask(0o077)
        try:
            os.makedirs(store.DIR, mode=0o700, exist_ok=True)
            with open(TOKEN_FILE, "w") as f:
                f.write(secrets.token_urlsafe(32))
        finally:
            os.umask(old)
    os.chmod(TOKEN_FILE, 0o600)
    return open(TOKEN_FILE).read().strip()


class Limiter:
    def __init__(self):
        self.hits, self.lock = defaultdict(deque), threading.Lock()

    def allow(self, ip):
        now = time.monotonic()
        with self.lock:
            if len(self.hits) > 5000:                    # forget addresses that have gone quiet, so memory stays small
                for k in [k for k, v in self.hits.items() if not v or now - v[-1] > WINDOW]:
                    del self.hits[k]
            q = self.hits[ip]
            while q and now - q[0] > WINDOW:
                q.popleft()
            if len(q) >= RATE:
                return False
            q.append(now)
            return True


def make_handler(refresh=None):
    limiter, token = Limiter(), admin_token()

    class Handler(BaseHTTPRequestHandler):
        server_version, sys_version = "FloodWatch", ""
        protocol_version = "HTTP/1.1"
        timeout = 20                                # drop connections that send too slowly (slow-request attacks)

        def log_message(self, *a):                  # quiet: no request log on the console
            pass

        # ---------------------------------------------------------- responses
        def send(self, code, body=b"", ctype="application/json; charset=utf-8", gz=False, cache="no-store"):
            if gz and "gzip" not in self.headers.get("Accept-Encoding", ""):
                body, gz = gzip.decompress(body), False
            self.send_response(code)
            for k, v in SECURITY_HEADERS.items():
                self.send_header(k, v)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", cache)
            if gz:
                self.send_header("Content-Encoding", "gzip"); self.send_header("Vary", "Accept-Encoding")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def error(self, code, msg):
            self.send(code, json.dumps({"error": msg}).encode())

        def send_error(self, code, message=None, explain=None):
            """Python's own error page (bad request line, unknown method) is replaced by the same short JSON
            answer as every other error, with the security headers and without the server's details."""
            self.close_connection = True
            try:
                self.error(code, "method not allowed" if code in (405, 501) else "bad request" if code < 500 else "server error")
            except Exception:
                pass

        def guard(self):
            if len(self.path) > 2048 or len(self.headers) > 60:
                self.error(414, "address too long"); return False
            if not limiter.allow(self.client_address[0]):
                self.error(429, "too many requests, slow down"); return False
            return True

        # ---------------------------------------------------------- methods
        def do_GET(self):
            if not self.guard():
                return
            url = urlsplit(self.path)
            try:
                if url.path.startswith("/api/"):
                    self.api(url)
                else:
                    self.static(url.path)
            except store.IntegrityError:
                self.error(500, "stored data failed its integrity check; rebuild the store")
            except FileNotFoundError:
                self.error(503, "the data store is empty; run python3 backend/build_store.py")
            except (BrokenPipeError, ConnectionResetError):
                pass                                        # the browser went away; nothing to answer
            except Exception:                               # anything unexpected: a short answer, never a stack trace
                self.close_connection = True
                self.error(500, "server error")

        do_HEAD = do_GET

        def do_POST(self):
            if not self.guard():
                return
            n = self.headers.get("Content-Length", "0")
            if not n.isdigit() or int(n) > 1024:          # the refresh call has no body; refuse anything large
                self.close_connection = True
                return self.error(413, "request body too large")
            self.rfile.read(int(n))                        # read it so the next request on this connection lines up
            if urlsplit(self.path).path != "/api/refresh":
                return self.error(405, "method not allowed")
            auth = self.headers.get("Authorization", "")
            if not (auth.startswith("Bearer ") and hmac.compare_digest(auth[7:].encode(), token.encode())):
                time.sleep(0.5)                       # slow down token guessing
                return self.error(401, "admin token required")
            if refresh is None:
                return self.error(503, "live updates are not running")
            ok = refresh()
            self.error(200 if ok else 500, "updated" if ok else "update failed, kept the last one")

        def do_PUT(self):
            self.close_connection = True                 # any body is not read, so do not reuse the connection
            self.error(405, "method not allowed")
        do_DELETE = do_PATCH = do_OPTIONS = do_TRACE = do_CONNECT = do_PUT

        # ---------------------------------------------------------- data
        def api(self, url):
            parts = url.path.strip("/").split("/")[1:]
            con = store.connect_ro()
            try:
                if parts == ["health"]:
                    row = con.execute("SELECT updated FROM datasets WHERE name='live'").fetchone()
                    n = con.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
                    return self.send(200, json.dumps({"ok": True, "datasets": n, "last_live_update": row[0] if row else None}).encode())
                if parts == ["history"]:
                    q = parse_qs(url.query).get("days", ["7"])[0]
                    days = min(store.KEEP_DAYS, max(1, int(q))) if q.isdigit() else 7
                    return self.send(200, json.dumps(store.history(con, days)).encode())
                if len(parts) == 1 and parts[0] in NAMED:
                    name = parts[0]
                elif len(parts) == 2 and parts[0] == "storm" and DATE.match(parts[1]):
                    name = "storm:" + parts[1]
                elif len(parts) == 2 and parts[0] == "model" and MODEL.match(parts[1]):
                    name = "model:" + parts[1]
                else:
                    return self.error(404, "no such dataset")
                body = store.get_gz(con, name)
                if body is None and name in OPTIONAL:          # optional extras: "nothing here" is a normal answer
                    return self.send(200, b"null", cache="private, max-age=300")
                if body is None:
                    return self.error(404, "no such dataset")
                self.send(200, body, gz=True, cache="no-store" if name == "live" else "private, max-age=300")
            finally:
                con.close()

        def static(self, path):
            if path in ("", "/"):
                path = "/index.html"
            if "\x00" in path or "\\" in path or any(p.startswith(".") for p in path.split("/") if p):
                return self.error(404, "not found")
            full = os.path.realpath(os.path.join(FRONTEND, path.lstrip("/")))
            ext = os.path.splitext(full)[1].lower()
            if not full.startswith(FRONTEND + os.sep) or ext not in STATIC_TYPES or not os.path.isfile(full):
                return self.error(404, "not found")
            with open(full, "rb") as f:
                body = f.read()
            self.send(200, body, STATIC_TYPES[ext], cache="no-cache" if ext in (".html", ".js", ".css") else "public, max-age=86400")

    return Handler


def serve(host="127.0.0.1", port=8000, refresh=None):
    srv = ThreadingHTTPServer((host, port), make_handler(refresh))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv

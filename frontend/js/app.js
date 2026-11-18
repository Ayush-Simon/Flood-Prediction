"use strict";
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;" }[c]));
// colours for a light page (classic theme); the map keeps its own dark overlays
const C = { rain:"#2a7fd4", flood:"#d0362a", warn:"#e08a00", watch:"#c9a400", route:"#8a6df0", soft:"#6e6e6e", line:"rgba(0,0,0,.09)", ink:"#333", ok:"#2a8f5b" };
const BODY = getComputedStyle(document.documentElement).getPropertyValue("--body");
const NICE = { "ANN-LSTM (LSTM + ANN rain forecast)": "ANN-LSTM", "CNN-LSTM + ANN rain forecast": "CNN-LSTM + ANN rain", "LSTM (standard)": "LSTM", "Persistence": "Persistence (baseline)",
  "LSTM + perfect rain forecast": "LSTM + perfect rain (upper bound)", "CNN-LSTM + perfect rain forecast": "CNN-LSTM + perfect rain (upper bound)", "XGBoost (no future rain)": "XGBoost",
  "XGBoost + perfect rain forecast": "XGBoost + perfect rain (upper bound)", "Att-LSTM + perfect rain forecast": "Attention-LSTM + perfect rain (upper bound)",
  "Att-LSTM + perfect forecast + terrain": "Attention-LSTM + terrain + perfect rain (upper bound)", "Att-LSTM + nowcast + terrain (realistic)": "Attention-LSTM + terrain" };
const nice = m => NICE[m] || m;
const MINE = new Set(["ANN-LSTM (LSTM + ANN rain forecast)", "CNN-LSTM", "CNN-LSTM + ANN rain forecast"]);
const LIVE_MODELS = ["CNN-LSTM", "ANN-LSTM (LSTM + ANN rain forecast)", "CNN-LSTM + ANN rain forecast", "LSTM (standard)", "Persistence"];   // need no future data
const STYLE = { "Persistence": ["#8c8c8c", [4,4]], "LSTM (standard)": ["#2a7fd4", []], "ANN-LSTM (LSTM + ANN rain forecast)": ["#e08a00", []], "CNN-LSTM": ["#d0362a", []],
  "CNN-LSTM + ANN rain forecast": ["#c2185b", []], "LSTM + perfect rain forecast": ["#2a8f5b", [2,3]], "CNN-LSTM + perfect rain forecast": ["#d0362a", [2,3]],
  "XGBoost (no future rain)": ["#b8960a", []], "XGBoost + perfect rain forecast": ["#b8960a", [2,3]], "Att-LSTM + perfect rain forecast": ["#7357c9", [2,3]],
  "Att-LSTM + perfect forecast + terrain": ["#7357c9", [6,3]], "Att-LSTM + nowcast + terrain (realistic)": ["#7357c9", []] };
const DEFAULT_ON = new Set(["LSTM (standard)", "ANN-LSTM (LSTM + ANN rain forecast)", "CNN-LSTM"]);
const GAUGE = { NYC:["Central Park", "#8fd0ff"], LGA:["LaGuardia", "#c9b5ff"], JFK:["JFK", "#74e3b5"] };
const fmtDate = d => new Date(d + "T12:00:00").toLocaleDateString("en-GB", { day:"numeric", month:"long", year:"numeric" });
const hhmm = m => String(Math.floor(m / 60) % 24).padStart(2, "0") + ":" + String(m % 60).padStart(2, "0");
const slotLabel = i => hhmm((i + 1) * 5);            // slot i = the 5 minutes ending at this time

/* ---------------------------------------------------------------- charts */
function chart(cv, o) {
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const L = 40, R = o.y2 ? 36 : 10, T = 10, B = 24, n = o.n;
  const vals = o.series.filter(s => !s.y2).flatMap(s => s.data).filter(v => v != null), max = Math.max(o.minMax || 1, ...vals);
  const X = i => L + (W - L - R) * (n > 1 ? i / (n - 1) : 0), Y = v => T + (H - T - B) * (1 - v / max), Y2 = v => T + (H - T - B) * (1 - v / 100);
  g.font = "11px " + BODY; g.fillStyle = C.soft; g.strokeStyle = C.line; g.lineWidth = 1;
  for (let k = 0; k <= 4; k++) { const v = max * k / 4, y = Y(v); g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke(); g.textAlign = "right"; g.fillText(v.toFixed(max < 5 ? 1 : 0), L - 6, y + 4); if (o.y2) { g.textAlign = "left"; g.fillText(25 * k, W - R + 6, y + 4); } }
  g.textAlign = "center";   // keep labels apart on narrow screens: widen the step until each label has about 46 px
  { const room = Math.max(1, Math.floor((X(n - 1) - X(0)) / 46)), tk = Math.max(o.tick, Math.ceil((n - 1) / room / o.tick) * o.tick);
    for (let i = 0; i < n; i += tk) { const x = X(i), s = o.label(i), w = g.measureText(s).width;
      g.fillText(s, Math.min(Math.max(x, w / 2 + 1), W - w / 2 - 1), H - 6); } }
  if (o.mark != null && o.mark >= 0 && o.mark < n) { g.strokeStyle = "rgba(0,0,0,.4)"; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(X(o.mark), T); g.lineTo(X(o.mark), H - B); g.stroke(); g.setLineDash([]); }
  for (const s of o.series) {
    const yy = s.y2 ? Y2 : Y;
    if (s.bars) { g.fillStyle = s.color; const w = Math.max(1, (W - L - R) / n - 2); s.data.forEach((v, i) => { if (v) g.fillRect(X(i) - w / 2, yy(v), w, yy(0) - yy(v)); }); continue; }
    g.strokeStyle = s.color; g.lineWidth = s.width || 1.8; g.setLineDash(s.dash || []); g.beginPath(); let pen = false;
    s.data.forEach((v, i) => { if (v == null) { pen = false; return; } pen ? g.lineTo(X(i), yy(v)) : g.moveTo(X(i), yy(v)); pen = true; });
    g.stroke(); g.setLineDash([]);
  }
}

/* ---------------------------------------------------------------- road network + routing */
const KX = 0.758, M_PER = 1.113;
let ROADS = null, GRAPH = null;
function buildGraph(r) {
  const n = r.off.length - 1, key = new Map(), nx = [], ny = [];
  const a = new Int32Array(n), b = new Int32Array(n), len = new Float32Array(n), mx = new Float32Array(n), my = new Float32Array(n);
  const node = (x, y) => { const k = x * 200000 + y; let i = key.get(k); if (i === undefined) { i = nx.length; key.set(k, i); nx.push(x); ny.push(y); } return i; };
  for (let s = 0; s < n; s++) {
    const p0 = r.off[s], p1 = r.off[s + 1] - 1; let L = 0;
    for (let p = p0; p < p1; p++) L += Math.hypot((r.pts[2*p+2] - r.pts[2*p]) * KX, r.pts[2*p+3] - r.pts[2*p+1]);
    a[s] = node(r.pts[2*p0], r.pts[2*p0+1]); b[s] = node(r.pts[2*p1], r.pts[2*p1+1]); len[s] = L * M_PER;
    const pm = (p0 + p1) >> 1, pn = Math.min(pm + 1, p1); mx[s] = (r.pts[2*pm] + r.pts[2*pn]) / 2; my[s] = (r.pts[2*pm+1] + r.pts[2*pn+1]) / 2;
  }
  const N = nx.length, deg = new Int32Array(N + 1);
  for (let s = 0; s < n; s++) { deg[a[s] + 1]++; deg[b[s] + 1]++; }
  for (let i = 0; i < N; i++) deg[i + 1] += deg[i];
  const fill = deg.slice(0, N), adjSeg = new Int32Array(2 * n), adjTo = new Int32Array(2 * n);
  for (let s = 0; s < n; s++) { adjSeg[fill[a[s]]] = s; adjTo[fill[a[s]]++] = b[s]; adjSeg[fill[b[s]]] = s; adjTo[fill[b[s]]++] = a[s]; }
  return { n, N, a, b, len, mx, my, nx: Int32Array.from(nx), ny: Int32Array.from(ny), deg, adjSeg, adjTo };
}
function nearestNode(G, x, y) { let best = -1, bd = Infinity; for (let i = 0; i < G.N; i++) { if (G.deg[i+1] === G.deg[i]) continue; const d = ((G.nx[i]-x)*KX)**2 + (G.ny[i]-y)**2; if (d < bd) { bd = d; best = i; } } return best; }
function nearestStreet(G, x, y, limit) { let best = -1, bd = limit * limit; for (let s = 0; s < ROADS.n_streets; s++) { const d = ((G.mx[s]-x)*KX)**2 + (G.my[s]-y)**2; if (d < bd) { bd = d; best = s; } } return best; }
function route(G, from, to, mult) {
  const dist = new Float64Array(G.N).fill(Infinity), prevSeg = new Int32Array(G.N).fill(-1), prevNode = new Int32Array(G.N).fill(-1), done = new Uint8Array(G.N);
  const h = i => Math.hypot((G.nx[i]-G.nx[to])*KX, G.ny[i]-G.ny[to]) * M_PER;
  const heap = [[h(from), from]]; dist[from] = 0;
  const push = e => { heap.push(e); let i = heap.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (heap[p][0] <= heap[i][0]) break; [heap[p], heap[i]] = [heap[i], heap[p]]; i = p; } };
  const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let i = 0; for (;;) { let l = 2*i+1, r = l+1, m = i; if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (r < heap.length && heap[r][0] < heap[m][0]) m = r; if (m === i) break; [heap[m], heap[i]] = [heap[i], heap[m]]; i = m; } } return top; };
  while (heap.length) {
    const [, u] = pop(); if (done[u]) continue; done[u] = 1; if (u === to) break;
    for (let e = G.deg[u]; e < G.deg[u+1]; e++) { const s = G.adjSeg[e], v = G.adjTo[e], m = mult(s); if (m === Infinity || done[v]) continue;
      const nd = dist[u] + G.len[s] * m; if (nd < dist[v]) { dist[v] = nd; prevSeg[v] = s; prevNode[v] = u; push([nd + h(v), v]); } }
  }
  if (dist[to] === Infinity) return null;
  const segs = []; for (let u = to; u !== from; u = prevNode[u]) segs.push(prevSeg[u]);
  return segs.reverse();
}

/* ---------------------------------------------------------------- flood simulation (past storms)
   The road model gives each street a risk level for the whole storm day. To show HOW the storm unfolded,
   each street is linked to its nearest rain gauge and lit up when the gauge's recent rain passes a threshold
   that is lower for riskier streets. Recent rain = a leaky bucket of 5-minute gauge rain (drains ~8% every 5 min). */
const GAUGE_LL = { NYC: [40.7790, -73.9692], LGA: [40.7794, -73.8803], JFK: [40.6386, -73.7622] };
const GKEYS = ["NYC", "LGA", "JFK"], LEAK = 0.92;
const LIGHT_AT = { 3: 5, 2: 9, 1: 14 }, FULL_AFTER = 6;       // mm of recent rain at which a street starts / is fully shown
let SG = null, WB = null;                                      // street -> gauge index; bucket per gauge per 5-min slot
function streetGauges() {
  const gx = GKEYS.map(k => [(GAUGE_LL[k][1] + 74.27) * 1e5, (GAUGE_LL[k][0] - 40.48) * 1e5]);
  SG = new Uint8Array(ROADS.n_streets);
  for (let s = 0; s < ROADS.n_streets; s++) { let b = 0, bd = Infinity; gx.forEach(([x, y], i) => { const d = ((GRAPH.mx[s] - x) * KX) ** 2 + (GRAPH.my[s] - y) ** 2; if (d < bd) { bd = d; b = i; } }); SG[s] = b; }
}
function buckets(st) {
  const raw = GKEYS.map(k => st.gauges[k] || null);
  return raw.map((a, gi) => { const w = new Float32Array(288); let v = 0;
    for (let t = 0; t < 288; t++) { let r = a ? a[t] : null;
      if (r == null) { const o = raw.map(x => x && x[t]).filter(x => x != null); r = o.length ? o.reduce((p, q) => p + q, 0) / o.length : 0; }
      v = v * LEAK + r; w[t] = v; }
    return w; });
}
function bucketAt(g, t) { const a = WB[g], i = Math.min(286, Math.floor(t)), f = t - i; return a[i] * (1 - f) + a[i + 1] * f; }
let LIVE_BASE = 0, NOW_SLOT = 0;                               // live: slot 0 = two hours ago (minutes after midnight), NOW_SLOT = now
function labelFor(slot) { if (MODE !== "live") return slotLabel(slot); const m = LIVE_BASE + 5 * slot; return hhmm(((m % 1440) + 1440) % 1440); }
function liveBuckets(L) {
  const toMin = t => +t.slice(0, 2) * 60 + +t.slice(3, 5), q = L.rain15 || [], h = L.next24 || { time: [], rain: [] };
  if (q.length) { LIVE_BASE = toMin(q[0][0]) - 15; const k = q.findIndex(r => !r[2]); NOW_SLOT = Math.max(0, (k < 0 ? q.length : k) * 3); }
  else { LIVE_BASE = (h.time.length ? toMin(h.time[0]) - 60 : 0) - 120; NOW_SLOT = 24; }
  let h0 = h.time.length ? toMin(h.time[0]) - 60 : null;     // hourly value i covers the hour ending at time[i]
  if (h0 != null && h0 < LIVE_BASE - 60) h0 += 1440;        // forecast starts after midnight
  const city = new Float32Array(288);
  for (let t = 0; t < 288; t++) {
    const m = LIVE_BASE + 5 * t + 5, qi = Math.floor(5 * t / 15);
    if (q.length && qi < q.length) { city[t] = (q[qi][1] || 0) / 3; continue; }
    if (h0 != null) { const d = m - h0, hi = Math.floor((d - 1) / 60); if (hi >= 0 && hi < h.rain.length) city[t] = (h.rain[hi] || 0) / 12; }
  }
  const w = new Float32Array(288); let v = 0; for (let t = 0; t < 288; t++) { v = v * LEAK + city[t]; w[t] = v; }
  return [w, w, w];
}
function shown(level, g, t) { if (!WB || !level) return MODE === "live" ? 1 : 0; return Math.max(0, Math.min(1, (bucketAt(g, t) - LIGHT_AT[level]) / FULL_AFTER)); }
function floodedNow(t) { if (!map.groups) return 0; let n = 0; for (const lv of [3, 2, 1]) for (let g = 0; g < 3; g++) if (shown(lv, g, t) >= 0.5) n += map.groups[lv][g].n; return n; }
function drawTimeline() {
  const cv = $("tl"), d = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight; cv.width = W * d; cv.height = H * d;
  const g = cv.getContext("2d"); g.setTransform(d, 0, 0, d, 0, 0); g.clearRect(0, 0, W, H); if (!WB) return;
  const tot = t => WB.reduce((s, a) => s + a[t], 0) / 3, mx = Math.max(6, ...Array.from({ length: 288 }, (_, t) => tot(t)));
  g.fillStyle = "rgba(143,208,255,.55)"; for (let t = 0; t < 288; t++) { const h = tot(t) / mx * (H - 2); g.fillRect(t / 287 * W - 1, H - h, Math.max(1.2, W / 288), h); }
  g.strokeStyle = "rgba(255,106,92,.8)"; g.setLineDash([3, 3]); const y = H - LIGHT_AT[3] / mx * (H - 2); g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke(); g.setLineDash([]);
  if (MODE === "live") { const x = NOW_SLOT / 287 * W; g.strokeStyle = "rgba(255,255,255,.9)"; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke(); g.fillStyle = "rgba(255,255,255,.9)"; g.font = "10px " + BODY; g.fillText("now", Math.min(x + 3, W - 22), 9); }
}

/* ---------------------------------------------------------------- map */
const map = { s: 1, ox: 0, oy: 0, paths: {}, level: null, prob: null, start: -1, end: -1, routeSegs: null, plainSegs: null, pick: null, showRep: false };
const wx = x => x * KX, wy = y => -y;
let MODE = "live", LIVE = null, PAST = null, STORM = null, INDEX = null, WX = null, T = 0, model = "CNN-LSTM", ahead = "20", sensorId = null, horizon = "20", modelsOn = new Set(DEFAULT_ON);
function segPath(list) { const r = ROADS, p = new Path2D(); for (const s of list) { let q = r.off[s]; p.moveTo(wx(r.pts[2*q]), wy(r.pts[2*q+1])); for (q++; q < r.off[s+1]; q++) p.lineTo(wx(r.pts[2*q]), wy(r.pts[2*q+1])); } return p; }
function initMap() {
  map.cv = $("map"); map.g = map.cv.getContext("2d");
  const all = [...Array(ROADS.off.length - 1).keys()];
  map.paths.streets = segPath(all.filter(s => s < ROADS.n_streets)); map.paths.major = segPath(all.filter(s => s >= ROADS.n_streets));
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < ROADS.pts.length; i += 2) { const x = wx(ROADS.pts[i]), y = wy(ROADS.pts[i+1]); x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  map.bounds = [x0, y0, x1, y1];
  let drag = null, moved = 0;
  map.cv.addEventListener("pointerdown", e => { if (e.pointerType === "touch" || e.button !== 0) return; drag = [e.clientX, e.clientY]; moved = 0; map.cv.setPointerCapture(e.pointerId); map.cv.classList.add("drag"); });
  map.cv.addEventListener("pointermove", e => { if (drag) { map.ox += e.clientX - drag[0]; map.oy += e.clientY - drag[1]; moved += Math.abs(e.clientX - drag[0]) + Math.abs(e.clientY - drag[1]); drag = [e.clientX, e.clientY]; draw(); $("tip").style.display = "none"; } else hover(e); });
  map.cv.addEventListener("pointerup", e => { map.cv.classList.remove("drag"); if (e.pointerType === "touch" || (drag && moved < 10)) click(e); drag = null; });
  map.cv.addEventListener("pointerleave", () => $("tip").style.display = "none");
  // zoom only on purpose: Ctrl/Cmd + scroll (and trackpad pinch, which the browser reports with Ctrl). Plain scrolling scrolls the page.
  let hintT = 0;
  map.cv.addEventListener("wheel", e => {
    if (!(e.ctrlKey || e.metaKey)) { $("hint").classList.add("on"); clearTimeout(hintT); hintT = setTimeout(() => $("hint").classList.remove("on"), 1300); return; }
    e.preventDefault(); const r = map.cv.getBoundingClientRect(); zoom(Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.0025)), e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });
  $("z-in").onclick = () => zoom(1.6); $("z-out").onclick = () => zoom(1 / 1.6); $("z-fit").onclick = () => { fit(); draw(); };
  const mode = m => { map.pick = map.pick === m ? null : m; syncPick(); };
  $("b-start").onclick = () => mode("start"); $("b-end").onclick = () => mode("end");
  $("b-clear").onclick = () => { $("q-from").value = $("q-to").value = ""; map.start = map.end = -1; map.routeSegs = map.plainSegs = null; map.pick = null; syncPick(); routeInfo(); draw(); };
  $("b-rep").onclick = e => { map.showRep = !map.showRep; e.target.setAttribute("aria-pressed", map.showRep); draw(); };
  new ResizeObserver(() => { size(); if (!map.fitted) { fit(); map.fitted = true; } draw(); }).observe(map.cv);
}
function syncPick() { $("b-start").setAttribute("aria-pressed", map.pick === "start"); $("b-end").setAttribute("aria-pressed", map.pick === "end"); map.cv.classList.toggle("pick", !!map.pick); }
function size() { const d = window.devicePixelRatio || 1; map.cv.width = map.cv.clientWidth * d; map.cv.height = map.cv.clientHeight * d; }
function fit() { const [x0, y0, x1, y1] = map.bounds, W = map.cv.clientWidth, H = map.cv.clientHeight - 70; map.s = Math.min(W / (x1 - x0), H / (y1 - y0)) * 0.94; map.ox = (W - (x0 + x1) * map.s) / 2; map.oy = (H - (y0 + y1) * map.s) / 2; }
function zoom(f, cx, cy) { cx ??= map.cv.clientWidth / 2; cy ??= map.cv.clientHeight / 2; const ns = Math.max(0.004, Math.min(1.2, map.s * f)); f = ns / map.s; map.ox = cx - (cx - map.ox) * f; map.oy = cy - (cy - map.oy) * f; map.s = ns; draw(); }
const toMap = (px, py) => [(px - map.ox) / map.s / KX, -(py - map.oy) / map.s];
const toPx = (x, y) => [wx(x) * map.s + map.ox, wy(y) * map.s + map.oy];

function forecastAt(q, t) {
  if (MODE === "live") return q.fc ? { f: q.fc[(+ahead) / 5 - 1], a: null, now: q.now } : null;                      // standing at slot t, the model's depth forecast for t + ahead, and what really happened
  if (!q.pred) return null;
  const k = STORM.models.indexOf(model), tgt = t + (+ahead) / 5;
  if (k < 0 || tgt > 287) return null;
  return { f: q.pred[ahead][k][tgt], a: q.obs[tgt], now: q.obs[t] };
}
function sensorColor(v) { return v == null ? "#2a3a44" : v >= 4 ? C.flood : v > 0.5 ? C.warn : "#1d2c35"; }

/* ---------------------------------------------------------------- satellite photos under the map
   Esri World Imagery tiles (Web Mercator, 256 px). Each tile is placed by its exact corner longitude and latitude
   in the map's own coordinates, so the photos line up with the streets. If the tiles cannot load (no internet),
   the map falls back to the plain street drawing. */
const SAT = { cache: new Map(), ok: false, fails: 0, raf: 0,
  url: (z, x, y) => `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z}/${y}/${x}` };
const tileLon = (x, z) => x / 2 ** z * 360 - 180;
const tileLat = (y, z) => Math.atan(Math.sinh(Math.PI * (1 - 2 * y / 2 ** z))) * 180 / Math.PI;
const lonTile = (lon, z) => Math.floor((lon + 180) / 360 * 2 ** z);
const latTile = (lat, z) => { const r = lat * Math.PI / 180; return Math.floor((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2 * 2 ** z); };
const geoX = lon => (lon - ROADS.lon0) * ROADS.q * KX, geoY = lat => -(lat - ROADS.lat0) * ROADS.q;   // same projection as segPath
function satTile(z, x, y) {
  const k = `${z}/${x}/${y}`; let t = SAT.cache.get(k);
  if (t) { SAT.cache.delete(k); SAT.cache.set(k, t); return t; }          // keep the most recently used last
  t = { img: new Image(), ready: false };
  t.img.referrerPolicy = "no-referrer"; t.img.decoding = "async";
  t.img.onload = () => { t.ready = true; const first = !SAT.ok; SAT.ok = true; if (first) $("attrib").hidden = false; satRedraw(); };
  t.img.onerror = () => { SAT.fails++; SAT.cache.delete(k); };
  t.img.src = SAT.url(z, x, y);
  SAT.cache.set(k, t);
  if (SAT.cache.size > 700) SAT.cache.delete(SAT.cache.keys().next().value);
  return t;
}
function satRedraw() { if (!SAT.raf) SAT.raf = requestAnimationFrame(() => { SAT.raf = 0; map.baseKey = null; draw(); }); }
function drawSat(bg, d) {
  const W = map.cv.clientWidth, H = map.cv.clientHeight, s = map.s;
  const lonAt = px => (px - map.ox) / s / KX / ROADS.q + ROADS.lon0, latAt = py => -(py - map.oy) / s / ROADS.q + ROADS.lat0;
  const w = Math.max(-179, lonAt(0)), e = Math.min(179, lonAt(W)), sLat = Math.max(-80, latAt(H)), nLat = Math.min(80, latAt(0));
  // the tile zoom whose pixels best match the screen's pixels, capped so a view never needs more than about 160 tiles
  let z = Math.round(Math.log2(156543.034 * Math.cos((sLat + nLat) / 2 * Math.PI / 180) * s * d / M_PER) - 0.25);
  z = Math.max(3, Math.min(19, z));
  while (z > 3 && (lonTile(e, z) - lonTile(w, z) + 1) * (latTile(sLat, z) - latTile(nLat, z) + 1) > 160) z--;
  const pad = 0.6 / (s * d);                                                // a hair of overlap hides seams between tiles
  for (let ty = latTile(nLat, z); ty <= latTile(sLat, z); ty++) for (let tx = lonTile(w, z); tx <= lonTile(e, z); tx++) {
    const X0 = geoX(tileLon(tx, z)), X1 = geoX(tileLon(tx + 1, z)), Y0 = geoY(tileLat(ty, z)), Y1 = geoY(tileLat(ty + 1, z));
    const t = satTile(z, tx, ty);
    if (t.ready) { bg.drawImage(t.img, X0 - pad, Y0 - pad, X1 - X0 + 2 * pad, Y1 - Y0 + 2 * pad); continue; }
    for (let up = 1; up <= 5; up++) {                                       // until it arrives, stretch a coarser tile already in memory
      const p = SAT.cache.get(`${z - up}/${tx >> up}/${ty >> up}`);
      if (p && p.ready) { const n = 2 ** up, c = 256 / n;
        bg.drawImage(p.img, (tx % n) * c, (ty % n) * c, c, c, X0 - pad, Y0 - pad, X1 - X0 + 2 * pad, Y1 - Y0 + 2 * pad); break; }
    }
  }
}
function draw() {
  const g = map.g, d = window.devicePixelRatio || 1, s = map.s; if (!g || !ROADS) return;
  const px = w => w / s, z = Math.min(1, Math.max(0, (s - 0.01) / 0.05));
  // the street network only changes when the view moves: draw it once into a cached layer, then reuse it every frame
  const key = [map.cv.width, map.cv.height, s, map.ox, map.oy].join();
  if (map.baseKey !== key) {
    const b = map.base || (map.base = document.createElement("canvas")), bg = b.getContext("2d");
    if (b.width !== map.cv.width || b.height !== map.cv.height) { b.width = map.cv.width; b.height = map.cv.height; }
    bg.setTransform(1, 0, 0, 1, 0, 0); bg.clearRect(0, 0, b.width, b.height); bg.fillStyle = "rgba(8,15,20,.78)"; bg.fillRect(0, 0, b.width, b.height);
    bg.setTransform(d * s, 0, 0, d * s, d * map.ox, d * map.oy); bg.lineCap = "round"; bg.lineJoin = "round";
    drawSat(bg, d);
    if (SAT.ok) {                                                           // satellite photos: a light shade so the flood colours stand out
      bg.setTransform(1, 0, 0, 1, 0, 0); bg.fillStyle = "rgba(6,12,16,.22)"; bg.fillRect(0, 0, b.width, b.height);
      bg.setTransform(d * s, 0, 0, d * s, d * map.ox, d * map.oy);
    } else {                                                                // no internet: the plain street drawing
      bg.strokeStyle = "#24343e"; bg.lineWidth = px(0.6 + 2.6 * z); bg.stroke(map.paths.streets);
      bg.strokeStyle = "#5a6e7a"; bg.lineWidth = px(1.3 + 3.2 * z); bg.stroke(map.paths.major);
      if (z > 0.35) { bg.strokeStyle = "#3a4f5b"; bg.lineWidth = px(0.4 + 1.6 * z); bg.stroke(map.paths.streets); }
    }
    map.baseKey = key;
  }
  g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, map.cv.width, map.cv.height); g.drawImage(map.base, 0, 0);
  g.setTransform(d * s, 0, 0, d * s, d * map.ox, d * map.oy); g.lineCap = "round"; g.lineJoin = "round";
  if (map.groups) for (let gi = 0; gi < 3; gi++) {
    const a1 = shown(1, gi, T), a2 = shown(2, gi, T), a3 = shown(3, gi, T), G = map.groups;
    if (a1 > 0) { g.globalAlpha = 0.7 * a1; g.strokeStyle = C.watch; g.lineWidth = px(1.2 + 2.4 * z); g.stroke(G[1][gi].p); }
    if (a2 > 0) { g.globalAlpha = a2; g.strokeStyle = C.warn; g.lineWidth = px(1.8 + 2.8 * z); g.stroke(G[2][gi].p); }
    if (a3 > 0) { g.globalAlpha = a3; g.strokeStyle = "rgba(255,106,92,.3)"; g.lineWidth = px((6 + 6 * z) * (0.6 + 0.4 * a3)); g.stroke(G[3][gi].p);
      g.strokeStyle = C.flood; g.lineWidth = px(2.6 + 3.4 * z); g.stroke(G[3][gi].p); }
    g.globalAlpha = 1;
  }
  if (map.plainSegs) { g.strokeStyle = "rgba(255,255,255,.55)"; g.lineWidth = px(2.5); g.setLineDash([px(7), px(7)]); g.stroke(segPath(map.plainSegs)); g.setLineDash([]); }
  if (map.routeSegs) { const p = segPath(map.routeSegs); g.strokeStyle = "#0a1216"; g.lineWidth = px(9); g.stroke(p); g.strokeStyle = C.route; g.lineWidth = px(5); g.stroke(p); }
  g.setTransform(d, 0, 0, d, 0, 0);
  const st = STORM;
  if (st && map.showRep && st.has_reports) { g.strokeStyle = C.rain; g.lineWidth = 1.6; for (const i of st.reported) { const [x, y] = toPx(GRAPH.mx[i], GRAPH.my[i]); g.beginPath(); g.arc(x, y, 6, 0, 7); g.stroke(); } }
  if (st) for (const q of st.sensors) { if (!q.observed) continue; const [x, y] = toPx(q.x, q.y), fc = forecastAt(q, T), v = fc ? fc.f : 0, r = 6 + Math.min(8, (v || 0) / 3);
    g.beginPath(); g.moveTo(x, y - r); g.lineTo(x + r, y); g.lineTo(x, y + r); g.lineTo(x - r, y); g.closePath();
    g.fillStyle = sensorColor(v); g.fill(); g.strokeStyle = q.id === sensorId ? C.route : C.ink; g.lineWidth = q.id === sensorId ? 2.5 : 1.3; g.stroke(); }
  const pin = (node, label, col) => { if (node < 0) return; const [x, y] = toPx(GRAPH.nx[node], GRAPH.ny[node]); g.beginPath(); g.arc(x, y, 9, 0, 7); g.fillStyle = col; g.fill(); g.strokeStyle = "#0a1216"; g.lineWidth = 2; g.stroke(); g.fillStyle = "#0a1216"; g.font = "600 11px " + BODY; g.textAlign = "center"; g.fillText(label, x, y + 4); };
  pin(map.start, "A", C.ok); pin(map.end, "B", C.route);
}
function nearSensor(px, py) { if (!STORM) return null; for (const q of STORM.sensors) { const [x, y] = toPx(q.x, q.y); if (q.observed && Math.hypot(x - px, y - py) < 11) return q; } return null; }
let hoverT = 0;
function hover(e) { const now = performance.now(); if (now - hoverT < 50) return; hoverT = now;
  const r = map.cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top, tip = $("tip"), q = nearSensor(px, py);
  let html = "";
  if (q) { const fc = forecastAt(q, T); html = `<b>${esc(q.name)}</b><br>` + (MODE === "live" ? `Forecast in ${ahead} min: ${fmtIn(fc && fc.f)}` : fc ? `${esc(nice(model))} forecast for ${hhmm((T + 1) * 5 + +ahead)}: ${fmtIn(fc.f)}<br>Actual: ${fmtIn(fc.a)}` : `Peak that day: ${q.peak} in`); }
  else if (map.s > 0.02) { const [x, y] = toMap(px, py), s = nearestStreet(GRAPH, x, y, Math.max(40, 14 / map.s)); if (s >= 0) { const lv = map.level[s]; html = `<b>${esc(ROADS.names[ROADS.name[s]])}</b><br>${["No flag", "Low risk", "Medium risk", "High risk"][lv]}${lv ? ", " + (map.prob[s] / 100).toFixed(1) + "% chance of a report" : ""}`; } }
  if (!html) { tip.style.display = "none"; return; }
  tip.innerHTML = html; tip.style.display = "block"; tip.style.left = Math.min(px + 14, r.width - 260) + "px"; tip.style.top = (py + 14) + "px"; }
function click(e) { const r = map.cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  if (!map.pick) { const q = nearSensor(px, py); if (q && q.obs && MODE === "past") selectSensor(q.id); return; }
  const [x, y] = toMap(px, py), node = nearestNode(GRAPH, x, y);
  if (map.pick === "start") { map.start = node; map.pick = map.end < 0 ? "end" : null; } else { map.end = node; map.pick = null; }
  syncPick(); computeRoute(); draw(); }
function computeRoute() {
  map.routeSegs = map.plainSegs = null; if (map.start < 0 || map.end < 0) return routeInfo();
  const on = WB ? [0, 1, 2, 3].map(l => [0, 1, 2].map(gi => shown(l, gi, T) >= 0.5)) : null;
  const lv = s => { if (s >= ROADS.n_streets) return 0; const l = map.level[s]; return !on || (l && on[l][SG[s]]) ? l : 0; }, cost = [1, 1.5, 4, Infinity];
  const plain = route(GRAPH, map.start, map.end, () => 1);
  let safe = route(GRAPH, map.start, map.end, s => cost[lv(s)]), forced = false;
  if (!safe && plain) { safe = route(GRAPH, map.start, map.end, s => [1, 1.5, 4, 60][lv(s)]); forced = true; }
  map.routeSegs = safe; const same = plain && safe && plain.length === safe.length && plain.every((s, i) => s === safe[i]);
  map.plainSegs = same ? null : plain; routeInfo(plain, safe, forced);
}
function routeInfo(plain, safe, forced) {
  const el = $("route-info"), km = segs => (segs.reduce((t, s) => t + GRAPH.len[s], 0) / 1000).toFixed(1), act = s => !WB || shown(map.level[s], SG[s], T) >= 0.5, cnt = (segs, k) => segs.filter(s => s < ROADS.n_streets && map.level[s] === k && act(s)).length;
  if (map.start < 0 && map.end < 0) { el.innerHTML = "Search above, or press “Set start” and click the map."; return; }
  if (map.start < 0 || map.end < 0) { el.innerHTML = "Now click the other end of the trip."; return; }
  if (!safe) { el.innerHTML = "No road connection found between these two points."; return; }
  let h = (WB ? `At ${labelFor(Math.floor(T))}: ` : "") + `<b>safer route ${km(safe)} km</b><br>`;
  if (forced) h += `Every way through crosses a high-risk street. This one crosses ${cnt(safe, 3)}.`;
  else if (!map.plainSegs || (cnt(plain, 3) === 0 && cnt(plain, 2) === 0)) { h += "The shortest route is already clear of high-risk streets."; map.plainSegs = null; }
  else h += `The shortest route (dashed, ${km(plain)} km) crosses ${cnt(plain, 3)} high-risk and ${cnt(plain, 2)} medium-risk sections. This one crosses ${cnt(safe, 3)} and ${cnt(safe, 2)}.`;
  el.innerHTML = h;
}

/* ---------------------------------------------------------------- street search */
const ABBR = { street: "st", avenue: "ave", av: "ave", boulevard: "blvd", road: "rd", place: "pl", drive: "dr", parkway: "pkwy", lane: "ln", court: "ct",
  terrace: "ter", highway: "hwy", expressway: "expy", square: "sq", west: "w", east: "e", north: "n", south: "s", saint: "st" };
const BOROS = { manhattan: 1, bronx: 2, "the bronx": 2, brooklyn: 3, queens: 4, "staten island": 5 };
const norm = q => q.toLowerCase().replace(/[.,']/g, " ").replace(/(\d+)(st|nd|rd|th)\b/g, "$1").split(/\s+/).filter(Boolean).map(w => ABBR[w] || w).join(" ");
let NAMEIDX = null;
function buildNames() {
  NAMEIDX = { norm: ROADS.names.map(norm), segs: ROADS.names.map(() => []) };
  for (let s = 0; s < ROADS.name.length; s++) NAMEIDX.segs[ROADS.name[s]].push(s);
  const opts = ROADS.names.map((n, i) => [n, NAMEIDX.segs[i].length]).filter(x => x[1] >= 2 && x[0] !== "Unnamed road").sort((a, b) => b[1] - a[1]).slice(0, 4000);
  $("street-names").innerHTML = opts.map(([n]) => `<option value="${esc(n)}">`).join("");
}
function segsFor(part, boro) {
  const q = norm(part); if (!q) return [];
  const pick = ids => { let segs = ids.flatMap(i => NAMEIDX.segs[i]); if (boro) segs = segs.filter(s => s < ROADS.n_streets && ROADS.boro[s] === boro); return segs; };
  const exact = pick(NAMEIDX.norm.flatMap((n, i) => n === q ? [i] : [])); if (exact.length) return exact;
  const words = q.split(" "), whole = n => { const w = n.split(" "); return words.every(x => w.includes(x)); };   // all typed words, as whole words
  return pick(NAMEIDX.norm.flatMap((n, i) => whole(n) ? [i] : []).sort((x, y) => NAMEIDX.segs[y].length - NAMEIDX.segs[x].length).slice(0, 6));
}
function resolvePlace(text) {
  let q = text.trim(), boro = 0; const m = q.match(/,\s*([a-z ]+)$/i);
  if (m && BOROS[m[1].trim().toLowerCase()]) { boro = BOROS[m[1].trim().toLowerCase()]; q = q.slice(0, m.index); }
  const parts = q.split(/\s*(?:&|\/|@|\band\b|\bat\b)\s*/i).filter(Boolean);
  if (!parts.length) return { err: "Type a street name." };
  const A = segsFor(parts[0], boro);
  if (!A.length) return { err: `No street called “${parts[0]}” found${boro ? " in that borough" : ""}.` };
  if (parts.length > 1) {
    const B = segsFor(parts[1], boro); if (!B.length) return { err: `No street called “${parts[1]}” found.` };
    const nodesA = new Set(A.flatMap(s => [GRAPH.a[s], GRAPH.b[s]]));
    const meet = B.flatMap(s => [GRAPH.a[s], GRAPH.b[s]]).filter(n => nodesA.has(n));
    if (!meet.length) return { err: `${parts[0]} and ${parts[1]} don't cross. Try one street, or add the borough.` };
    return { node: meet[0], label: `${ROADS.names[ROADS.name[A[0]]]} & ${ROADS.names[ROADS.name[B[0]]]}` };
  }
  let segs = A;                                             // one street: if it is in several boroughs, take the one with most of it
  if (!boro) { const cnt = {}; A.forEach(s => { const b = s < ROADS.n_streets ? ROADS.boro[s] : 0; cnt[b] = (cnt[b] || 0) + 1; });
    const best = +Object.entries(cnt).sort((x, y) => y[1] - x[1])[0][0]; if (best) segs = A.filter(s => s < ROADS.n_streets && ROADS.boro[s] === best); }
  const cx = segs.reduce((t, s) => t + GRAPH.mx[s], 0) / segs.length, cy = segs.reduce((t, s) => t + GRAPH.my[s], 0) / segs.length;
  let mid = segs[0], bd = Infinity; for (const s of segs) { const d = ((GRAPH.mx[s] - cx) * KX) ** 2 + (GRAPH.my[s] - cy) ** 2; if (d < bd) { bd = d; mid = s; } }
  const bn = ["", "Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"][segs[0] < ROADS.n_streets ? ROADS.boro[segs[0]] : 0];
  return { node: GRAPH.a[mid], label: ROADS.names[ROADS.name[mid]] + (bn ? `, ${bn}` : "") };
}
function fitRoute() {
  const segs = map.routeSegs || []; if (!segs.length) return;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const s of segs) { x0 = Math.min(x0, wx(GRAPH.mx[s])); x1 = Math.max(x1, wx(GRAPH.mx[s])); y0 = Math.min(y0, wy(GRAPH.my[s])); y1 = Math.max(y1, wy(GRAPH.my[s])); }
  const L = 250, R = 60, T = 70, B = 90, W = map.cv.clientWidth - L - R, H = map.cv.clientHeight - T - B;   // keep clear of the legend, buttons and time bar
  map.s = Math.min(0.5, Math.min(W / ((x1 - x0) * 1.15 || 1), H / ((y1 - y0) * 1.15 || 1)));
  map.ox = L + W / 2 - (x0 + x1) / 2 * map.s; map.oy = T + H / 2 - (y0 + y1) / 2 * map.s; draw();
}
function searchRoute(e) {
  e.preventDefault();
  const a = resolvePlace($("q-from").value), b = resolvePlace($("q-to").value);
  if (a.err || b.err) { $("route-info").innerHTML = esc(a.err || b.err); return; }
  map.start = a.node; map.end = b.node; map.pick = null; syncPick();
  computeRoute(); $("route-info").innerHTML = `<b>${esc(a.label)}</b> → <b>${esc(b.label)}</b><br>` + $("route-info").innerHTML; fitRoute();
}

/* ---------------------------------------------------------------- time + sensor list */
const fmtIn = v => v == null ? "no data" : v < 0.05 ? "dry" : v.toFixed(1) + " in";
let lastSig = "", lastSlot = -1;
function setTime(t) {
  T = Math.max(0, Math.min(287, t)); $("time").value = T; const slot = Math.floor(T);
  $("clock").textContent = labelFor(slot);
  if (MODE === "past") {
    const n = floodedNow(T);
    $("clock-note").innerHTML = n ? `<b>${n.toLocaleString()} high-risk streets</b>` : "no streets at risk";
  }
  if (MODE === "live" && WB) {
    const n = floodedNow(T), dh = (T - NOW_SLOT) * 5, when = Math.abs(dh) < 3 ? "now" : dh > 0 ? `in ${dh >= 60 ? (dh / 60).toFixed(1) + " h" : Math.round(dh) + " min"}` : `${Math.round(-dh)} min ago`;
    $("clock-note").innerHTML = (n ? `<b>${n.toLocaleString()} high-risk streets</b>` : "no streets at risk") + ` · ${when}`;
  }
  if (WB) {
    const sig = [1, 2, 3].map(l => [0, 1, 2].map(gi => +(shown(l, gi, T) >= 0.5)).join("")).join("");
    if (sig !== lastSig) { lastSig = sig; if (map.start >= 0 && map.end >= 0) computeRoute(); }
  }
  if (!WB) $("clock-note").textContent = "";
  if (typeof skyAt === "function") skyAt(T);
  if (slot !== lastSlot) { lastSlot = slot; sensorList(); if (sensorId && MODE === "past") drawDepth(); if (MODE === "past") rainPast(slot); }
  draw();
}
const sensorState = v => v == null || v <= 0.5 ? "dry" : v >= 4 ? "flood" : "wet";
const shortName = n => n.replace(/^[A-Z]{1,2} ?- ?/, "");
function sensorList() {
  const el = $("sensor-list"), live = MODE === "live";
  document.querySelectorAll("#ahead-seg button").forEach(b => b.setAttribute("aria-pressed", b.dataset.a === String(ahead)));
  if (!live && !STORM.has_pred) { $("sense-sum").innerHTML = ""; el.innerHTML = ""; $("sensor-note").textContent = "Depth forecasts exist for storms from July to October 2026, the months the models were tested on. Pick one of those storms."; return; }
  const rows = live ? STORM.sensors.filter(q => q.fc).map(q => ({ q, f: q.fc[(+ahead) / 5 - 1], ref: q.now }))
                    : STORM.sensors.filter(q => q.pred).map(q => { const r = forecastAt(q, T) || {}; return { q, f: r.f, ref: r.a }; });
  const dryAll = live ? 0 : STORM.sensors.filter(q => q.observed && !q.pred).length;
  const n = { flood: 0, wet: 0, dry: dryAll }; rows.forEach(r => n[sensorState(r.f)]++);
  $("sense-sum").innerHTML = [["flood", "flooding"], ["wet", "some water"], ["dry", "dry"]].map(([k, l]) => `<div class="s-${k}"><b>${n[k]}</b><span>${l}</span></div>`).join("");
  const wet = rows.filter(r => sensorState(r.f) !== "dry" || (!live && sensorState(r.ref) !== "dry")).sort((a, b) => (b.f || 0) - (a.f || 0) || (b.ref || 0) - (a.ref || 0));
  el.innerHTML = wet.map(r => { const st = sensorState(r.f);
    const sub = live ? (r.ref == null ? "" : `now ${fmtIn(r.ref)}`) : `really ${fmtIn(r.ref)}`;
    return `<li data-id="${esc(r.q.id)}"><span class="nm">${esc(shortName(r.q.name))}${sub ? `<small>${sub}</small>` : ""}</span><span class="tag ${st}">${st === "dry" ? "dry" : fmtIn(r.f)}</span></li>`; }).join("")
    || `<li style="cursor:default; color:var(--soft)">All sensors should stay dry ${live ? "in the next " + ahead + " minutes" : "at this time"}.</li>`;
  el.querySelectorAll("li[data-id]").forEach(li => li.onclick = () => live ? (sensorId = li.dataset.id, draw()) : selectSensor(li.dataset.id));
  $("sensor-note").textContent = live
    ? "Forecast by CNN-LSTM with the ANN rain forecast. " + (STORM.depth_connected ? "Starts from the sensors' current depths." : "Live sensor depths are not connected, so each forecast starts from a dry street: a lower bound.")
    : `${nice(model)} forecast made at ${slotLabel(T)} for ${hhmm((T + 1) * 5 + +ahead)}. “Really” is what the sensor measured. Click a sensor to see its chart below.`;
}
let playing = false, lastFrame = 0, played = false;
const SLOTS_PER_SEC = 6;                                       // 30 minutes of storm per second
function rainSpan() { let a = 0, b = 287; if (!WB) return [a, b]; const m = t => Math.max(WB[0][t], WB[1][t], WB[2][t]);
  while (a < 287 && m(a) < 0.5) a++; while (b > a && m(b) < 0.5) b--; return [Math.max(0, a - 12), Math.min(287, b + 12)]; }
function frame(now) { if (!playing) return; const dt = lastFrame ? (now - lastFrame) / 1000 : 0; lastFrame = now;
  const end = MODE === "live" ? 287 : rainSpan()[1]; if (T >= end) { play(); return; } setTime(T + dt * SLOTS_PER_SEC); requestAnimationFrame(frame); }
function play() { if (playing) { playing = false; $("play").textContent = "▶"; return; }
  let [a, b] = rainSpan(); if (MODE === "live") a = NOW_SLOT; if (!played || T >= b - 1) setTime(a); played = true;
  playing = true; lastFrame = 0; $("play").textContent = "❚❚"; requestAnimationFrame(frame); }


/* ---------------------------------------------------------------- raining now: depth 5, 15, 30 and 40 minutes ahead
   The live model (ANN rain forecast -> CNN-LSTM) forecasts every sensor 5 to 40 minutes ahead; four of those steps are shown,
   each with how far it was off on the July-October 2026 test storms (typical miss = RMSE, flood-related windows).
   TRUST is measured on this exact model file by scripts/16a_rain_now_past.py (7,028 flood-related test windows). */
const NOW_STEPS = [[0, 5], [2, 15], [5, 30], [7, 40]];
const TRUST = { 5: [1.2, 0.80], 15: [1.8, 0.53], 30: [2.3, 0.21], 40: [2.5, 0.08] };
function trustTable(el) {
  table(el, ["How far to trust it", "", ...NOW_STEPS.map(([, m]) => `${m} min`)],
    [["Typical miss on 2026 test storms", "", ...NOW_STEPS.map(([, m]) => "± " + TRUST[m][0].toFixed(1) + " in")],
     ["R² (1 = perfect, 0 = no better than the average)", "", ...NOW_STEPS.map(([, m]) => TRUST[m][1].toFixed(2))]]);
}

/* ---------------------------------------------------------------- raining now, on a past storm
   q.nc[m][p]: the main model's depth forecast made at slot p for m minutes later (scripts/16a_rain_now_past.py);
   q.ncr[m][p]: the rain its ANN expected by then; q.obs[p + m/5]: what the sensor really measured. */
const RP_MIN = [5, 15, 30, 40];
const cityRain = (st, a, b) => { const gs = Object.values(st.gauges || {}); if (!gs.length) return 0; let s = 0, n = 0;
  for (const g of gs) { let v = 0, ok = false; for (let i = Math.max(0, a); i <= Math.min(287, b); i++) if (g[i] != null) { v += g[i]; ok = true; } if (ok) { s += v; n++; } } return n ? s / n : 0; };
const ncSensors = st => st.sensors.filter(q => q.nc);
function rpHeavy(st) { let best = -1, bv = -1; const qs = ncSensors(st);
  for (let p = 0; p < 288; p++) { if (!qs.some(q => q.nc["5"][p] != null)) continue; const r = cityRain(st, p - 2, p); if (r > bv) { bv = r; best = p; } } return best; }
function rainPast(p) {
  const st = STORM; if (!st || MODE !== "past") return;
  const qs = ncSensors(st), on = p != null && qs.some(q => q.nc["5"][p] != null);
  $("rp-time").value = p ?? 0; $("rp-clock").textContent = slotLabel(p ?? 0);
  trustTable("t-rp-trust");
  if (!qs.length) {
    $("rp-title").textContent = st.has_pred ? "No sensor had water in this storm" : "No forecasts for this storm";
    $("rp-say").textContent = st.has_pred ? "The flood sensors stayed dry all day, so there is nothing to forecast." : "The 5 to 40 minute forecasts exist for July to October 2026, the months the model was tested on. Pick one of those storms above.";
    $("t-rp").innerHTML = ""; $("rp-note").textContent = ""; $("fe-box").hidden = true; return;
  }
  const r15 = cityRain(st, p - 2, p), raining = r15 > 0;
  $("rp-title").textContent = `${slotLabel(p)} on ${fmtDate(st.date)}: ` + (raining ? `it is raining, ${r15.toFixed(1)} mm in the last 15 minutes` : "no rain in the last 15 minutes");
  if (!on) {
    $("rp-say").textContent = "The model only runs while a storm is on at a sensor (from an hour before the water to after it drains). Move the time, or press “Heaviest rain”.";
    $("t-rp").innerHTML = ""; $("rp-note").textContent = ""; fePast(p, []); return;
  }
  const at = (q, m) => q.nc[m][p], real = (q, m) => p + m / 5 <= 287 ? q.obs[p + m / 5] : null;
  const live = qs.filter(q => q.nc["5"][p] != null);
  const peak = q => Math.max(0, ...RP_MIN.map(m => at(q, m) || 0), ...RP_MIN.map(m => real(q, m) || 0), q.obs[p] || 0);
  const list = live.filter(q => peak(q) >= 0.05).sort((a, b) => peak(b) - peak(a)).slice(0, 10);
  const cls = v => v == null ? "" : v >= 4 ? "flood" : v >= 0.05 ? "wet" : "dry";
  const cell = (f, r) => `<td class="n rp"><b class="${cls(f)}">${esc(fmtIn(f))}</b><small>really ${esc(fmtIn(r))}</small></td>`;
  const cnt = (m, real) => live.filter(q => ((real ? (p + m / 5 <= 287 ? q.obs[p + m / 5] : null) : at(q, m)) || 0) >= 1).length;
  const rainBy = m => live.reduce((a, q) => a + (q.ncr[m][p] || 0), 0) / live.length;
  let h = `<tr><th>Sensor</th><th class="n">Now (measured)</th>${RP_MIN.map(m => `<th class="n">In ${m} min</th>`).join("")}</tr>`;
  h += `<tr class="rp-sum"><td>Sensors at 1 in or more</td><td class="n">${live.filter(q => (q.obs[p] || 0) >= 1).length} of ${live.length}</td>${RP_MIN.map(m => `<td class="n rp"><b>${cnt(m)} of ${live.length}</b><small>really ${cnt(m, true)}</small></td>`).join("")}</tr>`;
  h += `<tr class="rp-sum"><td>Rain by then (city average)</td><td class="n">${r15.toFixed(1)} mm last 15 min</td>${RP_MIN.map(m => `<td class="n rp"><b>${rainBy(m).toFixed(1)} mm</b><small>really ${cityRain(st, p + 1, p + m / 5).toFixed(1)} mm</small></td>`).join("")}</tr>`;
  h += list.map(q => `<tr><td>${esc(shortName(q.name))}</td><td class="n"><b class="${cls(q.obs[p])}">${esc(fmtIn(q.obs[p]))}</b></td>${RP_MIN.map(m => cell(at(q, m), real(q, m))).join("")}</tr>`).join("");
  if (!list.length) h += `<tr><td colspan="6" style="color:var(--soft)">All ${live.length} sensors are dry now and forecast to stay dry for 40 minutes.</td></tr>`;
  $("t-rp").innerHTML = h;
  const top = list[0];
  $("rp-say").textContent = !top ? (raining ? "It is raining, but the model expects no water at any sensor in the next 40 minutes." : "No water now and none expected in the next 40 minutes.")
    : `Deepest at ${shortName(top.name)}: now ${fmtIn(top.obs[p])}; forecast ${RP_MIN.map(m => `${fmtIn(at(top, m))} in ${m} min`).join(", ")}. It really measured ${RP_MIN.map(m => fmtIn(real(top, m))).join(", ")}.`;
  fePast(p, list);
  $("rp-note").textContent = `Forecast made at ${slotLabel(p)} from the last hour of water depth and rain at each sensor's nearest gauge, nothing later. Big number: the forecast. “Really”: what the sensor measured then. ${live.length} sensors had the model running at this moment; sensors that stayed dry are left out of the list. Times are New York time.`;
}
function rainNow(L, raining) {
  const sens = L.sensors.filter(s => s.fc), w = L.weather || {};
  if (!sens.length) { $("rainnow-box").hidden = true; return; }
  $("rainnow-box").hidden = false;
  const peak = s => Math.max(0, ...s.fc.map(v => v || 0)), top = sens.slice().sort((a, b) => peak(b) - peak(a)).slice(0, 8);
  const rainBy = k => sens.reduce((a, s) => a + (s.rain_fc || []).slice(0, k + 1).reduce((x, y) => x + y, 0), 0) / sens.length;
  const nowCell = s => L.depth_connected && s.now != null ? fmtIn(s.now) : "not measured";
  const head = ["Sensor", "Now", ...NOW_STEPS.map(([, m]) => `In ${m} min`)];
  const rows = [["Whole city: sensors at 1 in or more", L.depth_connected ? String(sens.filter(s => (s.now || 0) >= 1).length) : "–", ...NOW_STEPS.map(([k]) => `${sens.filter(s => (s.fc[k] || 0) >= 1).length} of ${sens.length}`)],
    ["Rain expected by then (ANN, city average)", (w.rain_now ?? 0) + " mm now", ...NOW_STEPS.map(([k]) => rainBy(k).toFixed(1) + " mm")],
    ...top.map(s => [shortName(s.name), nowCell(s), ...NOW_STEPS.map(([k]) => fmtIn(s.fc[k]))])];
  table("t-rainnow", head, rows);
  trustTable("t-trust");
  const deepS = top[0], deep15 = deepS ? deepS.fc[2] : 0, anyWater = sens.some(s => s.fc.some(v => v >= 0.05));
  $("rainnow-title").textContent = raining ? `It is raining now: flood depth in 5, 15, 30 and 40 minutes` : `Not raining now: flood depth in 5, 15, 30 and 40 minutes`;
  $("rainnow-say").textContent = !anyWater ? (raining ? `It is raining (${w.rain_now ?? 0} mm in the last 15 minutes), but not hard enough for any sensor to get water in the next 40 minutes.` : "No sensor is forecast to get water in the next 40 minutes.")
    : `${raining ? `It is raining (${w.rain_now ?? 0} mm in the last 15 minutes). ` : ""}The deepest water is expected at ${shortName(deepS.name)}: ${fmtIn(deepS.fc[0])} in 5 minutes, ${fmtIn(deep15)} in 15, ${fmtIn(deepS.fc[5])} in 30 and ${fmtIn(deepS.fc[7])} in 40.`;
  $("rainnow-note").textContent = (L.depth_connected ? "" : "Live sensor depths are not connected, so each forecast starts from a dry street: where water is already standing, real depths will be higher. ")
    + `Forecast made ${L.now_local} by the ANN rain forecast and CNN-LSTM. The further ahead, the less exact: by 40 minutes the depth is a rough guide, so use “Highest in 40 min” in Today’s prediction for planning.`;
}

/* ---------------------------------------------------------------- Improved LSTM vs Feature-enhanced LSTM (notebook models, scripts/15a + 16b)
   q.fe.imp / q.fe.enh [m][p]: forecast made at slot p for m minutes later; q.rn[p]: rain at the sensor's gauge (mm per 5 min). */
const FE_COND = [   // nb_rainfall_condition_analysis_nyc.csv: condition, points, [RMSE, R²] improved, [RMSE, R²] feature-enhanced
  ["No rain", 37891, [1.24, 0.71], [1.56, 0.53]], ["Light rain (up to 0.3 mm)", 8187, [1.62, 0.47], [1.56, 0.51]],
  ["Moderate rain (0.3 to 0.8 mm)", 4097, [2.69, 0.31], [2.66, 0.32]], ["Heavy rain (over 0.8 mm)", 4513, [4.12, 0.01], [3.95, 0.09]],
  ["Rapidly increasing rain", 3250, [3.49, 0.06], [3.35, 0.13]]];
const FE_COL = { imp: "#2a7fd4", enh: "#c2185b" }, FE_NAME = { imp: "Improved LSTM", enh: "Feature-enhanced LSTM" };
function feCond() {
  const max = Math.max(...FE_COND.flatMap(r => [r[2][0], r[3][0]]));
  $("fe-cond").innerHTML = FE_COND.map(([c, n, a, b]) => `<div class="hb"><span class="hb-l">${esc(c)} <small>${n.toLocaleString()} checks</small></span>` +
    [["imp", a], ["enh", b]].map(([k, v]) => `<span class="hb-r"><i style="width:${(v[0] / max * 100).toFixed(1)}%; background:${FE_COL[k]}"></i><em>${v[0].toFixed(2)} in · R² ${v[1].toFixed(2)}</em></span>`).join("") + "</div>").join("") +
    `<div class="keys" style="margin-top:8px"><span><i style="background:${FE_COL.imp}"></i>Improved LSTM</span><span><i style="background:${FE_COL.enh}"></i>Feature-enhanced LSTM</span></div>`;
}
function fePast(p, order) {
  const st = STORM, qs = st.sensors.filter(q => q.fe), box = $("fe-box");
  if (!qs.length) { box.hidden = true; return; } box.hidden = false; feCond();
  const on = qs.filter(q => q.fe.imp["5"][p] != null);
  const real = (q, m) => p + m / 5 <= 287 ? q.obs[p + m / 5] : null, f1 = v => v == null ? "–" : fmtIn(v);
  // storm accuracy (does not depend on the moment)
  const acc = { imp: RP_MIN.map(() => [0, 0]), enh: RP_MIN.map(() => [0, 0]) };
  for (const q of qs) for (let t = 0; t < 288; t++) {
    if (q.fe.imp["5"][t] == null) continue;
    const wet = q.obs.slice(Math.max(0, t - 11), Math.min(288, t + 9)).some(v => (v || 0) > 0); if (!wet) continue;
    RP_MIN.forEach((m, i) => { const a = t + m / 5 <= 287 ? q.obs[t + m / 5] : null; if (a == null) return;
      for (const k of ["imp", "enh"]) { const e = q.fe[k][m][t] - a; acc[k][i][0] += e * e; acc[k][i][1]++; } });
  }
  const rm = (k, i) => acc[k][i][1] ? Math.sqrt(acc[k][i][0] / acc[k][i][1]) : null;
  $("t-fe-acc").innerHTML = `<tr><th>Model</th>${RP_MIN.map(m => `<th class="n">${m} min</th>`).join("")}</tr>` + ["imp", "enh"].map(k => `<tr><td style="color:${FE_COL[k]}">${FE_NAME[k]}</td>${RP_MIN.map((_, i) => {
    const a = rm(k, i), o = rm(k === "imp" ? "enh" : "imp", i); return `<td class="n">${a == null ? "–" : (o != null && a < o ? `<b>${a.toFixed(2)}</b>` : a.toFixed(2))}</td>`; }).join("")}</tr>`).join("");
  if (!on.length) {
    $("fe-say").textContent = "Neither model is running at this moment (it runs while a storm is on at a sensor). Move the time above.";
    $("t-fe").innerHTML = ""; $("t-feat").innerHTML = ""; $("feat-title").textContent = "What the Feature-enhanced LSTM reads"; return;
  }
  const list = (order || []).filter(q => q.fe && q.fe.imp["5"][p] != null).slice(0, 6);
  const show = list.length ? list : on.slice(0, 3);
  let wins = { imp: 0, enh: 0 };
  let h = `<tr><th>Sensor</th><th>Model</th><th class="n">Now</th>${RP_MIN.map(m => `<th class="n">In ${m} min</th>`).join("")}</tr>`;
  for (const q of show) {
    const err = k => RP_MIN.map(m => real(q, m) == null ? null : Math.abs(q.fe[k][m][p] - real(q, m)));
    const ei = err("imp"), ee = err("enh");
    ei.forEach((v, i) => { if (v != null && ee[i] != null && Math.abs(v - ee[i]) >= 0.05) wins[v < ee[i] ? "imp" : "enh"]++; });
    const row = (k, e, o, first) => `<tr class="${first ? "fe-first" : ""}">${first ? `<td rowspan="3">${esc(shortName(q.name))}</td>` : ""}<td style="color:${FE_COL[k]}">${FE_NAME[k]}</td><td class="n">${first ? f1(q.obs[p]) : ""}</td>` +
      RP_MIN.map((m, i) => `<td class="n">${e[i] != null && o[i] != null && e[i] + 0.05 <= o[i] ? `<b>${f1(q.fe[k][m][p])}</b>` : f1(q.fe[k][m][p])}</td>`).join("") + "</tr>";
    h += row("imp", ei, ee, true) + row("enh", ee, ei, false) + `<tr class="real"><td>Really</td><td class="n"></td>${RP_MIN.map(m => `<td class="n">${real(q, m) == null ? "?" : f1(real(q, m))}</td>`).join("")}</tr>`;
  }
  $("t-fe").innerHTML = h;
  $("fe-say").innerHTML = `At ${slotLabel(p)}, for the ${show.length} sensor${show.length > 1 ? "s" : ""} below: the <b>Improved LSTM</b> was closer ${wins.imp} time${wins.imp === 1 ? "" : "s"}, the <b>Feature-enhanced LSTM</b> ${wins.enh} time${wins.enh === 1 ? "" : "s"} (ties within 0.05 in not counted). Bold marks the closer forecast.`;
  // the features at the top sensor, last 30 minutes
  const q = show[0], rn = q.rn || [], val = j => rn[j] ?? 0, sum = (j, k) => { let s = 0; for (let i = Math.max(0, j - k + 1); i <= j; i++) s += val(i); return s; };
  $("feat-title").textContent = `What the Feature-enhanced LSTM reads at ${shortName(q.name)}`;
  const js = [...Array(6)].map((_, k) => p - 5 + k).filter(j => j >= 0);
  $("t-feat").innerHTML = `<tr><th>Time</th><th class="n">Rain (mm)</th><th class="n">Depth (in)</th><th class="n">Intensity (mm/h)</th><th class="n">15-min total</th><th class="n">30-min total</th><th class="n">60-min total</th><th class="n">Change</th></tr>` +
    js.map(j => `<tr${j === p ? ' class="real"' : ""}><td>${slotLabel(j)}${j === p ? " (now)" : ""}</td><td class="n">${val(j).toFixed(2)}</td><td class="n">${(q.obs[j] ?? 0).toFixed(1)}</td><td class="n">${(val(j) * 12).toFixed(1)}</td><td class="n">${sum(j, 3).toFixed(2)}</td><td class="n">${sum(j, 6).toFixed(2)}</td><td class="n">${sum(j, 12).toFixed(2)}</td><td class="n">${(j > 0 ? val(j) - val(j - 1) : 0).toFixed(2)}</td></tr>`).join("");
}

/* ---------------------------------------------------------------- today's prediction (live model output) */
function today(L) {
  const sc = L.scenarios || [], sens = L.sensors.filter(s => s.fc), dry = L.what_if;
  // the highest forecast depth in the next 40 minutes: far more reliable than the depth at exactly 40 minutes
  // (test R² 0.55 vs 0.11, scripts/14a): the models know how big a flood gets better than the minute it peaks
  const peak = s => Math.max(0, ...s.fc.map(v => v || 0));
  const deep = sens.reduce((m, s) => Math.max(m, peak(s)), 0), wet40 = sens.filter(s => peak(s) > 0.5).length;
  const rain40 = sens.length ? sens.reduce((a, s) => a + (s.rain_fc || []).reduce((x, y) => x + y, 0), 0) / sens.length : 0;
  const run = L.model_run;
  $("today-lede").textContent = `Worked out by the models at ${L.now_local} (New York time) from the rain forecast for the next 24 hours` +
    (run ? `: the road model scored ${run.road_streets.toLocaleString()} streets and the CNN-LSTM forecast ${run.depth_sensors} sensors in ${run.seconds} s` : "") + `. It is recalculated every ${L.refresh_minutes || 15} minutes, so these numbers follow the weather.`;
  $("today-facts").innerHTML = [[L.city_rain_mm + " mm", "rain forecast, next 24 hours"], [L.peak_1h_mm + " mm", "in the wettest hour"],
    [dry ? "0" : L.counts[0].toLocaleString(), "streets at high flood risk" + (dry ? "" : `, ${L.counts[1].toLocaleString()} medium`)],
    [rain40.toFixed(1) + " mm", "rain the ANN expects in the next 40 minutes"]].map(([v, l]) => `<div class="fact"><b>${v}</b><span>${esc(l)}</span></div>`).join("");
  // rain curve
  if (sc.length) {
    const mm = sc.map(s => s.mm), x = Math.min(mm.length - 1, (L.city_rain_mm <= mm[0] ? -1 : (() => { for (let i = 1; i < mm.length; i++) if (L.city_rain_mm <= mm[i]) return i - 1 + (L.city_rain_mm - mm[i - 1]) / (mm[i] - mm[i - 1]); return mm.length - 1; })()));
    const maxC = Math.max(1, ...sc.map(s => s.counts[0])), maxR = Math.max(1, ...sc.map(s => s.reports));
    chart($("scen-chart"), { n: sc.length, tick: 1, label: i => mm[i] + " mm", minMax: maxC, mark: x, y2: true,
      series: [{ data: sc.map(s => s.counts[0]), color: C.flood, width: 2.6 }, { data: sc.map(s => s.reports / maxR * 100), color: C.warn, width: 2, dash: [5, 4], y2: true }] });
    const next = sc.find(s => s.mm > L.city_rain_mm && s.counts[0] >= 10 && s.counts[0] > (dry ? 0 : L.counts[0]));
    $("scen-say").innerHTML = (dry ? `Today <b>${L.city_rain_mm} mm</b> is forecast, under the 5 mm the road model needs, so no street is flagged.` : `Today <b>${L.city_rain_mm} mm</b> is forecast, so <b>${L.counts[0].toLocaleString()}</b> streets are at high flood risk.`) +
      (next ? ` If the forecast grew to <b>${next.mm} mm</b>, that would be <b>${next.counts[0].toLocaleString()}</b> high-risk streets and about ${Math.round(next.reports)} flooding reports.` : "");
    $("scen-note").textContent = `The road model (XGBoost) runs again for each rain total on the bottom axis, with the storm shaped like ${L.storm_shape} and the ground as wet as it really is after the last 3 days. Right scale: expected flooding reports as a share of the highest point. The curve flattens because the model was trained on real storms, which rarely went past 60 mm.`;
  }
  // boroughs + sensors
  const B = L.boro || {};
  table("t-boro", ["Borough", "High", "Medium", "Low"], Object.entries(B).map(([b, c]) => [b, ...(dry ? [0, 0, 0] : c).map(v => v.toLocaleString())]));
  const top = sens.slice().sort((a, b) => peak(b) - peak(a)).slice(0, 5);
  const anyWet = sens.some(s => s.fc.some(v => v > 0.05));
  if (anyWet) table("t-sens", ["Sensor", "Highest in 40 min", "5 min", "20 min", "40 min"], top.map(s => [shortName(s.name), fmtIn(peak(s)), ...[0, 3, 7].map(k => fmtIn(s.fc[k]))]));
  else $("t-sens").innerHTML = `<tr><td>All ${sens.length} sensors are forecast to stay dry for the next 40 minutes.</td></tr>`;
  $("sens-say").textContent = (wet40 ? `${wet40} of ${sens.length} sensors are forecast to have water some time in the next 40 minutes, the deepest ${fmtIn(deep)}. ` : "")
    + "“Highest in 40 min” is the most reliable number here: the models judge how deep a flood gets much better than the exact minute it peaks. " + (L.depth_connected ? "" : "Live sensor depths are not connected, so each forecast starts from a dry street and comes from the coming rain alone.");
}

/* ---------------------------------------------------------------- page */
// all data comes from the backend API (backend/server.py), never from files in the frontend
/* On the hosted site the live forecast is refreshed every 30 minutes on the repository's live-data branch, while the
   site itself is only redeployed when the code changes. api/source names that file; the copy built into the site is the
   fallback, and whichever is newer wins. On your own computer api/source is null and the server's forecast is used. */
let LIVE_SRC = null;
async function liveJSON() {
  if (LIVE_SRC === null) {
    try { const s = await getJSON("source.json"); LIVE_SRC = s && /^https:\/\/raw\.githubusercontent\.com\/[\w.-]+\/[\w.-]+\/live-data\/live\.json$/.test(s.live) ? s.live : ""; }
    catch (e) { LIVE_SRC = ""; }
  }
  const local = getJSON("live.json").catch(() => null);
  if (!LIVE_SRC) return local;
  const remote = fetch(`${LIVE_SRC}?t=${Math.floor(Date.now() / 60000)}`, { cache: "no-store", credentials: "omit", referrerPolicy: "no-referrer" })
    .then(r => r.ok ? r.json() : null).catch(() => null);
  const [a, b] = await Promise.all([local, remote]);
  const ok = x => x && typeof x === "object" && typeof x.updated === "string";
  if (ok(a) && ok(b)) return b.updated >= a.updated ? b : a;
  if (ok(b)) return b;
  if (ok(a)) return a;
  throw new Error("no live forecast");
}
const API = f => f.startsWith("storm_") ? "api/storm/" + f.slice(6, 16) : "api/" + f.split(".")[0];
const getJSON = async f => { const r = await fetch(API(f), { headers: { Accept: "application/json" }, cache: "no-store" }); if (!r.ok) throw new Error(f + " " + r.status); return r.json(); };
function facts(items) { $("facts").innerHTML = items.map(([v, l]) => `<div class="fact"><b>${esc(v)}</b><span>${esc(l)}</span></div>`).join(""); }
function table(el, head, rows, bestCol, lowBest, mineCol) {
  let bi = -1; if (bestCol != null) { let bv = lowBest ? Infinity : -Infinity; rows.forEach((r, i) => { if (String(r[0]).includes("upper bound")) return; const v = parseFloat(r[bestCol]); if (lowBest ? v < bv : v > bv) { bv = v; bi = i; } }); }
  $(el).innerHTML = "<tr>" + head.map((h, i) => `<th class="${i ? "n" : ""}">${esc(h)}</th>`).join("") + "</tr>" +
    rows.map((r, i) => `<tr class="${i === bi ? "best" : ""} ${mineCol && MINE.has(mineCol[i]) ? "mine" : ""}">` + r.map((c, j) => `<td class="${j ? "n" : ""}">${esc(c)}</td>`).join("") + "</tr>").join("");
}
function busiestSlot(st) { let best = 0, bv = -1; for (let t = 0; t < 288; t++) { let v = 0; for (const q of st.sensors) if (q.obs) v += q.obs[t] || 0; if (v > bv) { bv = v; best = t; } } return Math.max(0, best - (+ahead) / 5 - 2); }

function paintStreets(st) {
  map.level = new Uint8Array(ROADS.n_streets); map.prob = new Uint16Array(ROADS.n_streets);
  const TH = INDEX.levels.map(l => l[1] * 1e4), by = [[], [], [], []];
  st.risk_ix.forEach((s, i) => { const p = st.risk_p[i], lv = p >= TH[0] ? 3 : p >= TH[1] ? 2 : 1; map.level[s] = lv; map.prob[s] = p; by[lv].push(s); });
  map.paths.watch = segPath(by[1]); map.paths.medium = segPath(by[2]); map.paths.high = segPath(by[3]);
  map.groups = { 1: [], 2: [], 3: [] };
  for (const lv of [1, 2, 3]) for (let gi = 0; gi < 3; gi++) { const segs = by[lv].filter(s => SG[s] === gi); map.groups[lv][gi] = { p: segPath(segs), n: segs.length }; }
  $("lvl").innerHTML = [["High risk", C.flood], ["Medium risk", C.warn], ["Low risk", C.watch]].map(([n, c], i) => `<div><b style="color:${c}">${st.counts[i].toLocaleString()}</b><span>${n}</span></div>`).join("");
  computeRoute();
}
const val = (v, u) => v == null || Number.isNaN(+v) ? "–" : v + u;   // a value the weather service left out shows as a dash
const compass = d => d == null ? "" : ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][Math.round(d / 45) % 8];
async function loadLive() {
  let L;
  try { L = await liveJSON(); } catch (e) { L = null; }
  if (!L) {
    $("headline").textContent = "Live data is not running";
    $("summary").textContent = LIVE_SRC ? "The live forecast is not available right now. It is refreshed every 30 minutes; meanwhile you can look at past storms." : "Start the live app from the project folder with: python3 backend/app.py. It fetches the weather every 15 minutes. You can still look at past storms.";
    $("facts").innerHTML = ""; $("now").style.display = "none"; return false;
  }
  $("now").style.display = ""; if (playing) play();
  if (L.what_if) { L.risk_ix = []; L.risk_p = []; L.counts = [0, 0, 0]; }      // no storm on the way: every street clear (also for files from older versions)
  LIVE = STORM = L; WB = liveBuckets(L); played = false; lastSig = ""; lastSlot = -1;
  const w = L.weather, raining = (w.rain_now || 0) > 0 || L.rain15.some(r => r[2] && r[1] > 0);
  $("headline").textContent = L.what_if ? "No flooding expected: streets are clear" : L.counts[0] > 150 || L.peak_1h_mm >= 10 ? "Flooding likely on some streets" : "Some streets may flood";
  $("summary").textContent = `${L.now_local} in New York: ${(w.desc || "weather unavailable").toLowerCase()}, ${val(w.temp, " °C")}. ` + (L.what_if ? "Less than 5 mm of rain is forecast for the next 24 hours." : `About ${L.city_rain_mm} mm of rain is forecast over the next 24 hours, up to ${L.peak_1h_mm} mm in one hour.`);
  facts([[val(w.temp, " °C"), w.desc || "weather unavailable"], [(w.rain_now ?? 0) + " mm", raining ? "rain in the last 15 minutes" : "no rain right now"], [L.city_rain_mm + " mm", "rain expected in the next 24 hours"], [L.what_if ? "–" : L.counts[0], "streets flagged high risk"]]);
  // say so plainly if the live forecast is old (the updater stopped, or the laptop slept)
  const age = (Date.now() - Date.parse((L.updated || "").replace(" UTC", "Z").replace(" ", "T"))) / 60000;
  const every = L.refresh_minutes || 15;                     // 15 on your own computer, 30 on the hosted site
  if (age > every * 2 + 15) {
    const ago = age < 120 ? `${Math.round(age)} minutes` : age < 2880 ? `${Math.round(age / 60)} hours` : `${Math.round(age / 1440)} days`;
    $("live-banner").textContent = `This live forecast is ${ago} old: the updater has not run since ${L.updated}. ` + (L.host === "static"
      ? `The hosted site is normally refreshed every ${every} minutes, so the scheduled update is late or failed.`
      : "Start it with python3 backend/app.py; it refreshes every 15 minutes.");
    $("live-banner").hidden = false;
  } else $("live-banner").hidden = true;
  $("now-lede").textContent = `Current conditions and forecast from the Open-Meteo weather service, updated every ${L.refresh_minutes || 15} minutes. Last update ${L.updated}.`;
  $("wx-now").innerHTML = [[val(w.temp, " °C"), "temperature"], [val(w.feels, " °C"), "feels like"], [val(w.humidity, " %"), "humidity"], [val(w.cloud, " %"), "cloud cover"],
    [`${val(w.wind, " km/h")} ${compass(w.wind_dir)}`, "wind"], [val(w.gust, " km/h"), "gusts"], [val(w.pressure, " hPa"), "pressure"], [L.past12_mm + " mm", "rain in the last 12 hours"]]
    .map(([v, l]) => `<div><b>${esc(v)}</b><span>${l}</span></div>`).join("");
  const n24 = L.next24;
  chart($("next24-chart"), { n: n24.rain.length, tick: 3, label: i => n24.time[i], minMax: 2, y2: true, series: [{ data: n24.rain, color: C.rain, bars: true }, { data: n24.prob, color: C.warn, width: 1.6, y2: true }] });
  const q = L.rain15, cut = q.findIndex(r => !r[2]);
  chart($("q15-chart"), { n: q.length, tick: 4, label: i => q[i][0], minMax: 1, mark: cut > 0 ? cut - 0.5 : null, series: [{ data: q.map(r => r[1]), color: C.rain, bars: true }] });
  const G = { NYC: "Central Park", LGA: "LaGuardia", JFK: "JFK" }, times = [...new Set(Object.values(L.gauges).flatMap(a => a.map(r => r[0])))].sort().slice(-6);
  $("gauge-table").innerHTML = times.length ? "<tr><th>Gauge</th>" + times.map(t => `<th class="n">${esc(t)}</th>`).join("") + "</tr>" +
    Object.entries(L.gauges).map(([k, a]) => `<tr><td>${esc(G[k] || k)}</td>` + times.map(t => { const r = a.find(x => x[0] === t); return `<td class="n">${r && r[1] != null ? esc(r[1]) : "–"}</td>`; }).join("") + "</tr>").join("")
    : "<tr><td>Gauge readings are not available right now.</td></tr>";
  $("map-lede").textContent = L.what_if ? "Less than 5 mm of rain is forecast for the next 24 hours, so every street is shown clear. Streets light up here when a storm is on the way." : "The map shows the streets as they are now: clear unless rain is falling. Press play, or drag the slider, to see which streets the forecast rain would flood over the next 22 hours and how the safe route changes. Diamonds are flood sensors with the CNN-LSTM depth forecast.";
  $("rep-note").textContent = L.what_if ? "No streets flagged." : `${L.expected_reports} streets are expected to get a flooding report.`;
  paintStreets(L); drawTimeline(); setTime(NOW_SLOT); today(L); rainNow(L, raining); if (typeof heroLive === "function") heroLive(L); return true;
}
async function setMode(m) {
  MODE = m; document.body.className = m;
  document.querySelectorAll(".modes button").forEach(b => b.setAttribute("aria-pressed", b.dataset.mode === m));
  history.replaceState(null, "", m === "past" ? "#" + $("storm").value : "#live");
  if (m === "live") { await loadLive(); } else { await loadStorm($("storm").value); }
}
async function loadStorm(date) {
  const st = STORM = PAST = await getJSON(`storm_${date}.json`), meta = INDEX.storms.find(s => s.date === date);
  if (playing) play(); WB = buckets(st); played = false; lastSig = ""; lastSlot = -1;
  $("map-lede").textContent = "Press play to watch the storm unfold. Streets light up as rain builds at the nearest gauge, riskiest streets first, and clear again as it drains. Diamonds are flood sensors with the chosen model's depth forecast. Set a start and destination to see the safe route change as streets flood.";
  history.replaceState(null, "", "#" + date);
  const flooded = st.sensors.filter(s => s.peak > 0), deep = Math.max(0, ...st.sensors.map(s => s.peak));
  $("headline").textContent = fmtDate(date) + ": " + (deep >= 12 ? "streets under deep water" : deep >= 4 ? "flooding across the city" : deep > 0 ? "shallow street flooding" : "rain, no flooding at the sensors");
  $("summary").textContent = `About ${meta.gauge_mm} mm of rain fell at the gauges, up to ${meta.peak5} mm in five minutes. ${flooded.length} of ${st.sensors.filter(s => s.observed).length} working flood sensors recorded water.`;
  facts([[meta.gauge_mm + " mm", "rain at the gauges"], [meta.peak5 + " mm", "heaviest 5 minutes"], [deep + " in", "deepest water measured"], [st.counts[0], "streets flagged high risk"]]);

  paintStreets(st);
  if (typeof heroPast === "function") heroPast(st, meta);
  $("rep-note").textContent = st.has_reports ? `Flooding was reported on ${st.reported.length} streets; ${st.reported_caught} of them were flagged. Turn on “Reported flooding” to see them.` : "No flooding reports are available for this date.";
  $("b-rep").style.display = st.has_reports ? "" : "none";

  // sensors + time
  const withPred = st.sensors.filter(s => s.obs).sort((a, b) => b.peak - a.peak);
  if (!withPred.some(s => s.id === sensorId)) sensorId = withPred.length ? withPred[0].id : null;
  $("depth-body").style.display = withPred.length ? "" : "none";
  $("depth-lede").textContent = withPred.length ? "The white line is the water the sensor measured. Each coloured line is a model's forecast, drawn at the time it was forecasting for: the closer the lines, the better the model."
    : st.has_pred ? "No sensor recorded water on this day." : "Depth forecasts exist for July to October 2026. Pick a storm from those months.";
  if (withPred.length) {
    $("sensor").innerHTML = withPred.map(s => `<option value="${esc(s.id)}">${esc(s.name)} (peak ${s.peak} in)</option>`).join(""); $("sensor").value = sensorId;
    if (st.storm_errors.length) table("storm-err", ["Model", "5 min", "20 min", "40 min"], st.storm_errors.filter(r => LIVE_MODELS.includes(r[0]) || modelsOn.has(r[0])).map(r => [nice(r[0]), ...r.slice(1)]), 2, true, st.storm_errors.filter(r => LIVE_MODELS.includes(r[0]) || modelsOn.has(r[0])).map(r => r[0]));
    else $("storm-err").innerHTML = "<tr><td>Too few flooded readings to score the models.</td></tr>";
  }
  drawTimeline();
  let peak = 0; for (let t = 0; t < 288; t++) if (floodedNow(t) > floodedNow(peak)) peak = t;
  if (!ncSensors(st).some(q => q.nc["5"][peak] != null)) { const h = rpHeavy(st); if (h >= 0) peak = h; }   // open where the depth model is running
  setTime(peak);

  // weather + rain
  const gs = Object.keys(GAUGE).filter(k => st.gauges[k]);
  chart($("rain-chart"), { n: 288, tick: 36, label: slotLabel, minMax: 2, series: gs.map(k => ({ data: st.gauges[k], color: GAUGE[k][1], width: 1.6 })) });
  $("rain-note").innerHTML = gs.map(k => { const v = st.gauges[k].map(x => x || 0); return `<span class="dot" style="background:${GAUGE[k][1]}"></span>${GAUGE[k][0]} ${v.reduce((a, b) => a + b, 0).toFixed(1)} mm`; }).join(" &nbsp; ") + ". Times are New York time.";
  const w = WX && WX[date], mm = a => Math.min(...a).toFixed(0) + " to " + Math.max(...a).toFixed(0), avg = a => (a.reduce((x, y) => x + y, 0) / a.length).toFixed(0);
  $("wx-lede").textContent = w ? "Hourly weather for New York City that day, and rain measured every 5 minutes at three gauges." : "Rain measured every 5 minutes at three gauges. Temperature and wind appear here once the weather file is downloaded.";
  table("wx-table", ["", ""], w ? [["Temperature", mm(w.temp) + " °C"], ["Humidity", avg(w.humidity) + " %"], ["Strongest wind", Math.max(...w.wind).toFixed(0) + " km/h"], ["Strongest gust", Math.max(...w.gust).toFixed(0) + " km/h"], ["Lowest pressure", Math.min(...w.pressure).toFixed(0) + " hPa"], ["Cloud cover", avg(w.cloud) + " %"], ["Rain (weather model)", st.city_rain_mm + " mm"]]
    : [["Rain (weather model)", st.city_rain_mm + " mm"], ["Wettest hour", Math.max(0, ...st.hourly).toFixed(1) + " mm"], ["Hours with rain", st.hourly.filter(v => v >= 0.5).length]]);
}
function selectSensor(id) { sensorId = id; $("sensor").value = id; drawDepth(); draw(); }
function drawDepth() {
  const q = STORM.sensors.find(s => s.id === sensorId); if (!q || !q.obs) return;
  let a = 288, b = 0; q.obs.forEach((v, i) => { if (v > 0) { a = Math.min(a, i); b = Math.max(b, i); } }); a = Math.max(0, a - 12); b = Math.min(287, b + 12);
  const cut = arr => arr.slice(a, b + 1), n = b - a + 1;
  $("depth-title").textContent = q.name.replace(/^[A-Z]{1,2} ?- ?/, "") + ": water depth in inches, forecast " + horizon + " minutes ahead";
  const series = STORM.models.map((m, i) => modelsOn.has(m) ? { data: cut(q.pred[horizon][i]), color: STYLE[m][0], dash: STYLE[m][1], width: MINE.has(m) ? 2.2 : 1.6 } : null).filter(Boolean);
  series.push({ data: cut(q.obs), color: C.ink, width: 2.8 });
  $("depth-keys").innerHTML = `<span><i style="background:${C.ink}"></i>Measured</span>` + STORM.models.filter(m => modelsOn.has(m)).map(m => `<span><i style="background:${STYLE[m][0]}"></i>${esc(nice(m))}</span>`).join("");
  let pk = 0; q.obs.forEach((v, i) => { if ((v || 0) > (q.obs[pk] || 0)) pk = i; });
  const mi = STORM.models.indexOf("LSTM (standard)"), fpk = mi >= 0 ? q.pred[horizon][mi][pk] : null;
  $("depth-say").innerHTML = `The water peaked at <b>${fmtIn(q.obs[pk])}</b> at ${slotLabel(pk)}.` + (fpk != null ? ` ${horizon} minutes earlier, the standard LSTM expected <b>${fmtIn(fpk)}</b>${Math.abs(fpk - q.obs[pk]) < 1 ? ", close to right" : fpk < q.obs[pk] ? ", too low" : ", too high"}.` : "");
  chart($("depth-chart"), { n, tick: Math.max(1, Math.round(n / 7)), label: i => slotLabel(a + i), minMax: 1, mark: T + (+ahead) / 5 - a, series });
}
function r2Demo() {
  const cv = $("r2-chart"), sl = $("r2-q"); if (!cv) return;
  const n = 60, obs = Array.from({ length: n }, (_, i) => { const t = i / n; return Math.max(0, 9 * Math.exp(-((t - .42) ** 2) / .012) + 3 * Math.exp(-((t - .7) ** 2) / .006)); });
  const mean = obs.reduce((a, b) => a + b, 0) / n, late = obs.map((_, i) => obs[Math.max(0, i - 6)] * 0.8);
  const ssMean = obs.reduce((a, v) => a + (v - mean) ** 2, 0);
  const upd = () => { const k = +sl.value / 100;                      // 1 = perfect, 0 = average, below 0 = worse (late, damped guess pushed further)
    const fc = obs.map((v, i) => k >= 0 ? mean + (v - mean) * k + (late[i] - v) * (1 - k) * 0.5 * k : mean - (v - mean) * (-k) * 0.8);
    const r2 = 1 - obs.reduce((a, v, i) => a + (v - fc[i]) ** 2, 0) / ssMean;
    $("r2-val").textContent = r2.toFixed(2);
    $("r2-say").textContent = r2 > 0.9 ? "Almost on top of the real curve. Our models reach this 5 minutes ahead." : r2 > 0.6 ? "Follows the rises and falls, with some misses." : r2 > 0.3 ? "Gets the shape of the flood but not its size, like our models 20 minutes ahead." : r2 > -0.05 ? "Barely better than always guessing the average depth." : "Worse than just guessing the average: the forecast is late or wrong-sized.";
    chart(cv, { n, tick: 12, label: i => "+" + i * 5 + "m", minMax: 10, series: [{ data: obs.map(() => mean), color: C.soft, dash: [4, 4], width: 1.4 }, { data: fc.map(v => Math.max(0, v)), color: C.rain, width: 2.2 }, { data: obs, color: C.ink, width: 2.6 }] }); };
  sl.oninput = upd; upd(); window.addEventListener("resize", upd);
}
async function init() {
  try {
    [INDEX, ROADS] = await Promise.all([getJSON("index.json"), getJSON("roads.json")]);
    try { WX = await getJSON("weather.json"); } catch (e) { WX = null; }
    GRAPH = buildGraph(ROADS); streetGauges(); initMap(); routeInfo(); buildNames();
    $("route-form").onsubmit = searchRoute; $("swap").onclick = () => { const f = $("q-from").value; $("q-from").value = $("q-to").value; $("q-to").value = f; };
    const sel = $("storm"), sorted = [...INDEX.storms].sort((a, b) => b.date.localeCompare(a.date));
    sel.innerHTML = sorted.map(s => `<option value="${s.date}">${fmtDate(s.date)}: ${s.gauge_mm} mm, deepest ${s.deepest_in} in</option>`).join("");
    const want = location.hash.slice(1), pick = INDEX.storms.some(s => s.date === want) ? want : (INDEX.storms.find(s => s.date === "2026-07-18") || sorted[0]).date;
    sel.value = pick; sel.onchange = () => { history.replaceState(null, "", "#" + sel.value); loadStorm(sel.value); };
    document.querySelectorAll(".modes button").forEach(b => b.onclick = () => setMode(b.dataset.mode));
    const models = (await getJSON(`storm_${pick}.json`)).models;
    $("model").innerHTML = LIVE_MODELS.filter(m => models.includes(m)).map(m => `<option value="${esc(m)}">${esc(nice(m))}</option>`).join(""); $("model").value = model;
    $("model").onchange = e => { model = e.target.value; lastSlot = -1; setTime(T); };
    $("ahead").onchange = e => { ahead = e.target.value; horizon = ahead; segSync(); lastSlot = -1; setTime(T); };
    $("time").oninput = e => { if (playing) play(); setTime(+e.target.value); }; $("play").onclick = play;
    const rpGo = t => { if (playing) play(); setTime(Math.max(0, Math.min(287, t))); };
    $("rp-time").oninput = e => rpGo(+e.target.value); $("rp-back").onclick = () => rpGo(Math.floor(T) - 1); $("rp-fwd").onclick = () => rpGo(Math.floor(T) + 1);
    $("rp-heavy").onclick = () => { const h = rpHeavy(STORM); if (h >= 0) rpGo(h); };
    $("sensor").onchange = e => selectSensor(e.target.value);
    $("h-seg").innerHTML = ["5", "20", "40"].map(h => `<button data-h="${h}">${h} min ahead</button>`).join("");
    const segSync = () => $("h-seg").querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", b.dataset.h === horizon));
    $("h-seg").querySelectorAll("button").forEach(b => b.onclick = () => { horizon = ahead = b.dataset.h; $("ahead").value = ahead; segSync(); lastSlot = -1; setTime(T); }); segSync();
    document.querySelectorAll("#ahead-seg button").forEach(b => b.onclick = () => { $("ahead").value = b.dataset.a; $("ahead").dispatchEvent(new Event("change")); });
    table("t-rec", ["Model", "5 min", "20 min", "40 min"], [["Persistence (baseline)", "0.93", "0.44", "-0.10"], ["LSTM, all at once", "0.94", "0.52", "0.11"], ["Recursive ANN-LSTM", "0.95", "0.49", "-0.25"], ["Recursive XGBoost-LSTM", "0.95", "0.52", "-0.02"], ["Recursive + perfect rain (upper bound)", "0.95", "0.56", "0.10"]]);
    r2Demo();
    $("model-toggles").innerHTML = models.map(m => `<label><input type="checkbox" data-m="${esc(m)}" ${modelsOn.has(m) ? "checked" : ""}><i style="background:${STYLE[m][0]}"></i>${esc(nice(m))}</label>`).join("");
    $("model-toggles").querySelectorAll("input").forEach(c => c.onchange = () => { c.checked ? modelsOn.add(c.dataset.m) : modelsOn.delete(c.dataset.m); drawDepth(); });
    const R = INDEX.results, keepD = new Set(["Persistence", "XGBoost (no future rain)", "LSTM (standard)", ...MINE, "LSTM + perfect rain forecast", "CNN-LSTM + perfect rain forecast"]);
    const dRows = R.depth.filter(r => keepD.has(r[0])).sort((a, b) => [...keepD].indexOf(a[0]) - [...keepD].indexOf(b[0]));
    table("t-depth", ["Model", "R² 5 min", "R² 20 min", "R² 40 min", "Caught early"], dRows.map(r => [nice(r[0]), r[1].toFixed(2), r[2].toFixed(2), r[3].toFixed(2), r[7] + "%"]), 2, false, dRows.map(r => r[0]));
    table("t-road", ["Model", "Top 1%", "Top 5%"], R.road.map(r => [r[0], r[1] + "%", r[2] + "%"]), 1);
    await setMode(want === "live" ? "live" : "past");                         // opens on a past storm; #live opens the live view
    setInterval(() => { if (MODE === "live") loadLive(); }, 60000);
    let rz = 0; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => setMode(MODE), 250); });
  } catch (e) {
    $("app").innerHTML = `<div class="err"><b>The dashboard data did not load.</b><br>Start it from the project folder with <code>python3 backend/app.py</code> and open http://localhost:8000. (${esc(e.message)})</div>`;
  }
}
init();

/* Model Lab: load data, run ANN-LSTM and/or CNN-LSTM (js/models.js) on every hour-long window, show the results. */
"use strict";
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const COL = { ann_lstm: "#ffb648", cnn_lstm: "#ff6a5c", persist: "#a9b8c2", depth: "#f2f6f6", rain: "rgba(143,208,255,.55)", soft: "rgba(232,240,241,.7)", line: "rgba(255,255,255,.1)" };
const NAME = { ann_lstm: "ANN-LSTM", cnn_lstm: "CNN-LSTM", persist: "Persistence (baseline)" };
const BODY = getComputedStyle(document.documentElement).getPropertyValue("--body");
const M = {}; let SAMPLES = [], choice = "ann_lstm", ahead = 3, RES = null, STEP = 11, PLAY = null;
const fmt = v => v == null || isNaN(v) ? "–" : v.toFixed(1) + " in";

/* ------------------------------------------------------------ chart: lines on the left axis, rain bars on the right */
function chart(cv, o) {
  const d = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  cv.width = W * d; cv.height = H * d; const g = cv.getContext("2d"); g.setTransform(d, 0, 0, d, 0, 0); g.clearRect(0, 0, W, H);
  const L = 44, R = o.bars ? 44 : 24, T = 18, B = 26, n = o.n;
  const mx = Math.max(1, ...o.lines.flatMap(s => s.data.filter(v => v != null)));
  const bm = o.bars ? Math.max(1, ...o.bars.filter(v => v != null)) : 1;
  const X = i => L + (W - L - R) * (n > 1 ? i / (n - 1) : 0.5), Y = v => T + (H - T - B) * (1 - v / mx), YB = v => T + (H - T - B) * (1 - v / bm);
  g.font = "11px " + BODY; g.lineWidth = 1;
  for (let k = 0; k <= 4; k++) { const y = T + (H - T - B) * k / 4; g.strokeStyle = COL.line; g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke();
    g.fillStyle = COL.soft; g.textAlign = "right"; g.fillText((mx * (1 - k / 4)).toFixed(mx <= 1 ? 2 : mx < 5 ? 1 : 0), L - 6, y + 4);
    if (o.bars) { g.textAlign = "left"; g.fillText((bm * (1 - k / 4)).toFixed(1), W - R + 6, y + 4); } }
  g.textAlign = "left"; g.fillText(o.unit ?? "in", 4, 10); if (o.bars) { g.textAlign = "right"; g.fillText("mm", W - 2, 12); }
  g.textAlign = "center";   // about one label per 52 px, kept inside the canvas
  const step = Math.max(1, Math.ceil(n / Math.max(2, Math.floor((W - L - R) / 52))));
  for (let i = 0; i < n; i += step) { const s = o.label(i), w = g.measureText(s).width; g.fillText(s, Math.min(Math.max(X(i), w / 2 + 1), W - w / 2 - 1), H - 8); }
  if (o.shade) { g.fillStyle = "rgba(255,255,255,.07)"; g.fillRect(X(o.shade[0]), T, X(o.shade[1]) - X(o.shade[0]), H - T - B);
    g.strokeStyle = "rgba(255,255,255,.35)"; g.strokeRect(X(o.shade[0]) + .5, T + .5, X(o.shade[1]) - X(o.shade[0]), H - T - B); }
  if (o.mark != null) { g.strokeStyle = "rgba(255,255,255,.45)"; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(X(o.mark), T); g.lineTo(X(o.mark), H - B); g.stroke(); g.setLineDash([]); }
  if (o.bars) { g.fillStyle = COL.rain; const bw = Math.max(1, (W - L - R) / n - 1); o.bars.forEach((v, i) => { if (v) g.fillRect(X(i) - bw / 2, YB(v), bw, YB(0) - YB(v)); }); }
  for (const s of o.lines) {
    g.strokeStyle = s.color; g.lineWidth = s.width || 2; g.setLineDash(s.dash || []); g.beginPath(); let pen = false;
    s.data.forEach((v, i) => { if (v == null) { pen = false; return; } pen ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v)); pen = true; });
    g.stroke(); g.setLineDash([]);
  }
}

/* grouped bars: groups along x, one bar per series */
function bars(cv, o) {
  const d = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  cv.width = W * d; cv.height = H * d; const g = cv.getContext("2d"); g.setTransform(d, 0, 0, d, 0, 0); g.clearRect(0, 0, W, H);
  const L = 44, R = 12, T = 10, B = 26, n = o.labels.length, k = o.series.length;
  const vals = o.series.flatMap(s => s.data).filter(v => v != null && isFinite(v)), mx = Math.max(o.min0 ? 0.1 : 1e-9, ...vals), mn = o.min0 ? 0 : Math.min(0, ...vals);
  const Y = v => T + (H - T - B) * (mx - v) / (mx - mn || 1), gw = (W - L - R) / n, bw = Math.min(26, gw * 0.8 / k);
  g.font = "11px " + BODY;
  for (let i = 0; i <= 4; i++) { const v = mn + (mx - mn) * i / 4, y = Y(v); g.strokeStyle = COL.line; g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke(); g.fillStyle = COL.soft; g.textAlign = "right"; g.fillText(v.toFixed(Math.abs(mx - mn) < 3 ? 2 : 1), L - 6, y + 4); }
  if (mn < 0) { g.strokeStyle = "rgba(255,255,255,.5)"; g.beginPath(); g.moveTo(L, Y(0)); g.lineTo(W - R, Y(0)); g.stroke(); }
  o.labels.forEach((lab, i) => {
    const cx = L + gw * (i + 0.5); g.fillStyle = COL.soft; g.textAlign = "center";
    const short = gw < g.measureText(lab).width + 8 ? lab.replace(" min", "") : lab;   // "5 min" becomes "5" when bars are narrow
    if (gw >= g.measureText(short).width + 4 || i % 2 === 0) g.fillText(short, cx, H - 8);
    o.series.forEach((s, j) => { const v = s.data[i]; if (v == null || !isFinite(v)) return; const x = cx - (k * bw) / 2 + j * bw, y0 = Y(Math.max(0, mn < 0 ? 0 : 0)), y1 = Y(v);
      g.fillStyle = s.color; g.fillRect(x + 1, Math.min(y0, y1), bw - 2, Math.max(1, Math.abs(y1 - y0))); });
  });
}
/* scatter of forecast (y) against measured (x), with the diagonal */
function scatter(cv, o) {
  const d = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  cv.width = W * d; cv.height = H * d; const g = cv.getContext("2d"); g.setTransform(d, 0, 0, d, 0, 0); g.clearRect(0, 0, W, H);
  const L = 44, R = 12, T = 22, B = 30, mx = Math.max(1, ...o.series.flatMap(s => s.pts.flat()));
  const X = v => L + (W - L - R) * v / mx, Y = v => T + (H - T - B) * (1 - v / mx);
  g.font = "11px " + BODY;
  for (let i = 0; i <= 4; i++) { const v = mx * i / 4; g.strokeStyle = COL.line; g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(W - R, Y(v)); g.stroke(); g.fillStyle = COL.soft; g.textAlign = "right"; g.fillText(v.toFixed(0), L - 6, Y(v) + 4); g.textAlign = "center"; g.fillText(v.toFixed(0), X(v), H - 14); }
  g.fillText("measured depth (in)", L + (W - L - R) / 2, H - 1); g.textAlign = "left"; g.fillText("forecast (in)", 4, 10);
  g.strokeStyle = "rgba(255,255,255,.55)"; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(X(0), Y(0)); g.lineTo(X(mx), Y(mx)); g.stroke(); g.setLineDash([]);
  for (const s of o.series) { g.fillStyle = s.color; g.globalAlpha = 0.75; for (const [a, p] of s.pts) { g.beginPath(); g.arc(X(a), Y(p), 3.2, 0, 7); g.fill(); } g.globalAlpha = 1; }
}
const keys = (el, items) => $(el).innerHTML = items.map(([c, l, dash]) => `<span><i style="background:${c}${dash ? ";opacity:.7" : ""}"></i>${esc(l)}</span>`).join("");

/* ------------------------------------------------------------ data */
/* Reads any reasonable CSV: comma, semicolon or tab; quotes; Excel BOM; ISO or day/month/year dates.
   If the file holds several sensors or storms (like the training files), it is split into one series each. */
function splitLine(l, sep) {
  if (!l.includes('"')) return l.split(sep);
  const out = []; let cur = "", q = false;
  for (let i = 0; i < l.length; i++) { const c = l[i];
    if (c === '"') { if (q && l[i + 1] === '"') { cur += '"'; i++; } else q = !q; }
    else if (c === sep && !q) { out.push(cur); cur = ""; } else cur += c; }
  out.push(cur); return out;
}
const ISO = /^(\d{4})[-\/](\d{1,2})[-\/](\d{1,2})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?/;
const DMY = /^(\d{1,2})[\/.-](\d{1,2})[\/.-](\d{2,4})[ T,]+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)?/i;
function toTime(s, dayFirst) {
  let m = s.match(ISO); if (m) return Date.UTC(+m[1], m[2] - 1, +m[3], +m[4], +m[5], +(m[6] || 0));
  m = s.match(DMY);
  if (m) { const a = +m[1], b = +m[2]; let y = +m[3]; if (y < 100) y += 2000; const [day, mon] = dayFirst ? [a, b] : [b, a];
    let h = +m[4]; if (m[7]) { h %= 12; if (/pm/i.test(m[7])) h += 12; } return Date.UTC(y, mon - 1, day, h, +m[5], +(m[6] || 0)); }
  return NaN;
}
const placeName = s => s.replace(/-[a-z0-9]{6}$/i, "").replace(/^[A-Z]{1,2}-/, "").split("-").map(w => w.length > 2 ? w[0].toUpperCase() + w.slice(1) : w).join(" ");
function readGroups(text) {
  text = text.replace(/^﻿/, "");
  const lines = text.split(/\r\n|\n|\r/).filter(l => l.trim());
  if (lines.length < 2) throw new Error("No rows found. The file needs a header line and at least 13 rows.");
  const sep = [",", ";", "\t"].reduce((best, s) => lines[0].split(s).length > lines[0].split(best).length ? s : best, ",");
  const head = splitLine(lines[0], sep).map(h => h.trim().replace(/^"|"$/g, "").toLowerCase());
  const find = (...k) => head.findIndex(h => k.some(x => h.startsWith(x)));
  const ti = find("time", "date", "timestamp"), ri = find("rain", "precip"), di = find("depth", "water"), gi = find("storm"), si = find("sensor", "site", "location");
  if (ti < 0 || ri < 0 || di < 0) throw new Error(`The first line must name the columns time_utc, rain_mm and depth_in. This file has: ${head.slice(0, 8).join(", ")}.`);
  const cell = (c, i) => (c[i] ?? "").trim().replace(/^"|"$/g, "");
  // day/month or month/day? decided once for the whole file
  let dayFirst = true, slashDates = false;
  for (const l of lines.slice(1, 4000)) { const m = cell(splitLine(l, sep), ti).match(DMY); if (!m) continue; slashDates = true;
    if (+m[1] > 12) { dayFirst = true; break; } if (+m[2] > 12) { dayFirst = false; break; } }
  const groups = new Map();
  for (let k = 1; k < lines.length; k++) {
    const c = splitLine(lines[k], sep), ts = cell(c, ti); if (!ts) continue;
    const t = toTime(ts, dayFirst);
    if (isNaN(t)) throw new Error(`Row ${k + 1}: the time “${esc(ts)}” is not a date. Use 2026-07-18 15:05 (or 18/07/2026 15:05).`);
    const rs = cell(c, ri).replace(",", "."), ds = cell(c, di).replace(",", "."), r = rs === "" || /^(na|nan|null)$/i.test(rs) ? null : +rs, dep = ds === "" ? 0 : +ds;
    if ((r != null && isNaN(r)) || isNaN(dep)) throw new Error(`Row ${k + 1}: rain and depth must be numbers (found “${esc(rs)}” and “${esc(ds)}”).`);
    const key = gi >= 0 ? cell(c, gi) : si >= 0 ? cell(c, si) : "all";
    if (!groups.has(key)) groups.set(key, { key, place: si >= 0 ? cell(c, si) : "", rows: [] });
    groups.get(key).rows.push({ t, rain: r == null ? null : Math.max(0, r), depth: Math.max(0, dep) });
  }
  // tidy every group: sort, drop exact duplicates, split at long breaks, fill short gaps, cut long records into events
  const out = []; let filled = 0;
  for (const g of groups.values()) {
    g.rows.sort((a, b) => a.t - b.t);
    const rows = g.rows.filter((r, i) => i === 0 || r.t !== g.rows[i - 1].t);
    if (rows.length < g.rows.length * 0.9 && gi < 0 && si < 0)
      throw new Error("Many rows share the same time, so the file probably mixes several places. Keep one place, or add a Sensor ID or storm_id column.");
    let seg = [];
    const flush = () => { if (seg.length >= 13) out.push(...events(seg).map(rs => ({ ...g, rows: rs }))); seg = []; };
    for (const r of rows) {
      const prev = seg[seg.length - 1];
      if (prev) { const gap = Math.round((r.t - prev.t) / 300000);
        if (gap > 6) flush();
        else for (let s = 1; s < gap; s++) { seg.push({ t: prev.t + s * 300000, rain: null, depth: prev.depth + (r.depth - prev.depth) * s / gap }); filled++; } }
      seg.push({ ...r });
    }
    flush();
  }
  if (!out.length) throw new Error("No stretch of at least 13 readings 5 minutes apart was found. The models need one hour of history plus one more reading.");
  out.forEach(s => { s.rows.forEach(r => r.t = new Date(r.t)); s.peak = Math.max(...s.rows.map(r => r.depth)); s.rain = s.rows.reduce((a, r) => a + (r.rain || 0), 0); });
  out.sort((a, b) => b.peak - a.peak || b.rain - a.rain);
  return { series: out, filled, dateNote: slashDates ? (dayFirst ? "Dates were read as day/month/year." : "Dates were read as month/day/year.") : "" };
}
// a long record (weeks of readings) is cut into rain or flood events with an hour either side; short ones are kept whole
function events(rows) {
  if (rows.length <= 600) return [rows];
  const wet = rows.map(r => (r.rain || 0) > 0 || r.depth > 0.1), keep = new Array(rows.length).fill(false);
  wet.forEach((w, i) => { if (w) for (let k = Math.max(0, i - 12); k <= Math.min(rows.length - 1, i + 12); k++) keep[k] = true; });
  const ev = []; let cur = [];
  rows.forEach((r, i) => { if (keep[i]) cur.push(r); else if (cur.length) { ev.push(cur); cur = []; } }); if (cur.length) ev.push(cur);
  return ev.filter(e => e.length >= 13).flatMap(e => { const parts = []; for (let i = 0; i < e.length; i += 576) parts.push(e.slice(Math.max(0, i - 11), i + 576)); return parts; }).filter(e => e.length >= 13);
}
const toCsv = rows => "time_utc,rain_mm,depth_in\n" + rows.map(r => `${r.t.toISOString().slice(0, 16).replace("T", " ")},${r.rain == null ? "" : +r.rain.toFixed(3)},${+r.depth.toFixed(3)}`).join("\n");
const label = s => `${s.place ? placeName(s.place) + ", " : ""}${s.rows[0].t.toISOString().slice(0, 10)}: up to ${s.peak.toFixed(1)} in, ${s.rain.toFixed(0)} mm rain, ${s.rows.length} readings`;

let LOADED = null, SHOWN = "";
function load(text, from) {
  const st = $("status"); let got;
  try { got = readGroups(text); } catch (e) { st.textContent = e.message; st.classList.add("bad"); $("results").hidden = true; $("pick-wrap").hidden = true; return false; }
  st.classList.remove("bad"); LOADED = got;
  const many = got.series.length > 1;
  $("pick-wrap").hidden = !many;
  if (many) {
    $("pick-head").textContent = `${from || "This data"} holds ${got.series.length.toLocaleString()} separate storms or places. Pick one (deepest first${got.series.length > 400 ? ", top 400 shown" : ""}):`;
    $("pick").innerHTML = got.series.slice(0, 400).map((s, i) => `<option value="${i}">${esc(label(s))}</option>`).join("");
  }
  pick(0);
  return true;
}
let READY = null;   // data loaded and waiting for the Run button
function pick(i) {
  const s = LOADED.series[i];
  $("csv").value = SHOWN = toCsv(s.rows);
  READY = { rows: s.rows, note: [LOADED.filled ? `${LOADED.filled} missing 5-minute readings were filled in.` : "", LOADED.dateNote].filter(Boolean).join(" ") };
  waiting(`Ready: ${s.rows.length} readings from ${s.rows[0].t.toISOString().slice(0, 16).replace("T", " ")} UTC, up to ${s.peak.toFixed(1)} in of water. Press “Run the forecast” to calculate.`);
}
// clear old results and point at the Run button
function waiting(msg) {
  if (PLAY) { clearInterval(PLAY); PLAY = null; $("play").textContent = "Play"; }
  RES = null; $("results").hidden = true; $("status").classList.remove("bad"); $("status").textContent = msg;
  $("run").classList.add("ready");
}

function run(rows, extra = "") {
  const st = $("status");
  const use = choice === "both" ? ["ann_lstm", "cnn_lstm"] : [choice];
  if (use.some(m => !M[m])) { st.textContent = "The model is still loading, try again in a moment."; return; }
  const n = rows.length, preds = {}, rainFc = {};
  for (const m of use) {
    preds[m] = new Array(n).fill(null);
    for (let i = 11; i < n; i++) {
      const win = rows.slice(i - 11, i + 1).map(r => [r.depth, r.rain ?? 0, r.rain == null ? 1 : 0]);
      const out = FloodModels.predict(M[m], win, rows[i].t);
      preds[m][i] = out.depth; if (out.rain) rainFc[i] = out.rain;
    }
  }
  const pk = rows.reduce((b, r, i) => r.depth > rows[b].depth ? i : b, 0);
  RES = { rows, preds, rainFc, use };
  const rise = rows.findIndex((r, i) => i >= 11 && r.depth >= 1);   // open at the moment the water starts rising
  STEP = Math.min(n - 1, Math.max(11, rise > 0 ? rise : pk - 4));
  st.textContent = `Ran ${use.map(m => NAME[m]).join(" and ")} on ${n - 11} hour-long windows. ${extra}`.trim();
  show(true);
}
const runFromBox = () => {
  const v = $("csv").value;
  if (!(v.trim() && v === SHOWN && READY)) { if (!load(v, "The pasted data")) return; }   // pasted or edited by hand: read it first
  $("run").classList.remove("ready"); clearTimeout(toastTimer); $("toast").classList.remove("show"); $("toast").hidden = true;
  const btn = $("run"); btn.disabled = true; btn.textContent = "Running…";
  setTimeout(() => { try { run(READY.rows, READY.note); } finally { btn.disabled = false; btn.textContent = "Run the forecast"; } }, 30);
};

/* ------------------------------------------------------------ results */
const hhmm = t => t.toISOString().slice(11, 16);
function show(scroll) {
  const { rows, preds, rainFc, use } = RES, n = rows.length, last = n - 1;
  $("results").hidden = false;
  const peak = Math.max(...rows.map(r => r.depth));
  $("res-lede").textContent = `${n} readings from ${rows[0].t.toISOString().slice(0, 16).replace("T", " ")} to ${hhmm(rows[last].t)} UTC; the deepest water measured was ${peak.toFixed(1)} inches. The boxes below show the moment chosen in “Watch the model make one forecast”; move its slider to change them.`;
  // over time: forecast for row t was made at row t - (ahead + 1)
  const series = use.map(m => ({ data: rows.map((_, t) => { const s = t - (ahead + 1); return s >= 11 && preds[m][s] ? preds[m][s][ahead] : null; }), color: COL[m], width: 2.2 }));
  drawTime = () => chart($("c-time"), { n, label: i => hhmm(rows[i].t), bars: rows.map(r => r.rain), shade: [STEP - 11, Math.min(n - 1, STEP + 8)], mark: STEP, lines: [{ data: rows.map(r => r.depth), color: COL.depth, width: 2.6 }, ...series] });
  drawTime();
  keys("k-time", [[COL.depth, "Measured depth"], ...use.map(m => [COL[m], `${NAME[m]}, forecast ${5 * (ahead + 1)} min ahead`]), [COL.rain, "Rain (mm per 5 min, right scale)"]]);
  $("step").max = n - 1; $("step").min = 11; $("step").value = STEP; showStep();
  // accuracy on this data
  const acc = [];
  for (const m of use) for (const h of [0, 3, 7]) {
    const a = [], p = [];
    for (let s = 11; s + h + 1 < n; s++) { a.push(rows[s + h + 1].depth); p.push(preds[m][s][h]); }
    if (!a.length) continue;
    const mean = a.reduce((x, y) => x + y, 0) / a.length, se = a.reduce((x, v, i) => x + (v - p[i]) ** 2, 0), st = a.reduce((x, v) => x + (v - mean) ** 2, 0);
    acc.push([NAME[m], `${5 * (h + 1)} min`, Math.sqrt(se / a.length).toFixed(2), st > 1e-9 ? (1 - se / st).toFixed(2) : "–", a.length]);
  }
  $("t-acc").innerHTML = "<tr><th>Model</th><th>Ahead</th><th class='n'>RMSE (in)</th><th class='n'>R²</th><th class='n'>Checks</th></tr>" +
    acc.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td class="n">${r[2]}</td><td class="n">${r[3]}</td><td class="n">${r[4]}</td></tr>`).join("") || "<tr><td>Not enough rows after the first hour to check.</td></tr>";
  // compare the models at every forecast time, with persistence as the baseline
  const who = [...use, "persist"], H8 = [...Array(8).keys()], stats = {};
  for (const m of who) stats[m] = H8.map(h => {
    const a = [], p = [];
    for (let s = 11; s + h + 1 < n; s++) { a.push(rows[s + h + 1].depth); p.push(m === "persist" ? rows[s].depth : preds[m][s][h]); }
    if (!a.length) return { rmse: null, r2: null, pts: [] };
    const mean = a.reduce((x, y) => x + y, 0) / a.length, se = a.reduce((x, v, i) => x + (v - p[i]) ** 2, 0), st = a.reduce((x, v) => x + (v - mean) ** 2, 0);
    return { rmse: Math.sqrt(se / a.length), r2: st > 1e-9 ? 1 - se / st : null, pts: a.map((v, i) => [v, p[i]]) };
  });
  const labs = H8.map(h => `${5 * (h + 1)} min`);
  bars($("c-rmse"), { labels: labs, min0: true, series: who.map(m => ({ data: stats[m].map(s => s.rmse), color: COL[m] })) });
  keys("k-rmse", who.map(m => [COL[m], NAME[m]]));
  chart($("c-r2"), { n: 8, unit: "R²", label: i => labs[i], lines: who.map(m => ({ data: stats[m].map(s => s.r2 == null ? null : Math.max(0, s.r2)), color: COL[m], width: m === "persist" ? 1.6 : 2.4, dash: m === "persist" ? [5, 4] : [] })) });
  keys("k-r2", [...who.map(m => [COL[m], NAME[m]]), ["transparent", "R² below 0 is drawn at 0"]]);
  scatter($("c-scatter"), { series: who.map(m => ({ pts: stats[m][ahead].pts, color: COL[m] })) });
  keys("k-scatter", who.map(m => [COL[m], NAME[m]]));
  $("sc-title").textContent = `Forecast ${5 * (ahead + 1)} minutes ahead against what really happened`;
  $("t-win").innerHTML = "<tr><th>Ahead</th><th>Best on this data</th><th class='n'>Its RMSE</th><th class='n'>Persistence RMSE</th></tr>" + H8.filter(h => stats.persist[h].rmse != null).map(h => {
    const best = who.reduce((b, m) => stats[m][h].rmse < stats[b][h].rmse ? m : b, who[0]);
    return `<tr class="${best === "persist" ? "" : "best"}"><td>${labs[h]}</td><td>${NAME[best]}</td><td class="n">${stats[best][h].rmse.toFixed(2)}</td><td class="n">${stats.persist[h].rmse.toFixed(2)}</td></tr>`; }).join("");

  // every row
  const head = ["Time (UTC)", "Rain (mm)", "Depth (in)", ...use.flatMap(m => [`${NAME[m]} +20 min`, `${NAME[m]} +40 min`])];
  $("t-rows").innerHTML = "<tr>" + head.map((h, i) => `<th class="${i ? "n" : ""}">${h}</th>`).join("") + "</tr>" +
    rows.map((r, i) => `<tr class="${r.depth > 0.5 ? "wet" : ""}"><td>${r.t.toISOString().slice(0, 16).replace("T", " ")}</td><td class="n">${r.rain == null ? "–" : r.rain}</td><td class="n">${r.depth.toFixed(2)}</td>` +
      use.map(m => preds[m][i] ? `<td class="n">${preds[m][i][3].toFixed(2)}</td><td class="n">${preds[m][i][7].toFixed(2)}</td>` : `<td class="n">–</td><td class="n">–</td>`).join("") + "</tr>").join("");
  if (window.Sky) Sky.set({ rain: Math.max(0, ...rows.slice(-3).map(r => (r.rain || 0) * 12)) / 4, storm: Math.min(1, 0.3 + peak / 15), thunder: 0 });
  if (scroll) document.getElementById("results").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
}

/* one forecast, opened up: what the model reads, what it outputs, what really happened */
let drawTime = () => {};
function showStep() {
  if (!RES) return;
  const { rows, preds, rainFc, use } = RES, n = rows.length, s = STEP, past = rows.slice(s - 11, s + 1), fut = rows.slice(s + 1, s + 9);
  $("step-time").textContent = `${hhmm(rows[s].t)} UTC`;
  drawTime();
  const known = fut.length;
  chart($("c-next"), { n: 20, mark: 11, label: i => i < 11 ? `−${(11 - i) * 5}` : i === 11 ? "now" : `+${(i - 11) * 5}`,
    bars: [...past.map(r => r.rain), ...fut.map(r => r.rain), ...new Array(8 - known).fill(null)],
    lines: [{ data: [...past.map(r => r.depth), ...new Array(8).fill(null)], color: COL.depth, width: 2.6 },
            { data: [...new Array(11).fill(null), rows[s].depth, ...fut.map(r => r.depth), ...new Array(8 - known).fill(null)], color: COL.depth, width: 1.6, dash: [2, 3] },
            ...use.map(m => ({ data: [...new Array(11).fill(null), rows[s].depth, ...preds[m][s]], color: COL[m], width: 2.4, dash: [6, 4] }))] });
  keys("k-next", [[COL.depth, "What the model read (last hour)"], [COL.depth, "What really happened next", 1], ...use.map(m => [COL[m], `${NAME[m]} forecast`, 1]), [COL.rain, "Rain that fell (mm)"]]);
  // the four boxes follow the chosen moment, with what really happened next where it is known
  const m0 = use[0], f = preds[m0][s], at = t => `${hhmm(t)} UTC`, real = k => rows[s + k] ? ` (really ${rows[s + k].depth.toFixed(1)} in)` : "";
  const fc = (h, k) => [fmt(f[h]), `${NAME[m0]} forecast for ${at(new Date(rows[s].t.getTime() + k * 300000))}${real(k)}`];
  $("res-facts").innerHTML = [[fmt(rows[s].depth), `depth measured at ${at(rows[s].t)}`], fc(3, 4), fc(7, 8),
    rainFc[s] ? [rainFc[s].reduce((a, b) => a + b, 0).toFixed(1) + " mm", `rain the ANN expected in the next 40 minutes${fut.length === 8 ? ` (really ${fut.reduce((a, r) => a + (r.rain || 0), 0).toFixed(1)} mm)` : ""}`]
              : [fmt(Math.max(...rows.map(r => r.depth))), "deepest water measured in this storm"]]
    .map(([v, l]) => `<div class="fact"><b>${v}</b><span>${esc(l)}</span></div>`).join("");
  // the numbers going in and coming out
  const cells = (arr, f) => arr.map(v => `<td class="n">${f(v)}</td>`).join("");
  $("t-in").innerHTML = `<tr><th></th>${past.map(r => `<th class="n">${hhmm(r.t)}</th>`).join("")}</tr>
    <tr><td>Depth (in)</td>${cells(past, r => r.depth.toFixed(1))}</tr>
    <tr><td>Rain (mm)</td>${cells(past, r => r.rain == null ? "–" : r.rain.toFixed(1))}</tr>
    <tr><td>Rain missing</td>${cells(past, r => r.rain == null ? 1 : 0)}</tr>`;
  const ann = use.includes("ann_lstm") && rainFc[s];
  $("t-out").innerHTML = `<tr><th></th>${[...Array(8)].map((_, h) => `<th class="n">+${5 * (h + 1)}</th>`).join("")}</tr>` +
    (ann ? `<tr class="sub"><td>ANN rain forecast (mm)</td>${cells(rainFc[s], v => v.toFixed(1))}</tr>
            <tr class="sub"><td>Rain that fell (mm)</td>${cells([...Array(8)].map((_, h) => fut[h]), r => r ? (r.rain == null ? "–" : r.rain.toFixed(1)) : "?")}</tr>` : "") +
    use.map(m => `<tr><td style="color:${COL[m]}">${NAME[m]} depth (in)</td>${cells(preds[m][s], v => v.toFixed(1))}</tr>`).join("") +
    `<tr class="real"><td>Really happened (in)</td>${cells([...Array(8)].map((_, h) => fut[h]), r => r ? r.depth.toFixed(1) : "?")}</tr>`;
  // one plain sentence
  const f20 = preds[m0][s][3], r20 = rows[s + 4];
  const rainHr = past.reduce((a, r) => a + (r.rain || 0), 0);
  $("step-say").innerHTML = `At <b>${hhmm(rows[s].t)}</b> the water was <b>${rows[s].depth.toFixed(1)} in</b> and <b>${rainHr.toFixed(1)} mm</b> of rain had fallen in the hour before. ` +
    (ann ? `The ANN expected <b>${rainFc[s].reduce((a, b) => a + b, 0).toFixed(1)} mm</b> more in the next 40 minutes${fut.length === 8 ? ` (<b>${fut.reduce((a, r) => a + (r.rain || 0), 0).toFixed(1)} mm</b> really fell)` : ""}. ` : "") +
    `${NAME[m0]} forecast <b>${f20.toFixed(1)} in</b> for 20 minutes later` + (r20 ? `; it was really <b>${r20.depth.toFixed(1)} in</b>.` : "; that time is after your last reading, so it can't be checked yet.");
}

function info() {
  const one = m => M[m] ? `<p><b>${NAME[m]}</b>. ${m === "ann_lstm"
      ? "A small neural network (ANN) first forecasts the next 40 minutes of rain from the last hour; the LSTM then reads the hour of depth and rain, adds that forecast, and predicts the depth."
      : "Convolution filters scan the hour for short patterns, such as a burst of rain or water starting to rise; the LSTM reads those patterns and predicts the depth. It does not forecast rain."}</p>
      <div class="flowline">${m === "ann_lstm" ? "<span>last hour</span><i>→</i><span>ANN rain forecast</span><i>→</i><span>LSTM</span><i>→</i><span>depth +5…+40 min</span>" : "<span>last hour</span><i>→</i><span>CNN filters</span><i>→</i><span>LSTM</span><i>→</i><span>depth +5…+40 min</span>"}</div>
      <p>Test R² (Jul–Oct 2026): ${M[m].test_r2[0].toFixed(2)} at 5 min, ${M[m].test_r2[3].toFixed(2)} at 20 min, ${M[m].test_r2[7].toFixed(2)} at 40 min.</p>` : "<p>Loading the model…</p>";
  $("model-info").innerHTML = (choice === "both" ? ["ann_lstm", "cnn_lstm"] : [choice]).map(one).join("");
}

/* ------------------------------------------------------------ wiring */
const download = (name, text) => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); };
async function init() {
  const get = async p => { const r = await fetch(p, { cache: "no-store" }); if (!r.ok) throw new Error(p + " " + r.status); return r.json(); };
  try {
    SAMPLES = await get("api/lab_samples");
    $("sample").innerHTML += SAMPLES.map((s, i) => `<option value="${i}">${esc(s.name.replace(/^[A-Z]{1,2} - /, ""))}: storm ${esc(s.id.slice(-10))}, up to ${s.peak_in} in</option>`).join("");
  } catch (e) { $("status").textContent = "Sample storms are not in the data store yet: run python3 backend/build_store.py."; }
  $("sample").onchange = e => { if (e.target.value !== "") { fileLabel(""); load(SAMPLES[+e.target.value].csv, "This storm"); } };
  const openFile = async f => {
    if (!f) return;
    if (f.size > 300e6) { $("status").textContent = "That file is over 300 MB. Split it into smaller files first."; return; }
    fileLabel(`Reading ${f.name}…`); $("status").textContent = "Reading the file…"; $("sample").value = "";
    await new Promise(r => setTimeout(r, 30));
    try { const ok = load(await f.text(), f.name); fileLabel(ok ? `${f.name} loaded. Choose another file` : `${f.name} could not be read. Try another file`);
      if (ok) { const k = LOADED.series.length, rows = LOADED.series.reduce((a, s) => a + s.rows.length, 0);
        toast(`CSV received: ${f.name}`, `${rows.toLocaleString()} readings${k > 1 ? ` in ${k.toLocaleString()} storms` : ""} (${f.size < 1e5 ? Math.max(1, Math.round(f.size / 1e3)) + " KB" : (f.size / 1e6).toFixed(1) + " MB"}). Press “Run the forecast” to calculate.`); }
      else toast(`Couldn't read ${f.name}`, $("status").textContent, true); }
    catch (e) { $("status").textContent = "The browser could not open that file: " + e.message; fileLabel(""); }
  };
  $("file").onchange = e => { openFile(e.target.files[0]); e.target.value = ""; };
  const drop = $("drop");
  ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", e => openFile(e.dataTransfer.files[0]));
  $("pick").onchange = e => pick(+e.target.value);
  $("run").onclick = runFromBox;
  $("step").oninput = e => { STEP = +e.target.value; showStep(); };
  $("play").onclick = () => {
    if (PLAY) { clearInterval(PLAY); PLAY = null; $("play").textContent = "Play"; return; }
    if (!RES) return; if (STEP >= RES.rows.length - 1) STEP = 11;
    $("play").textContent = "Pause";
    PLAY = setInterval(() => { if (STEP >= RES.rows.length - 1) { $("play").click(); return; } STEP++; $("step").value = STEP; showStep(); }, 450);
  };
  $("template").onclick = e => { e.preventDefault(); download("model_lab_template.csv", "time_utc,rain_mm,depth_in\n2026-07-18 15:00,0,0\n2026-07-18 15:05,0.5,0\n"); };
  $("download").onclick = e => { e.preventDefault(); if (!RES) return; const t = $("t-rows"); download("model_lab_results.csv", [...t.rows].map(r => [...r.cells].map(c => c.textContent).join(",")).join("\n")); };
  $("model-seg").querySelectorAll("button").forEach(b => b.onclick = () => { choice = b.dataset.m; $("model-seg").querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x === b)); info(); if (RES) waiting("Model changed. Press “Run the forecast” to calculate again."); });
  $("ahead-seg").querySelectorAll("button").forEach(b => b.onclick = () => { ahead = +b.dataset.h; $("ahead-seg").querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x === b)); if (RES) show(false); });
  info();
  for (const m of ["ann_lstm", "cnn_lstm"]) get("api/model/" + m).then(j => { M[m] = j; info(); }).catch(() => { $("status").textContent = "The models are not in the data store yet: run python3 backend/build_store.py."; });
  if (window.Sky) Sky.set({ storm: 0.45, rain: 0, thunder: 0 });
  let rz; addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => RES && show(false), 250); });
}
let toastTimer;
function toast(title, text, bad) {
  const el = $("toast"); el.querySelector("b").textContent = title; el.querySelector("span").textContent = text;
  el.classList.toggle("bad", !!bad); el.hidden = false; requestAnimationFrame(() => el.classList.add("show"));
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { el.classList.remove("show"); setTimeout(() => el.hidden = true, 300); }, bad ? 7000 : 4500);
}
const fileLabel = t => $("file-label").textContent = t || "Upload a CSV file, or drop it here";
init();

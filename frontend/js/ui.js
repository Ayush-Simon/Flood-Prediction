/* NYC Flood Watch: presentation layer.
   Draws the hero (hour strip, rain wave, weather card, borough cards), keeps the side rail in step with
   the page, and tells the sky (sky.js) what the weather is. It only reads data that app.js has loaded;
   it does not change any forecast. */
"use strict";

const IC = {
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4"/>',
  cloud: '<path d="M7 18a4.5 4.5 0 1 1 1.3-8.8A6 6 0 0 1 19.5 11.5 3.5 3.5 0 0 1 18 18z"/>',
  rain: '<path d="M7 15a4.5 4.5 0 1 1 1.3-8.8A6 6 0 0 1 19.5 8.5 3.5 3.5 0 0 1 18 15z"/><path d="M9 18l-1 2.5M13 18l-1 2.5M17 18l-1 2.5"/>',
  storm: '<path d="M7 14a4.5 4.5 0 1 1 1.3-8.8A6 6 0 0 1 19.5 7.5 3.5 3.5 0 0 1 18 14z"/><path d="M12.5 13l-2.5 4h3l-2 4"/>',
  pin: '<path d="M12 21s-6.5-6-6.5-11a6.5 6.5 0 0 1 13 0c0 5-6.5 11-6.5 11z"/><circle cx="12" cy="10" r="2.3"/>',
  wind: '<path d="M3 8h10a2.5 2.5 0 1 0-2.5-2.5M3 12h15a2.5 2.5 0 1 1-2.5 2.5M3 16h8"/>',
  drop: '<path d="M12 3s-6 6.6-6 11a6 6 0 0 0 12 0c0-4.4-6-11-6-11z"/>',
  gauge: '<path d="M4 18a8 8 0 1 1 16 0"/><path d="M12 18l4-5"/>',
};
const icon = k => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${IC[k]}</svg>`;
const kindOf = (mm, prob, thunder) => thunder || mm >= 6 ? "storm" : mm >= 0.3 ? "rain" : prob >= 40 || mm > 0 ? "cloud" : "sun";

/* the rain wave under the hour strip: a smooth line through the hourly rain, with a glow on the wettest hour */
let waveAnim = 0;
function drawWave(vals, hi) {
  const sig = (vals || []).join() + "|" + hi;
  if (sig === drawWave.sig || matchMedia("(prefers-reduced-motion: reduce)").matches) return drawWaveAt(vals, hi, 1);
  drawWave.sig = sig; const t0 = performance.now(), id = ++waveAnim;
  const step = now => { if (id !== waveAnim) return; const k = Math.min(1, (now - t0) / 900); drawWaveAt(vals, hi, 1 - Math.pow(1 - k, 3)); if (k < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}
function drawWaveAt(vals, hi, prog) {
  const cv = $("wave"), d = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  cv.width = W * d; cv.height = H * d; const g = cv.getContext("2d"); g.setTransform(d, 0, 0, d, 0, 0); g.clearRect(0, 0, W, H);
  if (!vals || !vals.length) return;
  const mx = Math.max(2, ...vals), n = vals.length, X = i => (i + 0.5) / n * W, Y = v => H * 0.72 - v / mx * H * 0.58;
  const pts = vals.map((v, i) => [X(i), Y(v)]);
  const path = () => { g.beginPath(); g.moveTo(0, pts[0][1]); g.lineTo(...pts[0]);
    for (let i = 0; i < n - 1; i++) { const [x0, y0] = pts[i], [x1, y1] = pts[i + 1], cx = (x0 + x1) / 2; g.bezierCurveTo(cx, y0, cx, y1, x1, y1); }
    g.lineTo(W, pts[n - 1][1]); };
  g.save(); g.beginPath(); g.rect(0, 0, W * prog, H); g.clip();
  path(); g.lineTo(W, H); g.lineTo(0, H); g.closePath();
  const fill = g.createLinearGradient(0, 0, 0, H); fill.addColorStop(0, "rgba(255,255,255,.22)"); fill.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = fill; g.fill();
  path(); g.strokeStyle = "rgba(255,255,255,.85)"; g.lineWidth = 2; g.shadowColor = "rgba(255,255,255,.6)"; g.shadowBlur = 8; g.stroke(); g.shadowBlur = 0;
  if (hi != null && hi >= 0) {
    const [x, y] = pts[Math.min(n - 1, hi)];
    const beam = g.createLinearGradient(0, 0, 0, H); beam.addColorStop(0, "rgba(255,255,255,0)"); beam.addColorStop(0.5, "rgba(255,255,255,.7)"); beam.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = beam; g.fillRect(x - 0.75, 0, 1.5, H);
    const glow = g.createRadialGradient(x, y, 0, x, y, 22); glow.addColorStop(0, "rgba(255,255,255,.95)"); glow.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = glow; g.globalAlpha = prog < 1 ? Math.max(0, (prog * W - x) / 60 + 1) * prog : 1; g.beginPath(); g.arc(x, y, 22, 0, 7); g.fill(); g.globalAlpha = 1;
  }
  g.restore();
}

function hourStrip(cells, labels, peak, past) {
  document.querySelector(".hours").classList.toggle("past", !!past);
  $("hours").innerHTML = cells.map((c, i) => `<div class="${i === peak ? "peak" : ""}">${c}</div>`).join("");
  $("hours-labels").innerHTML = labels.map((l, i) => `<span class="${i === peak ? "peak" : ""}">${esc(l)}</span>`).join("");
}

function boroCards(boro, dry) {
  const rows = Object.entries(boro || {}).map(([b, c]) => ({ b, c: dry ? [0, 0, 0] : c }));
  rows.sort((a, b) => (b.c[0] * 10 + b.c[1]) - (a.c[0] * 10 + a.c[1]));
  const pick = rows.some(r => r.c[0] + r.c[1]) ? rows.slice(0, 3) : rows.filter(r => ["Manhattan", "Brooklyn", "Queens"].includes(r.b));
  $("boro-cards").innerHTML = pick.map(({ b, c }) => {
    const k = c[0] >= 50 ? "storm" : c[0] > 0 ? "rain" : c[1] + c[2] > 0 ? "cloud" : "sun";
    const st = c[0] ? `${c[1].toLocaleString()} risky, ${c[2].toLocaleString()} to watch` : c[1] + c[2] ? `${(c[1] + c[2]).toLocaleString()} streets to watch` : "No streets flagged";
    return `<div class="glass boro ${c[0] ? "hot" : ""}"><div><small>New York City</small><b>${esc(b)}</b><div class="st">${st}</div></div>
      <div class="n" title="streets at high flood risk">${c[0].toLocaleString()}${icon(k)}</div></div>`;
  }).join("");
}

function heroLive(L) {
  const w = L.weather, n = L.next24, dry = L.what_if, thunder = /thunder/i.test(w.desc || "");
  $("hero-pill").textContent = "Live flood forecast";
  $("place-time").textContent = "New York City · " + L.now_local;
  // hour strip: the next 24 hours in 3-hour steps
  const cells = [], labels = []; let peak = -1, best = 0.29;
  for (let i = 0; i < 8; i++) {
    const k = i * 3, mm = n.rain.slice(k, k + 3).reduce((a, b) => a + b, 0), pr = Math.max(...n.prob.slice(k, k + 3));
    if (mm > best) { best = mm; peak = i; }
    cells.push(`${Math.round(n.temp[k])}°${icon(kindOf(mm, pr, thunder && i === 0))}`); labels.push(i === 0 ? "Now" : n.time[k]);
  }
  hourStrip(cells, labels, peak < 0 ? 0 : peak);
  const hi = n.rain.indexOf(Math.max(...n.rain));
  drawWave(n.rain, Math.max(...n.rain) > 0 ? hi : 0);
  // weather card
  const rainNow = w.rain_now || 0;
  $("now-card").innerHTML = `<div class="where">${icon("pin")}New York City</div>
    <div class="big">${w.temp == null ? "–" : Math.round(w.temp) + "°C"}</div><div class="desc">${esc(w.desc || "Weather unavailable")}${w.feels == null ? "" : ", feels like " + Math.round(w.feels) + "°"}</div>
    <div class="row"><span>${icon("wind")}${w.wind == null ? "–" : Math.round(w.wind) + " km/h"}</span><span>${icon("drop")}${w.humidity == null ? "–" : w.humidity + "%"}</span><span>${icon("rain")}${rainNow} mm</span></div>`;
  boroCards(L.boro, dry);
  // sky mood: cloud cover and the coming rain set how dark; rain falling now sets the rain
  const last15 = (L.rain15 || []).filter(r => r[2]).slice(-1)[0];
  if (window.Sky) Sky.set({ storm: 0.15 + 0.35 * (w.cloud || 0) / 100 + 0.5 * Math.min(1, (L.peak_1h_mm || 0) / 8),
    rain: Math.max(rainNow * 4, last15 ? last15[1] * 4 : 0), thunder: thunder ? 1 : (L.peak_1h_mm || 0) >= 12 ? 0.6 : 0 });
}

function heroPast(st, meta) {
  $("hero-pill").textContent = "Storm replay";
  $("place-time").textContent = "New York City · " + fmtDate(st.date);
  const h = st.hourly || [], cells = [], labels = []; let peak = 0, best = -1;
  for (let i = 0; i < 8; i++) {
    const mm = h.slice(i * 3, i * 3 + 3).reduce((a, b) => a + b, 0);
    if (mm > best) { best = mm; peak = i; }
    cells.push(`${mm < 0.05 ? 0 : mm < 10 ? mm.toFixed(1) : Math.round(mm)}<small>mm</small>${icon(kindOf(mm, 0, mm >= 10))}`); labels.push(hhmm(i * 180));
  }
  hourStrip(cells, labels, peak, true);
  drawWave(h, h.indexOf(Math.max(...h)));
  const deep = Math.max(0, ...st.sensors.map(s => s.peak || 0)), wet = st.sensors.filter(s => s.peak > 0).length;
  $("now-card").innerHTML = `<div class="where">${icon("pin")}${esc(fmtDate(st.date))}</div>
    <div class="big">${meta.gauge_mm}<span style="font-size:.45em; margin-left:6px">mm</span></div><div class="desc">rain at the gauges, deepest water ${deep} in</div>
    <div class="row"><span>${icon("rain")}${meta.peak5} mm in 5 min</span><span>${icon("gauge")}${wet} sensors wet</span>${st.has_reports ? `<span>${icon("drop")}${st.reported.length} reports</span>` : ""}</div>`;
  boroCards(st.boro, false);
  if (window.Sky) Sky.set({ storm: 0.35 + 0.65 * Math.min(1, meta.gauge_mm / 30), rain: 0, thunder: meta.peak5 >= 4 ? 0.8 : 0 });
}

/* while a past storm plays, the sky rains as hard as the gauges did at that moment */
function skyAt(t) {
  if (MODE !== "past" || !STORM || !STORM.gauges || !window.Sky) return;
  const s = Math.floor(t), v = Object.values(STORM.gauges).map(a => a[s] || 0);
  Sky.set({ rain: v.length ? v.reduce((a, b) => a + b, 0) / v.length * 12 / 2 : 0 });
}

/* side rail follows the section on screen; sections ease in once */
(() => {
  const links = [...document.querySelectorAll(".rail a")];
  const spy = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    links.forEach(a => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id));
  }), { rootMargin: "-45% 0px -50% 0px" });
  document.querySelectorAll("main section[id]").forEach(s => spy.observe(s));
  const rv = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); rv.unobserve(e.target); } }), { rootMargin: "0px 0px -8% 0px" });
  document.querySelectorAll("main section:not(.hero)").forEach(s => { s.classList.add("reveal"); rv.observe(s); });
  addEventListener("resize", () => { clearTimeout(skyAt.r); skyAt.r = setTimeout(() => { if (MODE === "live" && LIVE) drawWave(LIVE.next24.rain, LIVE.next24.rain.indexOf(Math.max(...LIVE.next24.rain))); else if (STORM && STORM.hourly) drawWave(STORM.hourly, STORM.hourly.indexOf(Math.max(...STORM.hourly))); }, 250); });
})();

/* smooth switching between Live and Past storms, and between storm days:
   the page content fades out, the new data loads, and it fades back in. Nothing about the data changes. */
(() => {
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches, wait = ms => new Promise(r => setTimeout(r, ms));
  let busy = false;
  async function fade(work) {
    if (still || busy) return work();
    busy = true; $("app").classList.add("fading"); await wait(200);
    try { return await work(); }
    finally { requestAnimationFrame(() => { $("app").classList.remove("fading"); busy = false; }); }
  }
  const baseSetMode = setMode, baseLoadStorm = loadStorm;
  setMode = m => (m !== MODE && document.body.dataset.ready) ? fade(() => baseSetMode(m)) : baseSetMode(m).then(r => { document.body.dataset.ready = "1"; return r; });
  loadStorm = date => (MODE === "past" && PAST && PAST.date !== date && !busy) ? fade(() => baseLoadStorm(date)) : baseLoadStorm(date);
})();

/* NYC Flood Watch: the animated sky behind the page.
   Purely visual. It reads the weather the page already shows and sets the mood:
   darker clouds as a storm builds, rain that matches the rain rate, lightning in thunderstorms,
   the Manhattan skyline along the bottom. Drawn on one fixed canvas; pauses when the tab is hidden
   and stays still for people who prefer reduced motion. */
"use strict";
(() => {
  const cv = document.getElementById("sky"); if (!cv) return;
  const g = cv.getContext("2d"), still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let W = 0, H = 0, dpr = 1, clouds = [], skyline = null, drops = [], t0 = performance.now(), last = 0, flash = 0, bolt = null, nextBolt = 4000;
  // current mood (eased towards the target so changes glide)
  const mood = { storm: 0.55, rain: 0, thunder: 0 }, target = { storm: 0.55, rain: 0, thunder: 0 };

  function rnd(seed) { return () => (seed = (seed * 16807) % 2147483647) / 2147483647; }

  function buildClouds() {
    // two big pre-drawn cloud sheets, drifted at different speeds for depth
    clouds = [0, 1].map(layer => {
      const c = document.createElement("canvas"), w = Math.ceil(W * 1.6), h = Math.ceil(H * (layer ? 0.75 : 0.62));
      c.width = w; c.height = h; const x = c.getContext("2d"), r = rnd(layer ? 91 : 37), n = layer ? 70 : 90;
      for (let i = 0; i < n; i++) {
        const cx = r() * w, cy = h * (0.12 + Math.pow(r(), 0.8) * 0.78), rad = (layer ? 120 : 90) + r() * (layer ? 260 : 200);
        const gr = x.createRadialGradient(cx, cy - rad * 0.25, rad * 0.05, cx, cy, rad);
        const tone = layer ? 0 : 1;
        gr.addColorStop(0, tone ? "rgba(214,226,224,0.55)" : "rgba(150,170,172,0.50)");
        gr.addColorStop(0.45, tone ? "rgba(120,140,145,0.32)" : "rgba(70,88,94,0.42)");
        gr.addColorStop(1, "rgba(30,40,46,0)");
        x.fillStyle = gr;
        for (const dx of [-w, 0, w]) { x.save(); x.translate(dx, 0); x.beginPath(); x.ellipse(cx, cy, rad * 1.5, rad, 0, 0, 7); x.fill(); x.restore(); }  // wraps, so the sheet tiles without a seam
      }
      // fade the top and bottom of the sheet so it has no hard edge
      x.globalCompositeOperation = "destination-in";
      const m = x.createLinearGradient(0, 0, 0, h); m.addColorStop(0, "rgba(0,0,0,0)"); m.addColorStop(0.15, "rgba(0,0,0,1)"); m.addColorStop(0.6, "rgba(0,0,0,1)"); m.addColorStop(1, "rgba(0,0,0,0)");
      x.fillStyle = m; x.fillRect(0, 0, w, h); x.globalCompositeOperation = "source-over";
      return { c, w, h, speed: layer ? 6 : 11, y: layer ? -H * 0.04 : 0 };
    });
  }

  function buildSkyline() {
    // a made-up Manhattan-like skyline: low blocks, a few towers, two spires, lit windows
    const c = document.createElement("canvas"), h = Math.round(Math.min(150, H * 0.17)); c.width = W; c.height = h;
    const x = c.getContext("2d"), r = rnd(7), base = h;
    x.fillStyle = "#0e1a1f";
    const lit = [];
    let px = 0;
    while (px < W) {
      const bw = 18 + r() * 46, tall = r() < 0.12, bh = (tall ? 0.55 + r() * 0.4 : 0.18 + r() * 0.32) * h;
      x.fillRect(px, base - bh, bw, bh);
      if (tall && r() < 0.35) { x.fillRect(px + bw * 0.35, base - bh - h * 0.12, bw * 0.3, h * 0.12); x.fillRect(px + bw * 0.48, base - bh - h * 0.24, 2, h * 0.12); }
      for (let wy = base - bh + 6; wy < base - 4; wy += 7) for (let wx = px + 4; wx < px + bw - 4; wx += 6) if (r() < 0.07) lit.push([wx, wy]);
      px += bw + (r() < 0.3 ? 2 + r() * 8 : 0);
    }
    // water line under the city
    const wg = x.createLinearGradient(0, base - 6, 0, base); wg.addColorStop(0, "rgba(11,21,25,0)"); wg.addColorStop(1, "#0b1519"); x.fillStyle = wg; x.fillRect(0, base - 6, W, 6);
    // distance haze over the upper buildings
    x.globalCompositeOperation = "source-atop";
    const hz = x.createLinearGradient(0, 0, 0, h); hz.addColorStop(0, "rgba(120,140,140,.45)"); hz.addColorStop(0.7, "rgba(120,140,140,0)");
    x.fillStyle = hz; x.fillRect(0, 0, W, h); x.globalCompositeOperation = "source-over";
    skyline = { c, h, lit };
  }

  function resize() {
    dpr = Math.min(2, window.devicePixelRatio || 1); W = innerWidth; H = innerHeight;
    cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + "px"; cv.style.height = H + "px";
    buildClouds(); buildSkyline(); drops = [];
    if (still) frame(performance.now());
  }

  function mix(a, b, k) { return a.map((v, i) => Math.round(v + (b[i] - v) * k)); }
  const rgb = a => `rgb(${a[0]},${a[1]},${a[2]})`;

  function makeBolt() {
    const pts = [[W * (0.25 + Math.random() * 0.5), H * 0.12]];
    while (pts[pts.length - 1][1] < H * 0.78) { const [x, y] = pts[pts.length - 1]; pts.push([x + (Math.random() - 0.5) * 70, y + 20 + Math.random() * 45]); }
    return pts;
  }

  function frame(now) {
    if (!still) requestAnimationFrame(frame);
    if (document.hidden || now - last < 33) return; // ~30 fps is plenty for weather
    const dt = Math.min(0.1, (now - (last || now)) / 1000); last = now;
    for (const k in mood) mood[k] += (target[k] - mood[k]) * Math.min(1, dt * 1.5);
    const s = mood.storm, time = (now - t0) / 1000;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);

    // sky: clear blue-grey to deep storm teal
    const top = mix([34, 58, 74], [14, 22, 26], s), mid = mix([84, 112, 124], [44, 60, 64], s), low = mix([150, 168, 164], [104, 120, 110], s);
    const sky = g.createLinearGradient(0, 0, 0, H); sky.addColorStop(0, rgb(top)); sky.addColorStop(0.55, rgb(mid)); sky.addColorStop(0.86, rgb(low)); sky.addColorStop(1, rgb(mix(low, [40, 50, 48], 0.6)));
    g.fillStyle = sky; g.fillRect(0, 0, W, H);

    // a soft glow where the sun breaks through near the horizon
    const glow = g.createRadialGradient(W * 0.78, H * 0.8, 10, W * 0.78, H * 0.8, W * 0.45);
    glow.addColorStop(0, `rgba(255,228,180,${0.32 - 0.18 * s})`); glow.addColorStop(1, "rgba(255,236,196,0)");
    g.fillStyle = glow; g.fillRect(0, 0, W, H);

    // clouds: thicker and darker with the storm
    const scroll = Math.min(1, scrollY / (H * 2));
    clouds.forEach((L, i) => {
      const off = still ? 0 : (time * L.speed * (0.6 + s)) % L.w;
      g.globalAlpha = (i ? 0.55 : 0.75) * (0.35 + 0.65 * s);
      const y = L.y - scroll * (i ? 30 : 60);
      g.drawImage(L.c, -off, y); g.drawImage(L.c, L.w - off, y);
    });
    g.globalAlpha = 1;
    // darkening veil under the cloud base
    const veil = g.createLinearGradient(0, 0, 0, H * 0.7); veil.addColorStop(0, `rgba(8,14,17,${0.35 * s})`); veil.addColorStop(1, "rgba(8,14,17,0)");
    g.fillStyle = veil; g.fillRect(0, 0, W, H);

    // lightning
    if (!still && mood.thunder > 0.3) {
      nextBolt -= dt * 1000;
      if (nextBolt <= 0) { flash = 1; bolt = makeBolt(); nextBolt = 5000 + Math.random() * 9000 / mood.thunder; }
    }
    if (bolt && flash > 0.55) {
      g.strokeStyle = `rgba(235,245,255,${flash})`; g.lineWidth = 2; g.shadowColor = "rgba(190,220,255,.9)"; g.shadowBlur = 18;
      g.beginPath(); bolt.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); g.stroke(); g.shadowBlur = 0;
    }

    // skyline with a few flickering windows
    if (skyline) {
      const y = H - skyline.h + scroll * 40;
      g.drawImage(skyline.c, 0, y);
      g.fillStyle = `rgba(255,214,140,${0.35 + 0.35 * s})`;
      skyline.lit.forEach(([x, wy], i) => { if (still || ((i * 7919 + Math.floor(time / 3)) % 23) > 1) g.fillRect(x, y + wy, 2, 2); });
    }

    const scrim = g.createLinearGradient(0, H * 0.55, 0, H); scrim.addColorStop(0, "rgba(6,12,15,0)"); scrim.addColorStop(1, "rgba(6,12,15,.55)");
    g.fillStyle = scrim; g.fillRect(0, H * 0.55, W, H * 0.45);

    // rain
    const want = Math.round(Math.min(700, mood.rain * 110) * (W * H) / (1440 * 900));
    while (drops.length < want) drops.push({ x: Math.random() * W * 1.2, y: Math.random() * H, v: 900 + Math.random() * 600, l: 12 + Math.random() * 16 });
    if (drops.length > want) drops.length = want;
    if (drops.length) {
      g.strokeStyle = "rgba(210,228,235,0.32)"; g.lineWidth = 1; g.beginPath();
      for (const d of drops) {
        if (!still) { d.y += d.v * dt; d.x -= d.v * dt * 0.18; if (d.y > H) { d.y = -d.l; d.x = Math.random() * W * 1.2; } }
        g.moveTo(d.x, d.y); g.lineTo(d.x + d.l * 0.18, d.y - d.l);
      }
      g.stroke();
    }

    // flash fades
    if (flash > 0) { g.fillStyle = `rgba(220,235,255,${flash * 0.22})`; g.fillRect(0, 0, W, H); flash = Math.max(0, flash - dt * 2.6); }
  }

  /* called by app.js: storm 0..1 (how dark), rain in mm per hour, thunder 0..1 */
  window.Sky = {
    set(o) {
      if (o.storm != null) target.storm = Math.max(0.12, Math.min(1, o.storm));
      if (o.rain != null) target.rain = Math.max(0, Math.min(6, o.rain));
      if (o.thunder != null) target.thunder = Math.max(0, Math.min(1, o.thunder));
      if (still) { Object.assign(mood, target); frame(performance.now()); }
    },
  };
  addEventListener("resize", () => { clearTimeout(resize.t); resize.t = setTimeout(resize, 200); });
  resize();
  if (!still) requestAnimationFrame(frame);
})();

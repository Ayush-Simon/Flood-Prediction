/* NYC Flood Watch: the ANN-LSTM and CNN-LSTM depth models, run in the browser.
   A plain-JavaScript copy of the trained PyTorch networks (weights exported by scripts/13a_export_lab_models.py).
   Input: the last hour = 12 five-minute steps of [depth in inches, rain in mm, rain-missing flag], and the time of the
   last step (UTC). Output: depth in inches 5, 10, ..., 40 minutes ahead. */
"use strict";
const FloodModels = (() => {
  const relu = v => v > 0 ? v : 0, sig = v => 1 / (1 + Math.exp(-v));
  const matvec = (W, x, b) => W.map((row, i) => row.reduce((s, w, k) => s + w * x[k], b ? b[i] : 0));

  // the ANN rain forecaster (scikit-learn MLP): last hour of rain + totals + time of day/year -> next 8 steps of rain
  function rainFeatures(hist, t) {
    const h = t.getUTCHours() + t.getUTCMinutes() / 60, m = t.getUTCMonth() + 1;
    const sum = hist.reduce((a, b) => a + b, 0), last3 = hist.slice(-3).reduce((a, b) => a + b, 0), wet = hist.filter(v => v > 0).length;
    return [...hist, sum, last3, wet, Math.sin(2 * Math.PI * h / 24), Math.cos(2 * Math.PI * h / 24), Math.sin(2 * Math.PI * m / 12), Math.cos(2 * Math.PI * m / 12)];
  }
  function annRain(M, hist, t) {
    let x = rainFeatures(hist, t).map((v, i) => (v - M.ann.mu[i]) / M.ann.sd[i]);
    M.ann.coefs.forEach((C, li) => {
      const out = C[0].map((_, j) => x.reduce((s, xv, i) => s + xv * C[i][j], M.ann.intercepts[li][j]));
      x = li < M.ann.coefs.length - 1 ? out.map(relu) : out;
    });
    return x.map(v => Math.max(0, v));
  }

  function conv1d(x, W, b) {                         // x: [channels][time], kernel 3, padding 1
    const T = x[0].length;
    return W.map((Wo, o) => Array.from({ length: T }, (_, t) => {
      let s = b[o];
      Wo.forEach((Wi, i) => Wi.forEach((w, k) => { const tt = t + k - 1; if (tt >= 0 && tt < T) s += w * x[i][tt]; }));
      return relu(s);
    }));
  }

  /* window: 12 rows of [depth, rain, missing]; time: Date of the last row (UTC) */
  function predict(M, window, time) {                               // "members": several trained runs, averaged (scripts/17a)
    if (M.members && M.members.length > 1) {
      const runs = M.members.map(w => one(M, w, window, time));
      return { depth: runs[0].depth.map((_, k) => runs.reduce((a, r) => a + r.depth[k], 0) / runs.length), rain: runs[0].rain };
    }
    return one(M, M.weights, window, time);
  }
  function one(M, w, window, time) {
    let s = window.map(r => r.map((v, i) => (v - M.seq_mean[i]) / M.seq_std[i]));           // time x 3
    if (M.cnn) {
      let x = [0, 1, 2].map(i => s.map(r => r[i]));                                          // channels x time
      x = conv1d(x, w["conv.0.weight"], w["conv.0.bias"]); x = conv1d(x, w["conv.2.weight"], w["conv.2.bias"]);
      s = x[0].map((_, t) => x.map(ch => ch[t]));
    }
    const Wi = w["lstm.weight_ih_l0"], Wh = w["lstm.weight_hh_l0"], bi = w["lstm.bias_ih_l0"], bh = w["lstm.bias_hh_l0"], hid = Wh[0].length;
    let h = new Array(hid).fill(0), c = new Array(hid).fill(0);
    for (const xt of s) {
      const gi = matvec(Wi, xt, bi), gh = matvec(Wh, h, bh), gsum = gi.map((v, k) => v + gh[k]);
      const I = gsum.slice(0, hid), F = gsum.slice(hid, 2 * hid), G = gsum.slice(2 * hid, 3 * hid), O = gsum.slice(3 * hid);
      c = c.map((cv, k) => sig(F[k]) * cv + sig(I[k]) * Math.tanh(G[k])); h = c.map((cv, k) => sig(O[k]) * Math.tanh(cv));
    }
    let rain = null, z = h;
    if (M.rain_forecast) { rain = annRain(M, window.map(r => r[1]), time); z = [...h, ...rain.map(v => (v - M.fut_mean) / M.fut_std)]; }
    const a = matvec(w["head.0.weight"], z, w["head.0.bias"]).map(relu);
    const out = matvec(w["head.3.weight"], a, w["head.3.bias"]).map(v => Math.max(0, v * M.y_sd));
    return { depth: out, rain };
  }
  /* The notebook's two LSTMs (scripts/15a, exported by 16b): "Improved LSTM" reads rain + depth, "Feature-enhanced LSTM"
     also reads rainfall intensity, 15/30/60-minute rain totals and the change in rain. Each forecasts one step ahead and
     is rolled forward 8 steps with the notebook's rainfall ANN, feeding its own forecasts back in.
     rain, depth: the whole series from the start of the storm (raw mm and inches); now: index of the last reading. */
  const features = (rain, depth, j) => {                              // the notebook's engineered features at row j
    const sum = k => { let s = 0; for (let i = Math.max(0, j - k + 1); i <= j; i++) s += rain[i]; return s; };
    return [rain[j], depth[j], rain[j] * 12, sum(3), sum(6), sum(12), j > 0 ? rain[j] - rain[j - 1] : 0];
  };
  const scale = (s, v, i = 0) => v * s.scale[i] + s.min[i], unscale = (s, v, i = 0) => (v - s.min[i]) / s.scale[i];
  function nbStep(M, hist, nr) {                                      // Keras LSTM (relu) -> concat next rain -> Dense 16 relu -> Dense 1
    const { kernel: K, recurrent: R, bias: B, units: u } = M.lstm;
    let h = new Array(u).fill(0), c = new Array(u).fill(0);
    for (const x of hist) {
      const z = B.map((b, k) => { let s = b; for (let i = 0; i < x.length; i++) s += x[i] * K[i][k]; for (let i = 0; i < u; i++) s += h[i] * R[i][k]; return s; });
      const ig = z.slice(0, u).map(sig), fg = z.slice(u, 2 * u).map(sig), cg = z.slice(2 * u, 3 * u).map(relu), og = z.slice(3 * u).map(sig);
      c = c.map((cv, k) => fg[k] * cv + ig[k] * cg[k]); h = c.map((cv, k) => og[k] * relu(cv));
    }
    const v = [...h, nr], a = M.dense1.b.map((b, k) => relu(v.reduce((s, x, i) => s + x * M.dense1.W[i][k], b)));
    return a.reduce((s, x, i) => s + x * M.dense2.W[i][0], M.dense2.b[0]);
  }
  function predictNB(M, rain, depth, now) {
    const L = M.lookback, rows = [...Array(L)].map((_, k) => Math.max(0, now - L + 1 + k));   // before the first reading: repeat it
    // stage 1: rainfall ANN on the last 75 minutes of (scaled) rain
    let x = rows.map(j => scale(M.scale_rain, rain[j]));
    M.ann.forEach((l, li) => { x = l.b.map((b, k) => relu(x.reduce((s, v, i) => s + v * l.W[i][k], b))); });
    const frS = x, fr = frS.map(v => Math.max(0, unscale(M.scale_rain, v)));
    // stage 2: one step at a time
    const sd = M.scale_depth, out = [];
    if (!M.enhanced) {
      let hist = rows.map(j => [scale(M.scale_rain, rain[j]), scale(sd, depth[j])]);
      for (let i = 0; i < M.H; i++) {
        const b = nbStep(M, hist, frS[i]); out.push(Math.max(0, unscale(sd, b)));
        hist = [...hist.slice(1), [frS[i], b]];
      }
    } else {
      const sf = M.scale_feat, sc = f => f.map((v, i) => scale(sf, v, i));
      let hist = rows.map(j => sc(features(rain, depth, j))), recent = rows.slice(-12).map(j => rain[j]);
      for (let i = 0; i < M.H; i++) {
        const e = nbStep(M, hist, frS[i]), ed = Math.max(0, unscale(sd, e)); out.push(ed);
        recent = [...recent.slice(1), fr[i]]; const cur = recent[11], s = k => recent.slice(12 - k).reduce((a, b) => a + b, 0);
        hist = [...hist.slice(1), sc([cur, ed, cur * 12, s(3), s(6), s(12), cur - recent[10]])];
      }
    }
    return { depth: out, rain: fr };
  }
  return { predict, predictNB, features };
})();
if (typeof module !== "undefined") module.exports = FloodModels;

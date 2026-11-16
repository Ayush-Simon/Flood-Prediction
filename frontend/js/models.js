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
  function predict(M, window, time) {
    const w = M.weights;
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
  return { predict };
})();
if (typeof module !== "undefined") module.exports = FloodModels;

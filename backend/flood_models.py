"""Numpy-only inference for the project's models, used by the live app (no PyTorch needed)."""
import numpy as np, pandas as pd

def rain_feats(hist, t):
    t = pd.DatetimeIndex(t); h = t.hour + t.minute / 60
    return np.column_stack([hist, hist.sum(1), hist[:, -3:].sum(1), (hist > 0).sum(1),
                            np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24), np.sin(2*np.pi*t.month/12), np.cos(2*np.pi*t.month/12)])

class DepthModel:
    """ANN rain forecaster -> CNN-LSTM depth model. Input: (n, 12, 3) = depth in, rain mm, rain-missing flag; times in UTC."""
    def __init__(self, j):
        self.j = j                                                    # "members": several trained runs, averaged (scripts/17a)
        self.members = [{k: np.asarray(v, dtype=np.float64) for k, v in m.items()} for m in j.get("members", [j["weights"]])]
        self.w = self.members[0]
        self.ann = [(np.asarray(c), np.asarray(b)) for c, b in zip(j["ann"]["coefs"], j["ann"]["intercepts"])]
    def rain_forecast(self, rain_hist, times):
        x = (rain_feats(rain_hist, times) - np.asarray(self.j["ann"]["mu"])) / np.asarray(self.j["ann"]["sd"])
        for i, (c, b) in enumerate(self.ann):
            x = x @ c + b
            if i < len(self.ann) - 1: x = np.maximum(x, 0)
        return np.clip(x, 0, None)                                    # mm per 5 min, next 8 steps
    def predict(self, seq, times, return_rain=False):
        j = self.j
        rain_fc = self.rain_forecast(seq[:, :, 1], times)
        s = (seq - np.asarray(j["seq_mean"])) / np.asarray(j["seq_std"])
        f = (rain_fc - j["fut_mean"]) / j["fut_std"]
        out = np.mean([self._net(w, s, f) for w in self.members], 0)  # average of the runs, as scored in 07d
        return (out, rain_fc) if return_rain else out
    def _net(self, w, s, f):
        j = self.j
        x = s.transpose(0, 2, 1)                                      # conv: (n, channels, time)
        for k in ("conv.0", "conv.2"):
            W, b = w[k + ".weight"], w[k + ".bias"]; xp = np.pad(x, ((0, 0), (0, 0), (1, 1)))
            x = np.maximum(np.einsum("oik,nitk->not", W, np.stack([xp[:, :, i:i + x.shape[2]] for i in range(3)], -1)) + b[None, :, None], 0)
        x = x.transpose(0, 2, 1)
        Wi, Wh, bi, bh = w["lstm.weight_ih_l0"], w["lstm.weight_hh_l0"], w["lstm.bias_ih_l0"], w["lstm.bias_hh_l0"]
        n, hid = x.shape[0], Wh.shape[1]; h = np.zeros((n, hid)); c = np.zeros((n, hid)); sig = lambda v: 1 / (1 + np.exp(-v))
        for t in range(x.shape[1]):
            g = x[:, t] @ Wi.T + bi + h @ Wh.T + bh; i_, f_, g_, o_ = np.split(g, 4, axis=1)
            c = sig(f_) * c + sig(i_) * np.tanh(g_); h = sig(o_) * np.tanh(c)
        z = np.concatenate([h, f], 1)
        z = np.maximum(z @ w["head.0.weight"].T + w["head.0.bias"], 0) @ w["head.3.weight"].T + w["head.3.bias"]
        return np.clip(z * j["y_sd"], 0, None)

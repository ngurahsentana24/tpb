"""Metrik akurasi, uji Diebold-Mariano (koreksi HLN), dan uji arah Pesaran-Timmermann (Bab 3.7)."""
import numpy as np
from scipy import stats


def metrics(a, p, prev):
    a, p, prev = map(lambda z: np.asarray(z, float), (a, p, prev))
    e = a - p
    ep = a - prev
    chg = np.abs(a - prev) > 1e-9
    out = {'rmse': float(np.sqrt(np.mean(e ** 2)) * 100), 'mae': float(np.mean(np.abs(e)) * 100),
           'mape': float(np.mean(np.abs(e) / np.maximum(np.abs(a), 1e-8)) * 100),
           'r2oos': float(1 - np.sum(e ** 2) / max(np.sum(ep ** 2), 1e-12)),
           'rmse_chg': float(np.sqrt(np.mean(e[chg] ** 2)) * 100) if chg.any() else None,
           'rmse_fix': float(np.sqrt(np.mean(e[~chg] ** 2)) * 100) if (~chg).any() else None,
           'n': int(len(a)), 'n_chg': int(chg.sum())}
    dp = np.sign(p - prev)[chg]
    da = np.sign(a - prev)[chg]
    if chg.any() and np.any(dp != 0):
        out['dir_acc'] = float(np.mean(dp == da) * 100)
        out['pt_p'] = pesaran_timmermann(da, dp)
    else:
        out['dir_acc'], out['pt_p'] = None, None
    return out


def pesaran_timmermann(da, dp):
    da, dp = np.asarray(da) > 0, np.asarray(dp) > 0
    n = len(da)
    if n < 5:
        return None
    p_hat = np.mean(da == dp)
    py, px = da.mean(), dp.mean()
    p_star = py * px + (1 - py) * (1 - px)
    v1 = p_star * (1 - p_star) / n
    v2 = ((2 * py - 1) ** 2 * px * (1 - px) / n + (2 * px - 1) ** 2 * py * (1 - py) / n
          + 4 * py * px * (1 - py) * (1 - px) / n ** 2)
    if v1 - v2 <= 0:
        return None
    z = (p_hat - p_star) / np.sqrt(v1 - v2)
    return float(1 - stats.norm.cdf(z))


def dm_test(e1, e2, h=1, loss='SE'):
    """DM < 0 → model pertama lebih akurat. Koreksi Harvey-Leybourne-Newbold, p dua sisi (t, n−1)."""
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    L = (lambda e: e ** 2) if loss == 'SE' else np.abs
    d = L(e1) - L(e2)
    n = len(d)
    if n < 5 or np.allclose(d, 0):
        return None, None
    dbar = d.mean()
    gam = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    if var <= 0:
        return None, None
    dm = dbar / np.sqrt(var)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)

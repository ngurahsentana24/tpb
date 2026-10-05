"""Dekomposisi CEEMDAN/EMD deskriptif dan kausal (jendela bergulir) + pengelompokan co-IMF (Bab 3.4, 4.2)."""
import numpy as np
from sklearn.cluster import KMeans

try:
    from PyEMD import EMD, CEEMDAN
    HAS_EMD = True
except Exception:  # pragma: no cover
    HAS_EMD = False

DAYS_PER_MONTH = 21


def decompose(x, method='ceemdan', trials=50, max_imf=8, seed=42):
    """Mengembalikan (imfs [k, n], residual [n]); imfs diurutkan dari frekuensi tertinggi."""
    x = np.asarray(x, float)
    if not HAS_EMD or np.ptp(x) < 1e-12:
        return np.zeros((0, len(x))), x.copy()
    if method == 'ceemdan':
        d = CEEMDAN(trials=trials, epsilon=0.005, parallel=False)
        d.noise_seed(seed)
        imfs = d.ceemdan(x, max_imf=max_imf)
    else:
        d = EMD()
        imfs = d.emd(x, max_imf=max_imf)
    try:
        imfs, res = d.get_imfs_and_residue()
    except Exception:
        res = x - imfs.sum(0)
    imfs = np.atleast_2d(imfs)
    if imfs.shape[1] != len(x):
        imfs = imfs.T
    return imfs, np.asarray(res, float)


def mean_period(c):
    c = np.asarray(c, float)
    peaks = np.sum((c[1:-1] > c[:-2]) & (c[1:-1] > c[2:]))
    return len(c) / peaks if peaks > 0 else np.nan


def sample_entropy(x, m=2, r=0.2):
    x = np.asarray(x, float)
    sd = x.std()
    if sd < 1e-12 or len(x) <= m + 2:
        return 0.0
    tol = r * sd

    def count(mm):
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)
        d = np.max(np.abs(emb[:, None, :] - emb[None, :, :]), axis=2)
        return (np.sum(d <= tol) - len(emb)) / 2

    B, A = count(m), count(m + 1)
    if B <= 0 or A <= 0:
        return 2.5
    return float(-np.log(A / B))


def describe(x, imfs, res):
    """Tabel karakteristik komponen: periode, variansi ternormalisasi, korelasi, kategori horizon."""
    comps = list(imfs) + [res]
    var = np.array([np.var(c) for c in comps])
    tot = var.sum() if var.sum() > 0 else 1.0
    rows = []
    for k, c in enumerate(comps):
        is_res = k == len(comps) - 1
        per = np.nan if is_res else mean_period(c) / DAYS_PER_MONTH
        if is_res:
            cat, imp = 'Tren jangka panjang', 'Jangkar level TBP'
        elif per < 1:
            cat, imp = 'Derau (< 1 bulan)', 'Diabaikan dalam penetapan TBP'
        elif per < 4:
            cat, imp = 'Siklus RDG (1–4 bulan)', 'Sinyal jangka pendek; dipantau'
        else:
            cat, imp = 'Siklus menengah (≥ 4 bulan)', 'Sinyal yang perlu direspons TBP'
        rows.append({'comp': 'Residual' if is_res else f'IMF{k + 1}', 'period_m': None if is_res else float(per),
                     'var_pct': float(var[k] / tot * 100), 'corr': float(np.corrcoef(c, x)[0, 1]) if np.std(c) > 0 else 0.0,
                     'category': cat, 'implication': imp})
    S = np.array(comps)
    G = S @ S.T
    off = G.sum() - np.trace(G)
    io = float(off / (np.sum(x ** 2) + 1e-12))
    return rows, io


def fit_k(imfs, K):
    """Menyamakan jumlah IMF ke K: IMF kasar berlebih digabung ke residual, kekurangan diisi nol."""
    k = imfs.shape[0]
    if k == K:
        return imfs, np.zeros(imfs.shape[1])
    if k > K:
        return imfs[:K], imfs[K:].sum(0)
    pad = np.zeros((K - k, imfs.shape[1]))
    return np.vstack([imfs, pad]), np.zeros(imfs.shape[1])


def causal_tails(y, window=252, K=6, method='emd', trials=20, seed=42, start=None, progress=None):
    """F[t] = nilai ujung (indeks t) tiap komponen dari dekomposisi y[t−W+1 : t+1] — hanya data s.d. t."""
    y = np.asarray(y, float)
    n = len(y)
    F = np.full((n, K + 1), np.nan)
    t0 = max(window - 1, 0) if start is None else max(start, window - 1)
    total = n - t0
    for i, t in enumerate(range(t0, n)):
        seg = y[t - window + 1:t + 1]
        imfs, res = decompose(seg, method=method, trials=trials, max_imf=K, seed=seed)
        imfs, extra = fit_k(imfs, K)
        F[t, :K] = imfs[:, -1]
        F[t, K] = (res + extra)[-1]
        if progress and i % 100 == 0:
            progress(i / max(total, 1))
    return F


def co_imf_labels(y_train, K=6, method='emd', trials=20, n_groups=3, seed=42):
    """Label co-IMF dari jendela latih: Sample Entropy setiap komponen → K-Means; 0 = frekuensi tinggi."""
    imfs, res = decompose(y_train, method=method, trials=trials, max_imf=K, seed=seed)
    imfs, extra = fit_k(imfs, K)
    comps = list(imfs) + [res + extra]
    se = np.array([sample_entropy(c[-min(len(c), 400):]) for c in comps])
    g = min(n_groups, len(comps))
    km = KMeans(n_clusters=g, n_init=10, random_state=seed).fit(se.reshape(-1, 1))
    order = np.argsort(-km.cluster_centers_.ravel())          # entropi tinggi → kelompok 0
    remap = {c: i for i, c in enumerate(order)}
    lab = np.array([remap[c] for c in km.labels_])
    lab[-1] = g - 1                                            # residual = frekuensi rendah
    periods = [mean_period(c) / DAYS_PER_MONTH for c in imfs]
    return lab, se, periods


def group_sum(F, labels, n_groups=3):
    return np.column_stack([F[:, labels == g].sum(1) for g in range(n_groups)])

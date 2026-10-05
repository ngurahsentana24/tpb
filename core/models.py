"""Model peramalan: ARIMA(0,1,0), CEEMDAN-LSTM berbasis ARIMA (koreksi λ, aturan 1-SE), LSTM murni,
dan pembanding naif (Bab 3.5). Jaringan saraf memakai PyTorch LSTM bila tersedia; jika tidak, memakai
MLPRegressor (scikit-learn) sebagai fallback ringan — tercatat di metadata run."""
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False
from sklearn.neural_network import MLPRegressor

LAMBDA_GRID = np.round(np.arange(0, 1.01, 0.1), 2)


def engine_name(pref='auto'):
    if pref in ('auto', 'lstm') and HAS_TORCH:
        return 'lstm'
    return 'mlp'


# ----------------------------------------------------------------------------- regresor sekuens
class _LSTMNet(nn.Module if HAS_TORCH else object):
    def __init__(self, n_feat, units, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, units, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(units, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.out(self.drop(h[:, -1, :])).squeeze(-1)


class SeqRegressor:
    """Ensemble beberapa seed. X: [N, L, F] → y: [N]."""

    def __init__(self, engine='auto', units=32, dropout=0.1, lr=2e-3, epochs=60, batch=32, patience=8, seeds=(42, 7, 2024)):
        self.engine = engine_name(engine)
        self.cfg = dict(units=units, dropout=dropout, lr=lr, epochs=epochs, batch=batch, patience=patience)
        self.seeds = list(seeds)
        self.models = []
        self.epochs_used = []

    def fit(self, X, y, Xv=None, yv=None):
        self.models = []
        for s in self.seeds:
            if self.engine == 'lstm':
                self.models.append(self._fit_torch(X, y, Xv, yv, s))
            else:
                m = MLPRegressor(hidden_layer_sizes=(self.cfg['units'],), alpha=1e-3, learning_rate_init=self.cfg['lr'],
                                 max_iter=300, early_stopping=True, validation_fraction=0.15, n_iter_no_change=12,
                                 random_state=s)
                m.fit(X.reshape(len(X), -1), y)
                self.epochs_used.append(int(m.n_iter_))
                self.models.append(m)
        return self

    def _fit_torch(self, X, y, Xv, yv, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        c = self.cfg
        net = _LSTMNet(X.shape[2], c['units'], c['dropout'])
        opt = torch.optim.Adam(net.parameters(), lr=c['lr'])
        lossf = nn.MSELoss()
        Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        has_v = Xv is not None and len(Xv) > 0
        if has_v:
            Xvt, yvt = torch.tensor(Xv, dtype=torch.float32), torch.tensor(yv, dtype=torch.float32)
        best, best_state, bad, used = np.inf, None, 0, 0
        n = len(Xt)
        for ep in range(c['epochs']):
            net.train()
            perm = torch.randperm(n)
            for i in range(0, n, c['batch']):
                idx = perm[i:i + c['batch']]
                opt.zero_grad()
                loss = lossf(net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            used = ep + 1
            if has_v:
                net.eval()
                with torch.no_grad():
                    vl = float(lossf(net(Xvt), yvt))
                if vl < best - 1e-7:
                    best, bad = vl, 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= c['patience']:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.epochs_used.append(used)
        return net

    def predict(self, X):
        if len(X) == 0:
            return np.zeros(0)
        out = []
        for m in self.models:
            if self.engine == 'lstm':
                with torch.no_grad():
                    out.append(m(torch.tensor(X, dtype=torch.float32)).numpy())
            else:
                out.append(m.predict(X.reshape(len(X), -1)))
        return np.mean(out, axis=0)


# ----------------------------------------------------------------------------- fitur
def build_hybrid_features(y, G, exog, L):
    """Matriks fitur per waktu t (dipakai untuk meramal y[t+1]) — hanya informasi s.d. t.
    Kolom: Δy, co-IMF (G), ΔBI, ΔTBP, (y − BI), (y − TBP)."""
    dy = np.r_[0.0, np.diff(y)]
    bi, tbp = exog[:, 0], exog[:, 1]
    cols = [dy, *[G[:, g] for g in range(G.shape[1])], np.r_[0.0, np.diff(bi)], np.r_[0.0, np.diff(tbp)], y - bi, y - tbp]
    return np.column_stack(cols)


def build_level_features(y, exog):
    """Fitur LSTM murni: deret asli + eksogen (tanpa komponen CEEMDAN)."""
    dy = np.r_[0.0, np.diff(y)]
    bi, tbp = exog[:, 0], exog[:, 1]
    return np.column_stack([dy, np.r_[0.0, np.diff(bi)], np.r_[0.0, np.diff(tbp)], y - bi, y - tbp])


def windows(M, idx, L):
    """Sekuens M[t−L+1 : t+1] untuk setiap t dalam idx (fitur s.d. t untuk target t+1)."""
    return np.stack([M[t - L + 1:t + 1] for t in idx]) if len(idx) else np.zeros((0, L, M.shape[1]))


class Scaler:
    def fit(self, M):
        self.mu = np.nanmean(M, 0)
        self.sd = np.nanstd(M, 0)
        self.sd[self.sd < 1e-9] = 1.0
        return self

    def tf(self, M):
        return (M - self.mu) / self.sd


def choose_lambda(y_true, base, corr):
    """Aturan 1-SE: λ terkecil yang galat kuadrat rata-ratanya ≤ minimum + 1 SE."""
    errs = [(y_true - (base + lam * corr)) ** 2 for lam in LAMBDA_GRID]
    means = np.array([e.mean() for e in errs])
    j = int(np.argmin(means))
    se = errs[j].std(ddof=1) / np.sqrt(len(errs[j])) if len(errs[j]) > 1 else 0.0
    ok = np.where(means <= means[j] + se)[0]
    return float(LAMBDA_GRID[ok.min()]), float(LAMBDA_GRID[j])


# ----------------------------------------------------------------------------- pembanding sederhana
def persistence(y, idx):
    return y[np.asarray(idx) - 1]


def linear_trend(y, idx, window=60):
    out = []
    for t in idx:
        seg = y[max(0, t - window):t]
        x = np.arange(len(seg))
        b = np.polyfit(x, seg, 1)
        out.append(np.polyval(b, len(seg)))
    return np.array(out)


def rolling_mean(y, idx, window=30):
    return np.array([y[max(0, t - window):t].mean() for t in idx])

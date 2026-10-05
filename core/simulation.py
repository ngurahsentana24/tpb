"""Simulasi lima langkah (Bab 4.9): skenario BI Rate → TBP/LF/DF → Deposito 1M (multiplier ARDL/NARDL
di atas jalur bootstrap proyeksi CEEMDAN-LSTM) → gap → P(breach) dan TBP minimum."""
import numpy as np

STEP = 0.25


def round_grid(x):
    return np.round(np.asarray(x, float) / STEP) * STEP


def ceil_grid(x):
    return np.ceil(np.asarray(x, float) / STEP - 1e-9) * STEP


def dyn_response(dev, m_pos, m_neg=None):
    dev = np.atleast_2d(dev)
    inc = np.diff(np.c_[np.zeros(len(dev)), dev], axis=1)
    H = dev.shape[1]
    res = np.zeros_like(dev, dtype=float)
    for j in range(H):
        ij = inc[:, j][:, None]
        mm = m_pos[None, :H - j] if m_neg is None else np.where(ij >= 0, m_pos[None, :H - j], m_neg[None, :H - j])
        res[:, j:] += ij * mm
    return res


class Simulator:
    def __init__(self, ctx):
        self.c = ctx
        self.base = np.asarray(ctx['base_paths'], float)        # [B0, Hm] rata-rata Deposito bulanan
        self.H = self.base.shape[1]
        m = ctx['multipliers']
        self.mult = {'ardl': np.asarray(m['ardl'][:self.H]),
                     'nardl': (np.asarray(m['nardl_pos'][:self.H]), np.asarray(m['nardl_neg'][:self.H]))}

    def _bi_paths(self, B, shock, stochastic, seed):
        start = self.c['bi_last'] + shock / 100
        if not stochastic:
            return np.full((B, self.H), start)
        k, th, sg = (self.c['cir'][x] for x in ('kappa', 'theta', 'sigma'))
        rng = np.random.RandomState(seed + 1)
        x = np.full(B, start)
        P = np.zeros((B, self.H))
        for h in range(self.H):
            x = np.maximum(x + k * (th + shock / 100 - x) + sg * np.sqrt(np.maximum(x, 0)) * rng.normal(size=B), 0)
            P[:, h] = x
        return round_grid(P)

    def _base(self, B, seed):
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, len(self.base), B) if B != len(self.base) else np.arange(B)
        return self.base[idx]

    def run(self, shock=0, rule='C', adj=0, lag=1, transmission='ardl', tol=10, B=1000, stochastic=False, seed=42,
            with_heat=True):
        c = self.c
        B = int(max(100, min(B, 5000)))
        base = self._base(B, seed)
        bi = self._bi_paths(B, shock, stochastic, seed)
        dev = bi - c['bi_last']
        sh = dyn_response(dev, *self.mult['nardl']) if transmission == 'nardl' else dyn_response(dev, self.mult['ardl'])
        dep = base + sh
        tbp = self._tbp(rule, bi, dep, adj, lag)
        tolf = tol / 100
        pb = (dep > tbp).mean(0)
        q = lambda p: np.percentile(dep, p, axis=0)
        req = ceil_grid(np.quantile(dep, 1 - tolf, axis=0))
        months = c['months'][:self.H]
        table = [{'month': months[h], 'bi': float(np.median(bi[:, h])), 'lf': float(np.median(bi[:, h]) + c['sp_lf']),
                  'df': float(np.median(bi[:, h]) + c['sp_df']), 'tbp': float(np.median(tbp[:, h])),
                  'dep': float(np.median(dep[:, h])), 'gap': float((np.median(dep[:, h]) - np.median(tbp[:, h])) * 100),
                  'pb': float(pb[h]), 'tbp_min': float(req[h])} for h in range(self.H)]
        out = {'months': months, 'pb': pb.tolist(), 'q05': q(5).tolist(), 'q25': q(25).tolist(), 'q50': q(50).tolist(),
               'q75': q(75).tolist(), 'q95': q(95).tolist(), 'tbp_med': np.median(tbp, 0).tolist(), 'tbp_min': req.tolist(),
               'pmax': float(pb.max()), 'pmax_month': months[int(np.argmax(pb))], 'tbp_min_max': float(req.max()),
               'tbp_min_month': months[int(np.argmax(req))], 'adj_needed_bp': float((req.max() - c['tbp_last']) * 100),
               'gap_end_bp': float((np.median(dep[:, -1]) - np.median(tbp[:, -1])) * 100),
               'tbp_end': float(np.median(tbp[:, -1])), 'dep_last': c['dep_last'], 'tbp_last': c['tbp_last'],
               'table': table}
        if with_heat:
            out['heat'] = self.heat(adj, lag, transmission, B=min(B, 600), seed=seed)
        return out

    def _tbp(self, rule, bi, dep, adj, lag):
        c = self.c
        lagmask = np.arange(self.H) >= 1                     # respons TBP mulai bulan ke-2 (lag 1 bulan)
        if rule == 'A':
            t = np.where(lagmask[None, :], round_grid(bi + c['spread_tbp_bi']), c['tbp_last'])
        elif rule == 'D':
            prev = np.column_stack([np.full(dep.shape[0], c['dep_last']), dep[:, :-1]])
            t = ceil_grid(prev + 0.25)
        else:
            t = np.full(bi.shape, c['tbp_last'])
        if adj:
            t = t + np.where(np.arange(self.H) + 1 >= lag, adj / 100, 0.0)[None, :]
        return t

    def heat(self, adj=0, lag=1, transmission='ardl', B=600, seed=42):
        shocks, rules = [-50, -25, 0, 25, 50], ['A', 'C', 'D']
        M = []
        for s in shocks:
            row = []
            for r in rules:
                o = self.run(s, r, adj, lag, transmission, 10, B, False, seed, with_heat=False)
                row.append(o['pmax'])
            M.append(row)
        return {'shocks': shocks, 'rules': rules, 'pmax': M}

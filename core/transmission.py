"""Transmisi BI Rate → TBP & Deposito: ARDL/ECM dengan uji bounds dan NARDL (asimetri), data bulanan (Bab 4.7)."""
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm

PSS_5 = {1: (4.94, 5.73), 2: (3.79, 4.85)}   # Pesaran et al. (2001), kasus III, 5%


def _frame(y, xs, p, q):
    d = pd.DataFrame({'dy': y.diff(), 'y1': y.shift(1)})
    for name, x in xs.items():
        d[f'{name}1'] = x.shift(1)
        for j in range(q):
            d[f'd{name}{j}'] = x.diff().shift(j)
    for i in range(1, p):
        d[f'dy{i}'] = y.diff().shift(i)
    return d


def _fit(y, xs, Pmax=4, Qmax=4):
    best = None
    burn = max(Pmax, Qmax) + 1
    for p, q in itertools.product(range(1, Pmax + 1), range(0, Qmax + 1)):
        d = _frame(y, xs, p, q).iloc[burn:].dropna()
        if len(d) < 20:
            continue
        r = sm.OLS(d['dy'], sm.add_constant(d.drop(columns='dy'))).fit()
        if best is None or r.aic < best[0].aic:
            best = (r, p, q)
    return best


def _theta(r, b):
    from scipy import stats
    a = r.params['y1']
    th = -r.params[b] / a
    g = np.array([r.params[b] / a ** 2, -1 / a])          # delta method: ∂θ/∂α, ∂θ/∂β
    V = r.cov_params().loc[['y1', b], ['y1', b]].values
    se = float(np.sqrt(max(g @ V @ g, 0)))
    p = float(2 * (1 - stats.norm.cdf(abs(th / se)))) if se > 0 else float('nan')
    return float(th), se, p


def half_life(a):
    return float(np.log(0.5) / np.log(1 + a)) if -1 < a < 0 else float('nan')


def ardl(y, x, xname='x'):
    r, p, q = _fit(y, {xname: x})
    th, se, pth = _theta(r, f'{xname}1')
    F = float(r.f_test(f'y1 = 0, {xname}1 = 0').fvalue)
    lo, hi = PSS_5[1]
    a = float(r.params['y1'])
    g0 = float(r.params.get(f'd{xname}0', 0.0))
    return {'p': p, 'q': q, 'n': int(r.nobs), 'F': F, 'coint': F > hi, 'inconclusive': lo <= F <= hi,
            'theta': th, 'se': se, 'p_theta': pth, 'alpha': a, 'p_alpha': float(r.pvalues['y1']),
            'hl': half_life(a), 'g0': g0, 'r2': float(r.rsquared)}


def nardl(y, x):
    dx = x.diff().fillna(0)
    xp, xn = dx.clip(lower=0).cumsum(), dx.clip(upper=0).cumsum()
    r, p, q = _fit(y, {'xp': xp, 'xn': xn})
    a = float(r.params['y1'])
    tp, _, _ = _theta(r, 'xp1')
    tn, _, _ = _theta(r, 'xn1')
    F = float(r.f_test('y1 = 0, xp1 = 0, xn1 = 0').fvalue)
    lo, hi = PSS_5[2]
    w = r.f_test('xp1 = xn1')
    return {'p': p, 'q': q, 'n': int(r.nobs), 'F': F, 'coint': F > hi, 'inconclusive': lo <= F <= hi,
            'theta_pos': tp, 'theta_neg': tn, 'lr_F': float(w.fvalue), 'lr_p': float(w.pvalue), 'alpha': a,
            'hl': half_life(a), 'g0p': float(r.params.get('dxp0', 0.0)), 'g0n': float(r.params.get('dxn0', 0.0))}


def multiplier(theta, alpha, g0, H=6):
    m = np.zeros(H)
    if not np.isfinite(theta):
        return m
    m[0] = g0 if np.isfinite(g0) else 0.0
    for h in range(1, H):
        m[h] = m[h - 1] + alpha * (m[h - 1] - theta) if -1.5 < alpha < 0 else theta
    return m


def analyse(monthly, target='Deposito_1M_Avg', H=6):
    M = monthly
    out = {'ardl': [], 'nardl': []}
    pairs = [('BI_Rate', 'LPS_Rate', 'BI Rate → TBP LPS'), ('BI_Rate', target, 'BI Rate → Deposito 1M'),
             ('LPS_Rate', target, 'TBP LPS → Deposito 1M')]
    res = {}
    for x, y, lab in pairs:
        try:
            e = ardl(M[y], M[x], 'x')
            e['path'] = lab
            out['ardl'].append(e)
            res[(x, y)] = e
        except Exception as ex:  # pragma: no cover
            out['ardl'].append({'path': lab, 'error': str(ex)})
    for x, y, lab in pairs[:2]:
        try:
            e = nardl(M[y], M[x])
            e['path'] = lab
            out['nardl'].append(e)
            res[('N', x, y)] = e
        except Exception as ex:  # pragma: no cover
            out['nardl'].append({'path': lab, 'error': str(ex)})
    e = res.get(('BI_Rate', target))
    n = res.get(('N', 'BI_Rate', target))
    out['multipliers'] = {
        'ardl': (multiplier(e['theta'], e['alpha'], e['g0'], H) if e else np.zeros(H)).tolist(),
        'nardl_pos': (multiplier(n['theta_pos'], n['alpha'], n['g0p'], H) if n else np.zeros(H)).tolist(),
        'nardl_neg': (multiplier(n['theta_neg'], n['alpha'], n['g0n'], H) if n else np.zeros(H)).tolist()}
    return out

"""Statistik deskriptif, uji stasioneritas, siklus kebijakan BI, dan efektivitas TBP (Bab 4.1 & 4.8)."""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from .io import VARS, TARGET, LABEL

warnings.filterwarnings('ignore')


def descriptive(D):
    rows = []
    for v in VARS:
        s = D['series'][v] if v == TARGET else D['ff'][v].dropna()
        rows.append({'var': LABEL[v], 'mean': float(s.mean()), 'sd': float(s.std()), 'min': float(s.min()),
                     'max': float(s.max()), 'skew': float(s.skew()), 'kurt': float(s.kurt()),
                     'fixed_pct': float((s.diff().abs() < 1e-9).mean() * 100), 'n': int(len(s))})
    return rows


def _kpss_p(x):
    try:
        return float(kpss(x, regression='c', nlags='auto')[1])
    except Exception:
        return float('nan')


def stationarity(D):
    rows = []
    for v in VARS:
        s = (D['series'][v] if v == TARGET else D['ff'][v].dropna()).values
        r = {'var': LABEL[v]}
        for lab, x in (('lvl', s), ('dif', np.diff(s))):
            try:
                a = adfuller(x, autolag='AIC')
                r[f'adf_{lab}'], r[f'adf_p_{lab}'] = float(a[0]), float(a[1])
            except Exception:
                r[f'adf_{lab}'], r[f'adf_p_{lab}'] = float('nan'), float('nan')
            r[f'kpss_p_{lab}'] = _kpss_p(x)
        rows.append(r)
    return rows


def cycles(bi):
    """Siklus pelonggaran/pengetatan dari perubahan BI Rate (identik kode v29)."""
    d = bi.diff()
    ch = d[d.abs() > 1e-9]
    runs = []
    for dt, val in ch.items():
        sg = int(np.sign(val))
        if runs and runs[-1]['dir'] == sg:
            runs[-1]['last'] = dt
            runs[-1]['n'] += 1
            runs[-1]['total'] += val * 100
        else:
            runs.append({'dir': sg, 'start': dt, 'last': dt, 'n': 1, 'total': val * 100})
    out = []
    for i, r in enumerate(runs):
        end = runs[i + 1]['start'] - pd.Timedelta(days=1) if i + 1 < len(runs) else bi.index[-1]
        a, b = r['start'].year, r['last'].year
        name = f"{'Pengetatan' if r['dir'] > 0 else 'Pelonggaran'} {a}" + (f'–{b}' if b != a else '')
        out.append({'name': name, 'dir': r['dir'], 'start': f"{r['start']:%Y-%m-%d}", 'end': f'{end:%Y-%m-%d}',
                    'n': r['n'], 'total_bp': float(r['total'])})
    return out


def _episodes(mask):
    res, start = [], None
    vals = mask.values
    for i, m in enumerate(vals):
        if m and start is None:
            start = i
        if (not m or i == len(vals) - 1) and start is not None:
            end = i if m else i - 1
            res.append((start, end))
            start = None
    return res


def effectiveness(D, cyc):
    al = D['aligned']

    def row(df, name):
        gap = (df[TARGET] - df['LPS_Rate']) * 100
        br = gap > 0
        eps = _episodes(br)
        depth = max([gap.iloc[a:b + 1].max() for a, b in eps], default=0.0)
        cw = (df['Lending_Facility'] - df['Deposit_Facility']).replace(0, np.nan)
        dbi = (df['BI_Rate'].iloc[-1] - df['BI_Rate'].iloc[0]) * 100
        dt = (df['LPS_Rate'].iloc[-1] - df['LPS_Rate'].iloc[0]) * 100
        dd = (df[TARGET].iloc[-1] - df[TARGET].iloc[0]) * 100
        return {'period': name, 'start': f'{df.index[0]:%Y-%m-%d}', 'end': f'{df.index[-1]:%Y-%m-%d}', 'days': len(df),
                'gap': float(gap.mean()), 'breach': float(br.mean() * 100), 'episodes': len(eps), 'depth': float(depth),
                'corridor': float(((df['LPS_Rate'] - df['Deposit_Facility']) / cw).mean()),
                'tbp_bi': float((dt / dbi) if abs(dbi) > 1e-9 else np.nan),
                'dep_bi': float((dd / dbi) if abs(dbi) > 1e-9 else np.nan)}

    rows = [row(al, f'Seluruh sampel {al.index[0].year}–{al.index[-1].year}')]
    for c in cyc:
        seg = al.loc[c['start']:c['end']]
        if len(seg) > 5:
            rows.append(row(seg, c['name']))
    return rows


def gap_stats(D):
    al = D['aligned']
    g = (al[TARGET] - al['LPS_Rate']) * 100
    return {'mean': float(g.mean()), 'sd': float(g.std()), 'min': float(g.min()), 'max': float(g.max()),
            'pos_pct': float((g > 0).mean() * 100), 'first': float(g.iloc[0]), 'last': float(g.iloc[-1])}


def monthly_series(D):
    m = D['monthly']
    return {'dates': [f'{d:%Y-%m}' for d in m.index], 'bi': m['BI_Rate'].round(4).tolist(),
            'lf': m['Lending_Facility'].round(4).tolist(), 'df': m['Deposit_Facility'].round(4).tolist(),
            'tbp': m['LPS_Rate'].round(4).tolist(), 'dep': m[TARGET].round(4).tolist()}


def corridor_series(D):
    m = D['monthly']
    cw = (m['Lending_Facility'] - m['Deposit_Facility']).replace(0, np.nan)
    return ((m['LPS_Rate'] - m['Deposit_Facility']) / cw).round(4).tolist()

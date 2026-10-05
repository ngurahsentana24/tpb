"""Menyusun data/paper/results.json (mode "Hasil Paper") dari angka yang dilaporkan dalam paper & lampiran,
serta proyeksi harian ARIMA-CEEMDAN-LSTM M1/M2/M3 (scripts/paper_src). Tidak ada training di sini.

Jalankan:  python scripts/build_paper_results.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.simulation import Simulator, ceil_grid  # noqa: E402

SRC = os.path.join(ROOT, 'scripts', 'paper_src')
OUT = os.path.join(ROOT, 'data', 'paper')
os.makedirs(OUT, exist_ok=True)

MONTHS = ['Okt 2026', 'Nov 2026', 'Des 2026', 'Jan 2027', 'Feb 2027', 'Mar 2027']
TBP, BI, DEP_LAST = 3.75, 5.75, 4.3080
HYB = 'Deposito_ARIMA-CEEMDAN-LSTM'
MODELS = ['CEEMDAN-LSTM', 'LSTM', 'ARIMA', 'Linear Trend', 'Rolling Mean (30)']


def m(rmse, mae, mape, r2=None, dacc=None, chg=None, fix=None, n=None):
    return {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2oos': r2, 'dir_acc': dacc, 'rmse_chg': chg, 'rmse_fix': fix,
            'n': n, 'n_chg': None, 'pt_p': None}


# ------------------------------------------------------------------ Bab 4.1 deskriptif (Tabel 1, 2, A5, A6)
desc = [dict(var='BI Rate', mean=4.93, sd=0.91, min=3.50, max=6.25, skew=-0.25, kurt=-1.29, fixed_pct=98.5, n=2356),
        dict(var='Lending Facility', mean=5.68, sd=0.91, min=4.25, max=7.00, skew=-0.25, kurt=-1.29, fixed_pct=98.5, n=2356),
        dict(var='Deposit Facility', mean=4.15, sd=0.91, min=2.75, max=5.50, skew=-0.19, kurt=-1.32, fixed_pct=98.5, n=2356),
        dict(var='TBP LPS', mean=4.84, sd=1.17, min=3.50, max=7.00, skew=0.48, kurt=-1.29, fixed_pct=98.9, n=2344),
        dict(var='Deposito 1M', mean=3.92, sd=0.62, min=2.95, max=5.68, skew=0.90, kurt=0.65, fixed_pct=0.0, n=1626)]
stat = [dict(var='BI Rate', adf_p_lvl=0.410, kpss_p_lvl=0.010, adf_p_dif=0.0005, kpss_p_dif=0.100, bds='−1,53 s.d. −0,73', bds_p='0,125–0,463'),
        dict(var='Lending Facility', adf_p_lvl=0.412, kpss_p_lvl=0.010, adf_p_dif=0.0005, kpss_p_dif=0.100, bds='−1,54 s.d. −0,74', bds_p='0,124–0,462'),
        dict(var='Deposit Facility', adf_p_lvl=0.382, kpss_p_lvl=0.010, adf_p_dif=0.0005, kpss_p_dif=0.100, bds='−1,38 s.d. −0,74', bds_p='0,169–0,462'),
        dict(var='TBP LPS', adf_p_lvl=0.807, kpss_p_lvl=0.010, adf_p_dif=0.0005, kpss_p_dif=0.100, bds='−1,13 s.d. −0,54', bds_p='0,260–0,587'),
        dict(var='Deposito 1M', adf_p_lvl=0.071, kpss_p_lvl=0.010, adf_p_dif=0.001, kpss_p_dif=0.010, bds='6,19 s.d. 11,21', bds_p='<0,001')]
cycles = [dict(name='Pelonggaran 2017', dir=-1, start='2017-08-22', end='2018-05-16', n=2, total_bp=-50),
          dict(name='Pengetatan 2018', dir=1, start='2018-05-17', end='2019-07-17', n=6, total_bp=175),
          dict(name='Pelonggaran 2019–2021', dir=-1, start='2019-07-18', end='2022-08-22', n=10, total_bp=-250),
          dict(name='Pengetatan 2022–2024', dir=1, start='2022-08-23', end='2024-09-17', n=8, total_bp=275),
          dict(name='Pelonggaran 2024–2025', dir=-1, start='2024-09-18', end='2026-05-19', n=6, total_bp=-150),
          dict(name='Pengetatan 2026', dir=1, start='2026-05-20', end='2026-09-30', n=3, total_bp=100)]
gap = dict(mean=-24.64, sd=25.52, min=-175.0, max=55.80, pos_pct=18.07, first=-61.61, last=55.80)

# ------------------------------------------------------------------ Bab 4.2 dekomposisi (Tabel A8/A9)
CAT = {'N': ('Derau (< 1 bulan)', 'Diabaikan dalam penetapan TBP'),
       'R': ('Siklus RDG (1–4 bulan)', 'Sinyal jangka pendek; dipantau'),
       'M': ('Siklus menengah (≥ 4 bulan)', 'Sinyal yang perlu direspons TBP'),
       'T': ('Tren jangka panjang', 'Jangkar level TBP')}


def drows(spec):
    out = []
    for comp, per, var, cor, c in spec:
        out.append(dict(comp=comp, period_m=per, var_pct=var, corr=cor, category=CAT[c][0], implication=CAT[c][1]))
    return out


dec_level = drows([('IMF1', 0.137, 0.180, 0.081, 'N'), ('IMF2', 0.194, 0.009, 0.051, 'N'), ('IMF3', 0.163, 0.078, 0.025, 'N'),
                   ('IMF4', 0.350, 0.067, 0.025, 'N'), ('IMF5', 0.790, 0.080, 0.011, 'N'), ('IMF6', 2.212, 0.170, 0.086, 'R'),
                   ('IMF7', 8.603, 0.721, 0.319, 'M'), ('IMF8', 38.714, 35.113, 0.265, 'M'), ('Residual', None, 63.582, 0.663, 'T')])
dec_gap = drows([('IMF1', 0.141, 1.611, 0.140, 'N'), ('IMF2', 0.230, 0.150, 0.120, 'N'), ('IMF3', 0.211, 0.508, 0.127, 'N'),
                 ('IMF4', 0.493, 0.712, 0.144, 'N'), ('IMF5', 1.291, 1.239, 0.272, 'R'), ('IMF6', 4.842, 3.870, 0.307, 'M'),
                 ('IMF7', 12.913, 8.544, 0.454, 'M'), ('IMF8', 38.738, 28.750, 0.220, 'M'), ('Residual', None, 54.618, 0.639, 'T')])
decomposition = {
    'target': dict(label='Deposito 1M (spesifikasi level)', dates=None, orig=None, comps=None, table=dec_level, io=0.023,
                   figure='paper/g_ceemdan.png'),
    'gap': dict(label='Gap Deposito − TBP', dates=None, orig=None, comps=None, table=dec_gap, io=-0.427,
                figure='paper/g_gap_decomp.png'),
    'coherence': [  # Tabel 4
        dict(comp='Frekuensi tinggi (IMF1–IMF6)', c0=0.02, lag=70, copt=0.12, dir='TBP mendahului'),
        dict(comp='IMF6', c0=0.01, lag=84, copt=0.21, dir='TBP mendahului'),
        dict(comp='IMF7', c0=0.17, lag=96, copt=0.31, dir='TBP mendahului'),
        dict(comp='IMF8', c0=0.23, lag=-126, copt=0.38, dir='Deposito mendahului'),
        dict(comp='Residual', c0=0.64, lag=126, copt=0.66, dir='TBP mendahului'),
        dict(comp='Frekuensi rendah (IMF7–IMF8 + residual)', c0=0.93, lag=-5, copt=0.93, dir='Deposito mendahului')],
    'turning': dict(median_lead_days=40, signals=8, followed=2, figure='paper/g_turning.png')}

# ------------------------------------------------------------------ Bab 4.3–4.4 evaluasi (Tabel 6, 7, 8, 9)
cv = {'CEEMDAN-LSTM': m(4.08, 3.15, 0.818, 0.307, 71.6, 4.420, 1.959, 349),
      'LSTM': m(4.19, 3.26, 0.844, 0.270, 69.1, 4.492, 2.413, 349),
      'ARIMA': m(4.90, 3.80, 0.985, 0.0, None, 5.417, 0.632, 349),
      'Linear Trend': m(4.84, 3.70, 0.961, 0.025, 67.4, None, None, 349),
      'Rolling Mean (30)': m(6.32, 4.78, 1.251, -0.661, 66.0, None, None, 349)}
ho = {'CEEMDAN-LSTM': m(5.44, 4.43, 1.080, None, 66.7, n=64), 'LSTM': m(5.16, 4.28, 1.039, None, 63.2, n=64),
      'ARIMA': m(5.54, 4.49, 1.096, 0.0, None, n=64), 'Linear Trend': m(7.11, 5.88, 1.434, None, 59.6, n=64),
      'Rolling Mean (30)': m(10.67, 8.68, 2.123, None, 54.4, n=64)}
FR = {'CEEMDAN-LSTM': [5.03, 4.45, 4.46, 3.32, 2.96, 3.92], 'LSTM': [4.78, 4.42, 4.69, 3.68, 2.84, 4.36],
      'ARIMA': [6.25, 5.44, 5.73, 4.05, 3.34, 3.86], 'Linear Trend': [5.34, 4.45, 4.63, 3.92, 2.79, 6.92],
      'Rolling Mean (30)': [4.88, 4.19, 5.86, 6.63, 3.57, 10.21]}
FM = {'CEEMDAN-LSTM': [3.96, 3.55, 3.40, 2.79, 2.27, 2.97], 'LSTM': [3.82, 3.56, 3.72, 3.03, 2.09, 3.28],
      'ARIMA': [4.79, 4.10, 4.60, 3.43, 2.67, 3.13], 'Linear Trend': [4.43, 3.35, 3.88, 3.13, 2.05, 5.29],
      'Rolling Mean (30)': [3.94, 3.24, 4.90, 5.75, 2.94, 7.55]}
FP = {'CEEMDAN-LSTM': [0.980, 0.881, 0.862, 0.754, 0.635, 0.793], 'LSTM': [0.945, 0.884, 0.943, 0.819, 0.583, 0.874],
      'ARIMA': [1.185, 1.019, 1.167, 0.930, 0.745, 0.840], 'Linear Trend': [1.100, 0.832, 0.990, 0.847, 0.572, 1.403],
      'Rolling Mean (30)': [0.980, 0.803, 1.256, 1.560, 0.822, 1.992]}
QN = ['2025Q1', '2025Q2', '2025Q3', '2025Q4', '2026Q1', '2026Q2']
NT = [58, 51, 64, 64, 55, 57]
LAM = [0.7, 0.7, 0.9, 1.0, 0.5, 0.7]
folds = [dict(name=f'Fold {k + 1} {QN[k]}', holdout=False, **{'lambda': LAM[k]}, n_test=NT[k],
              metrics={mm: m(FR[mm][k], FM[mm][k], FP[mm][k]) for mm in MODELS}) for k in range(6)]
folds.append(dict(name='Holdout 2026Q3', holdout=True, **{'lambda': 0.1}, n_test=64,
                  metrics={mm: m(ho[mm]['rmse'], ho[mm]['mae'], ho[mm]['mape']) for mm in MODELS}))
dm = [dict(pair='CEEMDAN-LSTM vs ARIMA', dm_se=-5.45, p_se=0.0005, dm_ae=-6.56, p_ae=0.0005),
      dict(pair='CEEMDAN-LSTM vs LSTM', dm_se=-1.44, p_se=0.150, dm_ae=-1.72, p_ae=0.086)]
horizon = [dict(h='Satu langkah', cl=4.1, ar=4.9, pi=8.0, big=0.6, grid_cl=84.5, grid_ar=82.2),
           dict(h='≤ 1 bulan', cl=7.0, ar=9.1, pi=13.8, big=7.1, grid_cl=73.8, grid_ar=73.8),
           dict(h='1–2 bulan', cl=10.9, ar=11.9, pi=21.3, big=33.3, grid_cl=56.3, grid_ar=56.3),
           dict(h='2–3 bulan', cl=18.4, ar=18.1, pi=36.1, big=64.9, grid_cl=36.1, grid_ar=36.1)]
evaluation = dict(models=MODELS, cv=cv, ho=ho, folds=folds, sample=None, dm=dm, horizon=horizon,
                  why_hybrid=[
                      'RMSE rolling-origin 4,08 bp vs ARIMA 4,90 bp (−16,7%) dan vs LSTM murni 4,19 bp; uji DM vs ARIMA signifikan pada α = 1%.',
                      'Basis ARIMA(0,1,0) menjaga ramalan tetap stabil pada hari tanpa perubahan; koreksi LSTM atas co-IMF hanya ditambahkan sebesar λ (aturan 1-SE) sehingga tidak overfit.',
                      'CEEMDAN memisahkan derau harian dari siklus menengah dan tren, sehingga LSTM belajar dari sinyal yang relevan dengan horizon kebijakan.',
                      'Pada holdout 2026Q3 keunggulan menyempit (λ = 0,1): hibrida mendekati ARIMA, perilaku yang diharapkan ketika sinyal koreksi lemah.'])

# ------------------------------------------------------------------ Bab 4.5 proyeksi (Tabel 10, C1, C10)
specs = {'level': 'M1', 'gap': 'M2', 'spread': 'M3'}
PB = {'level': [100, 97.7, 93.7, 89.5, 86.3, 83.3], 'gap': [100, 98, 95, 91, 87, 85], 'spread': [100, 98, 95, 90, 87, 85]}
TMIN = {'level': [4.50, 4.75, 4.75, 4.75, 4.75, 5.00], 'gap': [4.50, 4.75, 4.75, 5.00, 5.00, 5.00],
        'spread': [4.50, 4.75, 4.75, 4.75, 5.00, 5.00]}
D = {k: pd.read_csv(os.path.join(SRC, f'proyeksi_6_bulan_{v}.csv'), parse_dates=['date']) for k, v in specs.items()}
dates = D['level']['date'].dt.strftime('%Y-%m-%d').tolist()
per = D['level']['date'].dt.to_period('M')
days = per.value_counts().sort_index().tolist()
fseries, fmonthly = {}, {}


def monthly(df, col):
    g = df.groupby(df['date'].dt.to_period('M'))
    return g[col].mean().values, g[col + '_PI95_lo'].mean().values, g[col + '_PI95_hi'].mean().values


for spec in specs:
    df = D[spec]
    key = f'CEEMDAN-LSTM ({spec})'
    fseries[key] = dict(label=f'CEEMDAN-LSTM · spesifikasi {spec}', pred=df[HYB].round(4).tolist(),
                        lo=df[HYB + '_PI95_lo'].round(4).tolist(), hi=df[HYB + '_PI95_hi'].round(4).tolist())
    v, lo, hi = monthly(df, HYB)
    fmonthly[key] = [dict(v=round(float(v[h]), 3), lo=round(float(lo[h]), 3), hi=round(float(hi[h]), 3),
                          pb=PB[spec][h] / 100, tbpmin=TMIN[spec][h]) for h in range(6)]
for name in ('LSTM', 'ARIMA'):  # pembanding dari run yang sama (M1), P(breach) aproksimasi normal
    df, col = D['level'], f'Deposito_{name}'
    fseries[name] = dict(label=name, pred=df[col].round(4).tolist(), lo=df[col + '_PI95_lo'].round(4).tolist(),
                         hi=df[col + '_PI95_hi'].round(4).tolist())
    v, lo, hi = monthly(df, col)
    from scipy.stats import norm
    sd = (hi - lo) / 3.92
    fmonthly[name] = [dict(v=round(float(v[h]), 3), lo=round(float(lo[h]), 3), hi=round(float(hi[h]), 3),
                           pb=round(float(1 - norm.cdf((TBP - v[h]) / sd[h])), 3),
                           tbpmin=float(ceil_grid(v[h] + 1.2816 * sd[h]))) for h in range(6)]
forecast = dict(dates=dates, series=fseries, monthly=fmonthly, main='CEEMDAN-LSTM (level)', months=MONTHS, days=days,
                **{'lambda': 0.1}, tbp=TBP, bi=BI, dep_last=DEP_LAST, last_date='2026-09-30', figure='paper/g_forecast.png',
                note='P(breach) LSTM & ARIMA pada mode paper adalah aproksimasi normal dari PI 95% (paper hanya melaporkan '
                     'P(breach) CEEMDAN-LSTM).')

# ------------------------------------------------------------------ Bab 4.7 transmisi (Tabel 11, C3, C5, C9)
transmission = dict(
    ardl=[dict(path='BI Rate → TBP LPS', p=1, q=4, n=76, F=6.994, coint=True, inconclusive=False, theta=0.412, se=0.184,
               p_theta=0.025, g0=0.008, alpha=-0.075, p_alpha=0.002, hl=8.927, r2=0.453),
          dict(path='BI Rate → Deposito 1M', p=2, q=2, n=76, F=9.343, coint=True, inconclusive=False, theta=0.643, se=0.216,
               p_theta=0.003, g0=0.154, alpha=-0.033, p_alpha=0.0006, hl=20.809, r2=0.853),
          dict(path='TBP LPS → Deposito 1M', p=1, q=0, n=76, F=2.740, coint=False, inconclusive=False, theta=0.241, se=0.604,
               p_theta=0.690, g0=0.213, alpha=-0.029, p_alpha=0.173, hl=23.339, r2=0.770)],
    nardl=[dict(path='BI Rate → TBP LPS', p=4, q=4, n=76, F=3.846, coint=False, inconclusive=True, theta_pos=0.404,
                theta_neg=0.743, lr_F=1.195, lr_p=0.279, alpha=-0.103, hl=6.354),
           dict(path='BI Rate → Deposito 1M', p=2, q=3, n=76, F=5.268, coint=True, inconclusive=False, theta_pos=0.887,
                theta_neg=0.244, lr_F=2.855, lr_p=0.096, alpha=-0.019, hl=36.548)],
    multipliers=dict(ardl=[0.154, 0.170, 0.186, 0.201, 0.215, 0.229], nardl_pos=[0.155, 0.169, 0.183, 0.196, 0.209, 0.222],
                     nardl_neg=[0.036, 0.040, 0.044, 0.048, 0.051, 0.055]),
    figure='paper/g_passthrough.png')

# ------------------------------------------------------------------ Bab 4.8 efektivitas (Tabel 12)
E = [('Seluruh sampel 2020–2026', '2020-01-02', '2026-09-30', 1627, -24.6, 18.1, 15, 55.8, 0.07, -3.33, -1.77),
     ('Pelonggaran 2019–2021', '2020-01-02', '2022-08-22', 643, -37.8, 1.6, 4, 13.3, 0.91, 1.83, 1.78),
     ('Pengetatan 2022–2024', '2022-08-23', '2024-09-17', 499, -35.0, 0.0, 0, 0.0, -0.55, 0.30, 0.43),
     ('Pelonggaran 2024–2025', '2024-09-18', '2026-05-19', 395, -3.4, 49.1, 11, 35.0, -0.36, 0.60, 0.32),
     ('Pengetatan 2026', '2026-05-20', '2026-09-30', 90, 33.9, 100.0, 1, 55.8, -0.57, 0.50, 1.34),
     ('Proyeksi (TBP ditahan)', '2026-10-01', '2027-03-31', 130, 49.1, 91.5, 1, 54.5, -0.57, None, None)]
K = ['period', 'start', 'end', 'days', 'gap', 'breach', 'episodes', 'depth', 'corridor', 'tbp_bi', 'dep_bi']
effectiveness = dict(rows=[dict(zip(K, r)) for r in E], dep_end=4.238,
                     figures=['paper/g_gap.png', 'paper/g_corridor.png', 'paper/g_timeline.png'])

# ------------------------------------------------------------------ Bab 4.9 simulasi (Tabel 13, 14, C10, C11, C12)
# Jalur dasar bootstrap tidak dipublikasikan; direkonstruksi dari rata-rata & PI 95% bulanan M1 (approx = True).
v, lo, hi = monthly(D['level'], HYB)
sd = (hi - lo) / 3.92
rng = np.random.RandomState(42)
W = np.cumsum(rng.standard_normal((1000, 6)), axis=1)
W = (W - W.mean(0)) / W.std(0)
base_paths = v[None, :] + sd[None, :] * W
np.save(os.path.join(OUT, 'base_paths.npy'), base_paths)
context = dict(bi_last=BI, tbp_last=TBP, dep_last=DEP_LAST, sp_lf=0.75, sp_df=-1.0, spread_tbp_bi=-1.25,
               multipliers=transmission['multipliers'], cir=dict(kappa=0.005, theta=6.6, sigma=0.075), months=MONTHS)
sim = Simulator(dict(context, base_paths=base_paths))
baseline = sim.run(0, 'C', 0, 1, 'ardl', 10, 1000, False, 42, with_heat=True)
# angka paper menggantikan aproksimasi untuk baris tabel baseline (Tabel C10)
C10 = [(4.26, 100.0, 4.50), (4.24, 97.7, 4.75), (4.24, 93.7, 4.75), (4.24, 89.5, 4.75), (4.24, 86.3, 4.75), (4.24, 83.3, 5.00)]
for h, (dep, p, tm) in enumerate(C10):
    baseline['table'][h].update(dep=dep, pb=p / 100, tbp_min=tm, gap=(dep - TBP) * 100)
baseline['pb'] = [r[1] / 100 for r in C10]
baseline['tbp_min'] = [r[2] for r in C10]
baseline.update(pmax=1.0, pmax_month='Okt 2026', tbp_min_max=5.0, tbp_min_month='Mar 2027', adj_needed_bp=125.0)
baseline['heat'] = dict(shocks=[-50, -25, 0, 25, 50, 'CIR'], rules=['A', 'C', 'D'],
                        pmax=[[1, 1, .016], [1, 1, .019], [1, 1, .022], [1, 1, .022], [1, 1, .020], [1, 1, .025]])
MX = [(-50, (.582, 1), (.773, 1), (.016, .016), 4.75), (-25, (.425, 1), (.806, 1), (.019, .019), 4.75),
      (0, (.272, 1), (.833, 1), (.019, .022), 5.00), (25, (.153, 1), (.860, 1), (.019, .022), 5.00),
      (50, (.067, 1), (.892, 1), (.017, .020), 5.00), ('CIR', (.299, 1), (.829, 1), (.018, .025), 5.00)]
matrix = [dict(shock=s, A=dict(end=a[0], max=a[1]), C=dict(end=c[0], max=c[1]), D=dict(end=d[0], max=d[1]), tbp_min_end=t)
          for s, a, c, d, t in MX]
rules = [dict(rule='TBP aktual', breach=17.3, gap=-24.7, changes=23, rec=3.75, delta=0),
         dict(rule='Aturan estimasi (Dep + BI)', breach=7.4, gap=-31.6, changes=13, rec=4.50, delta=75),
         dict(rule='Margin +0 bp di atas Deposito', breach=17.3, gap=-19.2, changes=14, rec=4.25, delta=50),
         dict(rule='Margin +25 bp di atas Deposito', breach=6.2, gap=-43.0, changes=14, rec=4.50, delta=75),
         dict(rule='Margin +50 bp di atas Deposito', breach=1.2, gap=-66.7, changes=14, rec=4.75, delta=100)]
replay = [dict(episode='Pengetatan 2022–2024', start='Agu 2022', resp='TBP ditahan', dbi=209.5, ddep=67.0, dep_end=4.98,
               tbp_end=3.75, gap_end=122.8, pmax=1.0, tbp_min=5.75),
          dict(episode='Pengetatan 2022–2024', start='Agu 2022', resp='TBP mengikuti pola historis', dbi=209.5, ddep=67.0,
               dep_end=4.98, tbp_end=4.00, gap_end=97.8, pmax=1.0, tbp_min=5.75)]
simulation = dict(context=context, baseline=baseline, matrix=matrix, approx=True, rules=rules, replay=replay,
                  figure='paper/g_heatmap.png')

# ------------------------------------------------------------------ Bab 5 rekomendasi (Tabel 15, C13)
recommendations = dict(
    main=dict(tbp_now=4.50, adj_now_bp=75, tbp_peak=5.00, adj_peak_bp=125, peak_month='Mar 2027'),
    rows=[['TBP ke 4,50% (+75 bp) pada penetapan terdekat; ke 4,75–5,00% bila P(breach) proyeksi masih > 10%',
           'Gap +56 bp; P(breach) dengan TBP ditahan 83–85%', 'TBP minimum 4,50–5,00% (Tabel 10, 13; C10)', 'Jangka pendek'],
          ['Aturan kalibrasi berbasis estimasi deposito dan BI Rate dengan margin 0–25 bp',
           'Breach historis turun dari 17,3% menjadi 6,2–7,4% bulan', 'Tabel 14', 'Menengah'],
          ['Kajian TBP selaras dengan RDG BI; respons lebih cepat pada siklus pengetatan',
           'γ₀ BI→Deposito 0,154 vs BI→TBP 0,008; θ⁺ NARDL 0,89 vs 0,40', 'Tabel 11; C3', 'Jangka pendek–menengah'],
          ['P(breach) > 10% dan sinyal titik balik sebagai pemicu kajian',
           'Median lead time 40 hari; 2 dari 8 sinyal diikuti perubahan TBP', 'Gambar 6', 'Operasional'],
          ['Ramalan titik untuk satu siklus penetapan; distribusi ramalan untuk horizon lebih panjang',
           'PI 95% ±8 bp (satu langkah) vs ±36 bp (2–3 bulan)', 'Tabel 9', 'Operasional'],
          ['CEEMDAN-LSTM tiga spesifikasi sebagai perangkat pemantauan kuartalan',
           'RMSE −16,7% vs ARIMA; DM signifikan α = 1%', 'Tabel 6, 8', 'Operasional'],
          ['Uji ketahanan DPS terhadap pengetatan ekstrem', 'Replay 2022–2024: gap akhir +98 hingga +123 bp',
           'TBP minimum 5,75% (Tabel C12)', 'Stres']])

R = dict(
    descriptive=dict(table=desc, stationarity=stat, gap=gap, cycles=cycles, figures=['paper/g_series.png', 'paper/g_gap.png']),
    series=None, decomposition=decomposition, evaluation=evaluation, forecast=forecast, transmission=transmission,
    effectiveness=effectiveness, simulation=simulation, recommendations=recommendations,
    meta=dict(source='paper', created='Paper LPS 2026', data_start='2017-01-03', data_end='2026-09-30', n_days=1626,
              n_months=81, spec='level', spec_label='Tiga spesifikasi (level · gap · spread)', engine='LSTM (Bayesian optimization)',
              config=None, checks=[], notes=[
                  'Angka diambil dari paper dan lampiran (hasil olahan penulis); tidak dihitung ulang di server.',
                  'Simulator interaktif pada mode paper memakai jalur dasar yang direkonstruksi dari PI 95% bulanan '
                  '(aproksimasi); tabel & matriks resmi tetap angka paper.']),
    kpi=dict(date_last='30 Sep 2026', dep_last=DEP_LAST, dep_prev_month=None, tbp_last=TBP, bi_last=BI, gap_last_bp=55.8,
             pbreach_end=0.833))
json.dump(R, open(os.path.join(OUT, 'results.json'), 'w'), ensure_ascii=False)
print('ok', os.path.getsize(os.path.join(OUT, 'results.json')) / 1e3, 'KB · approx baseline pb',
      np.round(sim.run(0, 'C', with_heat=False)['pb'], 3))

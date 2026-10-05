"""Pipeline training ulang end-to-end (mengikuti Bab 3–4 paper, versi ringan yang bisa dijalankan di server web).

Tahapan: validasi data → statistik → CEEMDAN deskriptif → dekomposisi kausal → rolling-origin per kuartal
(CEEMDAN-LSTM berbasis ARIMA, LSTM murni, ARIMA, pembanding naif) → uji DM → proyeksi 6 bulan + PI bootstrap
→ transmisi ARDL/NARDL & efektivitas TBP → simulasi skenario & rekomendasi. Hasil disimpan sebagai JSON."""
import json
import os
import time
import traceback
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import io as dio, stats as dst, decomp as dcp, models as mdl, evaluation as ev, transmission as trn
from .simulation import Simulator, ceil_grid, round_grid

DEFAULTS = dict(spec='level', n_folds=6, holdout=True, window=252, K=6, decomp_causal='emd', trials=20,
                lookback=7, engine='auto', units=32, epochs=60, n_seeds=3, val_es=42, val_cal=63, n_train=750,
                horizon_months=6, B=1000, corr_horizon=21, seed=42)
STEPS = ['Validasi & penyelarasan data', 'Statistik deskriptif & uji stasioneritas', 'CEEMDAN deskriptif',
         'Dekomposisi kausal (jendela bergulir)', 'Rolling-origin: pelatihan & evaluasi', 'Uji Diebold-Mariano',
         'Proyeksi 6 bulan + PI bootstrap', 'Transmisi ARDL/NARDL & efektivitas TBP', 'Simulasi skenario & rekomendasi']
SPEC_LABEL = {'level': 'Spesifikasi level (Deposito 1M)', 'gap': 'Spesifikasi gap (Deposito − TBP)',
              'spread': 'Spesifikasi spread (Deposito − BI Rate)'}
MODELS = ['CEEMDAN-LSTM', 'LSTM', 'ARIMA', 'Linear Trend', 'Rolling Mean (30)']
MON = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']


def _mlab(d):
    return f'{MON[d.month - 1]} {d.year}'


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


class Pipeline:
    def __init__(self, data_path, out_dir, cfg=None, log=None, progress=None):
        self.path, self.out = data_path, out_dir
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self._log = log or (lambda m: print(m))
        self._prog = progress or (lambda step, pct: None)
        self.R = {}
        os.makedirs(out_dir, exist_ok=True)

    def log(self, m):
        self._log(m)

    def step(self, i, frac=0.0):
        self._prog(i, (i + frac) / len(STEPS))

    # ------------------------------------------------------------------ utama
    def run(self):
        t0 = time.time()
        c = self.cfg
        self.step(0)
        D, checks = dio.load(self.path)
        if D is None:
            raise ValueError('; '.join(x[1] for x in checks if x[0] == 'err'))
        self.D = D
        al = D['aligned']
        self.dates = al.index
        Dep = al[dio.TARGET].values
        bi, tbp = al['BI_Rate'].values, al['LPS_Rate'].values
        self.Dep, self.bi, self.tbp = Dep, bi, tbp
        P = {'level': np.zeros_like(Dep), 'gap': tbp, 'spread': bi}[c['spec']]
        self.P = P
        self.s = Dep - P
        self.exog = np.column_stack([bi, tbp])
        self.log(f'{len(al):,} hari selaras · {al.index[0]:%d %b %Y} – {al.index[-1]:%d %b %Y} · {len(D["monthly"])} bulan'.replace(',', '.'))
        self.log(f"Spesifikasi: {SPEC_LABEL[c['spec']]} · mesin jaringan saraf: {mdl.engine_name(c['engine']).upper()}")

        self.step(1)
        cyc = dst.cycles(D['ff']['BI_Rate'].dropna())
        self.R['descriptive'] = {'table': dst.descriptive(D), 'stationarity': dst.stationarity(D),
                                 'gap': dst.gap_stats(D), 'cycles': cyc}
        self.R['series'] = {'monthly': dst.monthly_series(D), 'corridor': dst.corridor_series(D)}
        self.log(f"ADF/KPSS selesai · {len(cyc)} siklus BI teridentifikasi · gap terakhir {self.R['descriptive']['gap']['last']:+.1f} bp")

        self.step(2)
        self._descriptive_decomp()

        self.step(3)
        self._causal()

        self.step(4)
        self._folds()

        self.step(5)
        self._dm()

        self.step(6)
        self._projection()

        self.step(7)
        self._transmission_effect(cyc)

        self.step(8)
        self._simulation()
        self._recommendations()

        self.R['meta'] = {'source': 'run', 'created': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                          'data_start': f'{al.index[0]:%Y-%m-%d}', 'data_end': f'{al.index[-1]:%Y-%m-%d}',
                          'n_days': len(al), 'n_months': len(D['monthly']), 'spec': c['spec'],
                          'spec_label': SPEC_LABEL[c['spec']], 'engine': mdl.engine_name(c['engine']),
                          'config': c, 'duration_s': round(time.time() - t0, 1), 'checks': checks,
                          'notes': self._notes()}
        g = (Dep[-1] - tbp[-1]) * 100
        self.R['kpi'] = {'date_last': f'{al.index[-1]:%d %b %Y}', 'dep_last': float(Dep[-1]),
                         'dep_prev_month': float(D['monthly'][dio.TARGET].iloc[-2]) if len(D['monthly']) > 1 else None,
                         'tbp_last': float(tbp[-1]), 'bi_last': float(bi[-1]), 'gap_last_bp': float(g),
                         'pbreach_end': self.R['forecast']['monthly']['CEEMDAN-LSTM'][-1]['pb']}
        res = _clean(self.R)
        with open(os.path.join(self.out, 'results.json'), 'w') as f:
            json.dump(res, f)
        np.save(os.path.join(self.out, 'base_paths.npy'), self.base_paths)
        self._prog(len(STEPS), 1.0)
        self.log(f'Selesai dalam {time.time() - t0:.0f} detik')
        return res

    def _notes(self):
        c = self.cfg
        n = [f"Dekomposisi kausal memakai {'CEEMDAN' if c['decomp_causal'] == 'ceemdan' else 'EMD'} pada jendela "
             f"{c['window']} hari (CEEMDAN deskriptif tetap dipakai untuk tab Dekomposisi)."]
        if mdl.engine_name(c['engine']) != 'lstm':
            n.append('Jaringan saraf memakai MLP (scikit-learn) — dipilih pengguna atau PyTorch tidak terpasang. '
                     'Pasang requirements-full.txt agar memakai LSTM.' if c['engine'] != 'mlp' else
                     'Jaringan saraf memakai MLP (scikit-learn) sesuai pilihan konfigurasi.')
        n.append('Hiperparameter tetap (tanpa Bayesian optimization) agar training ulang dapat berjalan di server web.')
        return n

    # ------------------------------------------------------------------ dekomposisi
    def _descriptive_decomp(self):
        c = self.cfg
        out = {}
        for key, x, lab in (('target', self.s, SPEC_LABEL[c['spec']]),
                            ('gap', self.Dep - self.tbp, 'Gap Deposito − TBP')):
            if key == 'gap' and c['spec'] == 'gap':
                out['gap'] = out['target']
                continue
            imfs, res = dcp.decompose(x, 'ceemdan', trials=max(20, c['trials']), max_imf=8, seed=c['seed'])
            rows, io_ = dcp.describe(x, imfs, res)
            st = max(1, len(x) // 400)
            out[key] = {'label': lab, 'dates': [f'{d:%Y-%m-%d}' for d in self.dates[::st]], 'orig': x[::st].tolist(),
                        'comps': [cc[::st].tolist() for cc in list(imfs) + [res]], 'table': rows, 'io': io_}
            self.log(f'CEEMDAN {lab}: {imfs.shape[0]} IMF + residual · IO = {io_:.3f}')
        self.R['decomposition'] = out

    def _causal(self):
        c = self.cfg
        n = len(self.s)
        self.log(f"Dekomposisi kausal y[t−{c['window']}:t] untuk {n - c['window']:,} titik…".replace(',', '.'))
        self.F = dcp.causal_tails(self.s, c['window'], c['K'], c['decomp_causal'], c['trials'], c['seed'],
                                  progress=lambda f: self._prog(3, (3 + f) / len(STEPS)))
        self.t_min = c['window'] - 1 + c['lookback'] + 1
        self.log('Audit kebocoran: fitur pada t hanya memakai y[t−W+1 : t] → selisih dengan data terpotong = 0')

    # ------------------------------------------------------------------ rolling-origin
    def _quarters(self):
        c = self.cfg
        idx = self.dates
        q = idx.to_period('Q')
        uq = q.unique()
        last_end = uq[-1].end_time.normalize()
        if (last_end - idx[-1]).days > 5:
            uq = uq[:-1]
        need = c['n_folds'] + (1 if c['holdout'] else 0)
        start_min = self.t_min + c['val_es'] + c['val_cal'] + 120
        sel = [p for p in uq if np.searchsorted(idx, p.start_time) >= start_min][-need:]
        out = []
        for p in sel:
            o = int(np.searchsorted(idx, p.start_time))
            e = int(np.searchsorted(idx, p.end_time + pd.Timedelta(seconds=1)))
            out.append((str(p), o, e))
        return out

    def _labels(self, fit_end):
        c = self.cfg
        seg = self.s[max(0, fit_end - 500):fit_end]
        lab, se, per = dcp.co_imf_labels(seg, c['K'], c['decomp_causal'], c['trials'], 3, c['seed'])
        return lab

    def _train_models(self, o, e_idx):
        """Melatih CEEMDAN-LSTM (koreksi) dan LSTM murni dengan origin o; memprediksi indeks e_idx."""
        c = self.cfg
        L = c['lookback']
        cal0 = o - c['val_cal']
        es0 = cal0 - c['val_es']
        fit = np.arange(max(self.t_min, es0 - c['n_train']), es0)
        es_idx, cal_idx = np.arange(es0, cal0), np.arange(cal0, o)
        lab = self._labels(es0)
        G = dcp.group_sum(np.nan_to_num(self.F), lab, 3)
        Mh = mdl.build_hybrid_features(self.s, G, self.exog, L)
        sc = mdl.Scaler().fit(Mh[fit])
        Mhs = sc.tf(Mh)
        ds = np.r_[0.0, np.diff(self.s)]
        r_sd = max(ds[fit].std(), 1e-6)
        net = mdl.SeqRegressor(c['engine'], units=c['units'], epochs=c['epochs'],
                               seeds=[c['seed'] + k for k in range(c['n_seeds'])])
        net.fit(mdl.windows(Mhs, fit - 1, L), ds[fit] / r_sd, mdl.windows(Mhs, es_idx - 1, L), ds[es_idx] / r_sd)
        corr = lambda idx: net.predict(mdl.windows(Mhs, np.asarray(idx) - 1, L)) * r_sd
        lam, lam_raw = mdl.choose_lambda(self.s[cal_idx], self.s[cal_idx - 1], corr(cal_idx))
        # LSTM murni: deret asli + eksogen, tanpa CEEMDAN dan tanpa penyusutan λ (ŷ = y[t−1] + Δ̂)
        Ml = mdl.build_level_features(self.s, self.exog)
        scl = mdl.Scaler().fit(Ml[fit])
        Mls = scl.tf(Ml)
        net2 = mdl.SeqRegressor(c['engine'], units=c['units'], epochs=c['epochs'],
                                seeds=[c['seed'] + 10 + k for k in range(c['n_seeds'])])
        net2.fit(mdl.windows(Mls, fit - 1, L), ds[fit] / r_sd, mdl.windows(Mls, es_idx - 1, L), ds[es_idx] / r_sd)
        lvl = lambda idx: self.s[np.asarray(idx) - 1] + net2.predict(mdl.windows(Mls, np.asarray(idx) - 1, L)) * r_sd
        return dict(lab=lab, G=G, sc=sc, r_sd=r_sd, net=net, lam=lam, lam_raw=lam_raw, corr=corr, net2=net2,
                    scl=scl, lvl=lvl, epochs=net.epochs_used + net2.epochs_used, n_fit=len(fit))

    def _folds(self):
        c = self.cfg
        qs = self._quarters()
        if len(qs) < 2:
            raise ValueError('Data terlalu pendek untuk validasi rolling-origin (butuh ≥ 2 kuartal uji).')
        self.folds = []
        for k, (qname, o, e) in enumerate(qs):
            is_ho = c['holdout'] and k == len(qs) - 1
            T = self._train_models(o, None)
            idx = np.arange(o, e)
            P_prev = self.P[idx - 1]
            preds = {'CEEMDAN-LSTM': self.s[idx - 1] + T['lam'] * T['corr'](idx) + P_prev,
                     'LSTM': T['lvl'](idx) + P_prev,
                     'ARIMA': self.Dep[idx - 1],
                     'Linear Trend': mdl.linear_trend(self.Dep, idx),
                     'Rolling Mean (30)': mdl.rolling_mean(self.Dep, idx)}
            f = {'name': ('Holdout ' if is_ho else f'Fold {k + 1} ') + qname, 'quarter': qname, 'holdout': is_ho,
                 'dates': [f'{d:%Y-%m-%d}' for d in self.dates[idx]], 'actual': self.Dep[idx], 'prev': self.Dep[idx - 1],
                 'preds': preds, 'lambda': T['lam'], 'lambda_raw': T['lam_raw'], 'labels': T['lab'].tolist(),
                 'n_fit': T['n_fit'], 'epochs_median': float(np.median(T['epochs'])) if T['epochs'] else None}
            f['metrics'] = {m: ev.metrics(f['actual'], p, f['prev']) for m, p in preds.items()}
            self.folds.append(f)
            r = f['metrics']
            self.log(f"{f['name']}: RMSE CEEMDAN-LSTM {r['CEEMDAN-LSTM']['rmse']:.2f} · LSTM {r['LSTM']['rmse']:.2f} · "
                     f"ARIMA {r['ARIMA']['rmse']:.2f} bp · λ = {T['lam']:.1f}")
            self._prog(4, (4 + (k + 1) / len(qs)) / len(STEPS))
        self.last_T = T
        cv = [f for f in self.folds if not f['holdout']]
        ho = [f for f in self.folds if f['holdout']]

        def pooled(fs):
            if not fs:
                return None
            a = np.concatenate([f['actual'] for f in fs])
            pv = np.concatenate([f['prev'] for f in fs])
            return {m: ev.metrics(a, np.concatenate([f['preds'][m] for f in fs]), pv) for m in MODELS}

        sample = (ho or cv)[-1]
        self.R['evaluation'] = {
            'models': MODELS, 'cv': pooled(cv), 'ho': pooled(ho),
            'folds': [{'name': f['name'], 'holdout': f['holdout'], 'lambda': f['lambda'], 'lambda_raw': f['lambda_raw'],
                       'n_fit': f['n_fit'], 'n_test': len(f['actual']), 'epochs_median': f['epochs_median'],
                       'labels': f['labels'], 'metrics': f['metrics']} for f in self.folds],
            'sample': {'name': sample['name'], 'dates': sample['dates'], 'actual': sample['actual'],
                       'preds': {m: sample['preds'][m] for m in ('CEEMDAN-LSTM', 'LSTM', 'ARIMA')}}}

    def _dm(self):
        cv = [f for f in self.folds if not f['holdout']] or self.folds
        a = np.concatenate([f['actual'] for f in cv])
        E = {m: a - np.concatenate([f['preds'][m] for f in cv]) for m in MODELS}
        rows = []
        for m1, m2 in (('CEEMDAN-LSTM', 'ARIMA'), ('CEEMDAN-LSTM', 'LSTM'), ('LSTM', 'ARIMA'), ('CEEMDAN-LSTM', 'Linear Trend')):
            dse, pse = ev.dm_test(E[m1], E[m2], loss='SE')
            dae, pae = ev.dm_test(E[m1], E[m2], loss='AE')
            rows.append({'pair': f'{m1} vs {m2}', 'dm_se': dse, 'p_se': pse, 'dm_ae': dae, 'p_ae': pae})
            if dse is not None:
                self.log(f'DM {m1} vs {m2}: SE {dse:+.2f} (p = {pse:.3f})')
        self.R['evaluation']['dm'] = rows
        self.resid = {m: np.concatenate([f['actual'] - f['preds'][m] for f in self.folds]) for m in MODELS}

    # ------------------------------------------------------------------ proyeksi
    def _projection(self):
        c = self.cfg
        n = len(self.s)
        last = self.dates[-1]
        end = (last + pd.offsets.MonthEnd(c['horizon_months'])).normalize()
        fdates = pd.bdate_range(last + pd.Timedelta(days=1), end)
        H = len(fdates)
        T = self._train_models(n, None)
        L, W, K = c['lookback'], c['window'], c['K']
        s_ext = list(self.s)
        F_ext = np.vstack([np.nan_to_num(self.F), np.zeros((H, K + 1))])
        ex_ext = np.vstack([self.exog, np.repeat(self.exog[-1:], H, 0)])
        hyb = []
        for k in range(H):
            t = n + k
            y = np.asarray(s_ext)
            corr = 0.0
            if T['lam'] > 0 and k < c['corr_horizon']:
                Gx = dcp.group_sum(F_ext[:t], T['lab'], 3)
                Mh = mdl.build_hybrid_features(y, Gx, ex_ext[:t], L)
                Xs = T['sc'].tf(Mh)[None, t - L:t]
                corr = float(T['net'].predict(Xs)[0]) * T['r_sd']
            yhat = y[-1] + T['lam'] * corr
            s_ext.append(yhat)
            seg = np.asarray(s_ext)[-W:]
            imfs, res = dcp.decompose(seg, c['decomp_causal'], c['trials'], K, c['seed'])
            imfs, extra = dcp.fit_k(imfs, K)
            F_ext[t, :K], F_ext[t, K] = imfs[:, -1], (res + extra)[-1]
            hyb.append(yhat)
        # LSTM murni rekursif
        l_ext = list(self.s)
        lst = []
        for k in range(H):
            t = n + k
            Ml = mdl.build_level_features(np.asarray(l_ext), ex_ext[:t])
            Xs = T['scl'].tf(Ml)[None, t - L:t]
            yhat = l_ext[-1] + (float(T['net2'].predict(Xs)[0]) * T['r_sd'] if k < c['corr_horizon'] else 0.0)
            l_ext.append(yhat)
            lst.append(yhat)
        Plast = self.P[-1]
        FC = {'CEEMDAN-LSTM': np.asarray(hyb) + Plast, 'LSTM': np.asarray(lst) + Plast,
              'ARIMA': np.full(H, self.Dep[-1])}
        rng = np.random.RandomState(c['seed'])
        mg = pd.Series(np.arange(H), index=fdates).groupby(fdates.to_period('M'))
        self.month_groups = [g.values for _, g in mg]
        months = [_mlab(p.to_timestamp()) for p, _ in mg]
        self.months = months
        tbp = self.tbp[-1]
        out = {'dates': [f'{d:%Y-%m-%d}' for d in fdates], 'series': {}, 'monthly': {}, 'main': 'CEEMDAN-LSTM',
               'months': months, 'lambda': T['lam'], 'tbp': float(tbp), 'bi': float(self.bi[-1]),
               'dep_last': float(self.Dep[-1]), 'last_date': f'{last:%Y-%m-%d}'}
        paths_all = {}
        for m in ('CEEMDAN-LSTM', 'LSTM', 'ARIMA'):
            r = self.resid[m] - self.resid[m].mean()
            draw = rng.randint(0, len(r), (c['B'], H))
            paths = FC[m][None, :] + np.cumsum(r[draw], axis=1)
            paths_all[m] = paths
            lo, hi = np.percentile(paths, 2.5, 0), np.percentile(paths, 97.5, 0)
            out['series'][m] = {'label': m, 'pred': FC[m], 'lo': lo, 'hi': hi}
            rows = []
            for g in self.month_groups:
                pm = paths[:, g].mean(1)
                rows.append({'v': float(FC[m][g].mean()), 'lo': float(lo[g].mean()), 'hi': float(hi[g].mean()),
                             'pb': float((pm > tbp).mean()), 'tbpmin': float(ceil_grid(np.quantile(pm, 0.9)))})
            out['monthly'][m] = rows
        out['days'] = [int(len(g)) for g in self.month_groups]
        self.base_paths = np.column_stack([paths_all['CEEMDAN-LSTM'][:, g].mean(1) for g in self.month_groups])
        self.breach_days = float((paths_all['CEEMDAN-LSTM'] > tbp).mean() * 100)
        self.FC = FC
        self.R['forecast'] = out
        rows = out['monthly']['CEEMDAN-LSTM']
        self.log(f"Proyeksi {months[0]}–{months[-1]}: CEEMDAN-LSTM {FC['CEEMDAN-LSTM'][-1]:.3f}% · "
                 f"P(breach) akhir {rows[-1]['pb'] * 100:.0f}% · λ = {T['lam']:.1f}")

    # ------------------------------------------------------------------ transmisi & efektivitas
    def _transmission_effect(self, cyc):
        M = self.D['monthly']
        tr = trn.analyse(M, dio.TARGET, len(self.month_groups))
        self.R['transmission'] = tr
        for e in tr['ardl']:
            if 'theta' in e:
                self.log(f"ARDL {e['path']}: θ = {e['theta']:.2f} · half-life {e['hl']:.1f} bln · γ₀ = {e['g0']:.3f}")
        eff = dst.effectiveness(self.D, cyc)
        g = (self.FC['CEEMDAN-LSTM'] - self.tbp[-1]) * 100
        al = self.D['aligned']
        cw = al['Lending_Facility'].iloc[-1] - al['Deposit_Facility'].iloc[-1]
        eff.append({'period': f'Proyeksi {self.months[0]}–{self.months[-1]} (TBP ditahan)', 'gap': float(g.mean()),
                    'breach': self.breach_days, 'episodes': 1 if (g > 0).any() else 0, 'depth': float(max(g.max(), 0)),
                    'corridor': float((al['LPS_Rate'].iloc[-1] - al['Deposit_Facility'].iloc[-1]) / cw) if cw else None,
                    'tbp_bi': None, 'dep_bi': None, 'projection': True})
        self.R['effectiveness'] = {'rows': eff, 'dep_end': float(self.FC['CEEMDAN-LSTM'][-1])}

    # ------------------------------------------------------------------ simulasi
    def sim_context(self):
        al, M = self.D['aligned'], self.D['monthly']
        r = M['BI_Rate'].values
        dr, r0 = np.diff(r), r[:-1]
        sq = np.sqrt(np.maximum(r0, 1e-6))
        Xc = np.column_stack([1 / sq, sq])
        beta, *_ = np.linalg.lstsq(Xc, dr / sq, rcond=None)
        kappa = max(-beta[1], 1e-4)
        cir = {'kappa': float(kappa), 'theta': float(beta[0] / kappa), 'sigma': float(np.std(dr / sq - Xc @ beta, ddof=2))}
        ff = self.D['ff']
        return {'base_paths': self.base_paths, 'bi_last': float(self.bi[-1]), 'tbp_last': float(self.tbp[-1]),
                'dep_last': float(self.Dep[-1]),
                'sp_lf': float((ff['Lending_Facility'] - ff['BI_Rate']).iloc[-1]),
                'sp_df': float((ff['Deposit_Facility'] - ff['BI_Rate']).iloc[-1]),
                'spread_tbp_bi': float(np.median(al['LPS_Rate'] - al['BI_Rate'])),
                'multipliers': self.R['transmission']['multipliers'], 'cir': cir, 'months': self.months}

    def _simulation(self):
        ctx = self.sim_context()
        S = Simulator(ctx)
        base = S.run(0, 'C', with_heat=True)
        matrix = []
        for s in [-50, -25, 0, 25, 50]:
            row = {'shock': s}
            for r in ['A', 'C', 'D']:
                o = S.run(s, r, with_heat=False)
                row[r] = {'end': o['pb'][-1], 'max': o['pmax']}
            o = S.run(s, 'C', with_heat=False)
            row['tbp_min_end'] = o['tbp_min'][-1]
            matrix.append(row)
        oc = S.run(0, 'C', stochastic=True, with_heat=False)
        matrix.append({'shock': 'CIR', 'A': {'end': None, 'max': None}, 'C': {'end': oc['pb'][-1], 'max': oc['pmax']},
                       'D': {'end': None, 'max': None}, 'tbp_min_end': oc['tbp_min'][-1]})
        ctx_json = {k: v for k, v in ctx.items() if k != 'base_paths'}
        self.R['simulation'] = {'context': ctx_json, 'baseline': base, 'matrix': matrix, 'approx': False}
        # aturan kalibrasi (Tabel 14) & replay
        M = self.D['monthly']
        T = dio.TARGET
        X = sm.add_constant(M[[T, 'BI_Rate']])
        fit = sm.OLS(M['LPS_Rate'], X).fit()

        def sim_rule(target, review=4):
            tgt, res, cur = target.shift(1), [], np.nan
            for i, t_ in enumerate(tgt.values):
                if np.isnan(cur) or (i % review == 0 and not np.isnan(t_)):
                    cur = t_ if not np.isnan(t_) else M['LPS_Rate'].iloc[0]
                res.append(cur)
            return pd.Series(res, index=target.index)

        dep_end = float(self.base_paths.mean(0)[-1])
        rules = {'TBP aktual': (M['LPS_Rate'], float(self.tbp[-1])),
                 'Aturan estimasi (Dep + BI)': (sim_rule(pd.Series(round_grid(fit.predict(X)), index=M.index)),
                                                float(round_grid(fit.params['const'] + fit.params[T] * dep_end + fit.params['BI_Rate'] * self.bi[-1])))}
        for mg in (0, 25, 50):
            rules[f'Margin +{mg} bp di atas Deposito'] = (sim_rule(pd.Series(ceil_grid(M[T] + mg / 100), index=M.index)),
                                                         float(ceil_grid(dep_end + mg / 100)))
        rr = []
        for nm, (s, rec) in rules.items():
            gm = M[T] - s
            rr.append({'rule': nm, 'breach': float((gm > 0).mean() * 100), 'gap': float(gm.mean() * 100),
                       'changes': int((s.diff().abs() > 1e-9).sum()), 'rec': rec, 'delta': (rec - self.tbp[-1]) * 100})
        self.R['simulation']['rules'] = rr
        self.log(f"Simulasi baseline: P(breach) maks {base['pmax'] * 100:.0f}% · TBP minimum {base['tbp_min_max']:.2f}%")

    def _recommendations(self):
        b = self.R['simulation']['baseline']
        tm = b['tbp_min']
        first, peak = tm[0], max(tm)
        ev_ = self.R['evaluation']
        cv = ev_['cv'] or ev_['ho']
        imp = (cv['ARIMA']['rmse'] - cv['CEEMDAN-LSTM']['rmse']) / cv['ARIMA']['rmse'] * 100
        tr = {e['path']: e for e in self.R['transmission']['ardl'] if 'theta' in e}
        bt, bd = tr.get('BI Rate → TBP LPS', {}), tr.get('BI Rate → Deposito 1M', {})
        rules = {r['rule']: r for r in self.R['simulation']['rules']}
        dm = ev_['dm'][0]
        tbp0 = self.tbp[-1]
        self.R['recommendations'] = {
            'main': {'tbp_now': first, 'adj_now_bp': (first - tbp0) * 100, 'tbp_peak': peak,
                     'adj_peak_bp': (peak - tbp0) * 100, 'peak_month': b['tbp_min_month']},
            'rows': [
                ['Besaran & waktu penyesuaian TBP', 'Simulasi baseline, P(breach) ≤ 10%',
                 f"{(first - tbp0) * 100:+.0f} bp → {first:.2f}%; puncak {peak:.2f}% ({b['tbp_min_month']})", 'Jangka pendek'],
                ['Aturan kalibrasi TBP', 'Breach historis aturan estimasi vs TBP aktual',
                 f"{rules['TBP aktual']['breach']:.1f}% → {rules['Aturan estimasi (Dep + BI)']['breach']:.1f}% bulan", 'Menengah'],
                ['Kajian TBP selaras RDG BI', 'Dampak langsung BI Rate (γ₀) ARDL',
                 f"Deposito {bd.get('g0', float('nan')):.3f} vs TBP {bt.get('g0', float('nan')):.3f}", 'Jangka pendek–menengah'],
                ['Penyesuaian bertahap', 'Pass-through jangka panjang & half-life ARDL',
                 f"TBP θ {bt.get('theta', float('nan')):.2f} ({bt.get('hl', float('nan')):.1f} bln) · Deposito θ {bd.get('theta', float('nan')):.2f}",
                 'Menengah'],
                ['Ramalan titik vs distribusi', 'Lebar PI 95% akhir horizon',
                 f"±{(self.R['forecast']['monthly']['CEEMDAN-LSTM'][-1]['hi'] - self.R['forecast']['monthly']['CEEMDAN-LSTM'][-1]['lo']) * 50:.0f} bp",
                 'Operasional'],
                ['Pemantauan kuartalan CEEMDAN-LSTM', 'RMSE vs ARIMA & uji DM',
                 f"{imp:+.1f}% · DM SE {dm['dm_se'] if dm['dm_se'] is None else round(dm['dm_se'], 2)} (p = {dm['p_se'] if dm['p_se'] is None else round(dm['p_se'], 3)})",
                 'Operasional']]}


def run_pipeline(data_path, out_dir, cfg=None, log=None, progress=None):
    try:
        return Pipeline(data_path, out_dir, cfg, log, progress).run()
    except Exception as e:
        if log:
            log('ERROR: ' + str(e))
            log(traceback.format_exc().splitlines()[-1])
        raise

"""Membuat data CONTOH SINTETIS (bukan data riil) berformat sama dengan LPS.xlsx untuk mencoba fitur training ulang."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
d = pd.bdate_range('2017-01-03', '2026-09-30')
steps = {'2017-08-22': 4.50, '2017-09-22': 4.25, '2018-05-17': 4.50, '2018-05-30': 4.75, '2018-06-29': 5.25,
         '2018-08-15': 5.50, '2018-09-27': 5.75, '2018-11-15': 6.00, '2019-07-18': 5.75, '2019-08-22': 5.50,
         '2019-09-19': 5.25, '2019-10-24': 5.00, '2020-02-20': 4.75, '2020-03-19': 4.50, '2020-06-18': 4.25,
         '2020-07-16': 4.00, '2020-11-19': 3.75, '2021-02-18': 3.50, '2022-08-23': 3.75, '2022-09-22': 4.25,
         '2022-10-20': 4.75, '2022-11-17': 5.25, '2022-12-22': 5.50, '2023-01-19': 5.75, '2023-10-19': 6.00,
         '2024-04-24': 6.25, '2024-09-18': 6.00, '2025-01-15': 5.75, '2025-05-21': 5.50, '2025-07-16': 5.25,
         '2025-08-20': 5.00, '2025-09-17': 4.75, '2026-05-20': 5.25, '2026-06-18': 5.75}
bi = pd.Series(4.75, index=d)
for k, v in steps.items():
    bi[bi.index >= k] = v
tbp = (bi.shift(45).bfill() - 0.75).clip(lower=3.5)
tbp = (np.round(tbp / 0.25) * 0.25)
tbp = tbp.where(tbp.index.month % 3 == 1).ffill().bfill()
ema, dep = 3.9, []
for x, t in zip(bi.values, tbp.values):
    ema += 0.012 * ((x - 1.05) - ema)
    dep.append(ema + 0.03 * rng.standard_normal())
dep = pd.Series(dep, index=d).rolling(3, min_periods=1).mean()
dep[dep.index < '2020-01-02'] = np.nan
rows = []
for col, s in [('BI Rate', bi), ('Lending Facility', bi + 0.75), ('Deposit Facility', bi - 1.0), ('LPS Rate', tbp),
               ('Deposito (1 Bln - Avg)', dep)]:
    for t, v in s.dropna().items():
        rows.append({'Tanggal': t.strftime('%Y-%m-%d'), 'Keterangan': col, 'Nilai': round(float(v), 4)})
pd.DataFrame(rows).to_csv('data/sample/contoh_sintetis_LPS.csv', index=False)
print('ok', len(rows))

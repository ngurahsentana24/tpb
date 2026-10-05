"""Membaca, memvalidasi, dan menyelaraskan data harian suku bunga (format long atau wide)."""
import re
import numpy as np
import pandas as pd

VARS = ['BI_Rate', 'Lending_Facility', 'Deposit_Facility', 'LPS_Rate', 'Deposito_1M_Avg']
TARGET = 'Deposito_1M_Avg'
LABEL = {'BI_Rate': 'BI Rate', 'Lending_Facility': 'Lending Facility', 'Deposit_Facility': 'Deposit Facility',
         'LPS_Rate': 'TBP LPS', 'Deposito_1M_Avg': 'Deposito 1M'}

_ALIASES = {
    'BI_Rate': ['bi rate', 'bi_rate', 'birate', 'bi7drr', 'bi 7 day rr', 'suku bunga bi'],
    'Lending_Facility': ['lending facility', 'lending_facility', 'lf'],
    'Deposit_Facility': ['deposit facility', 'deposit_facility', 'df'],
    'LPS_Rate': ['lps rate', 'lps_rate', 'tbp', 'tbp lps', 'tingkat bunga penjaminan', 'lps'],
    'Deposito_1M_Avg': ['deposito (1 bln - avg)', 'deposito_1m_avg', 'deposito 1m', 'deposito 1 bulan',
                        'deposito', 'deposito 1m avg', 'deposito (1 bulan)'],
}


def _norm(s):
    return re.sub(r'\s+', ' ', str(s).strip().lower())


def _map_name(s):
    n = _norm(s)
    for k, al in _ALIASES.items():
        if n == k.lower() or n in al:
            return k
    return None


def read_any(path):
    p = str(path).lower()
    if p.endswith(('.xlsx', '.xls')):
        return pd.read_excel(path)
    return pd.read_csv(path, sep=None, engine='python')


def to_wide(raw):
    """Mengubah data mentah menjadi tabel wide (index tanggal, kolom VARS) + daftar pemeriksaan."""
    checks = []
    cols = {_norm(c): c for c in raw.columns}
    date_col = next((cols[c] for c in cols if c in ('tanggal', 'date', 'tgl', 'periode')), raw.columns[0])
    if 'keterangan' in cols:
        val_col = next((cols[c] for c in cols if c in ('nilai', 'value', 'rate', 'suku bunga')), raw.columns[-1])
        df = raw[[date_col, cols['keterangan'], val_col]].copy()
        df.columns = ['date', 'var', 'val']
        df['var'] = df['var'].map(_map_name)
        df = df.dropna(subset=['var'])
        df['val'] = pd.to_numeric(df['val'], errors='coerce')
        wide = df.pivot_table(index='date', columns='var', values='val', aggfunc='mean')
        checks.append(['ok', 'Format long (Tanggal, Keterangan, Nilai) terdeteksi → dipivot ke wide'])
    else:
        ren = {c: _map_name(c) for c in raw.columns if c != date_col}
        wide = raw.rename(columns={c: v for c, v in ren.items() if v}).rename(columns={date_col: 'date'})
        wide = wide.set_index('date')[[v for v in VARS if v in wide.columns]]
        wide = wide.apply(pd.to_numeric, errors='coerce')
        checks.append(['ok', 'Format wide (satu kolom per variabel) terdeteksi'])
    wide.index = pd.to_datetime(wide.index, errors='coerce', dayfirst=False)
    wide = wide[~wide.index.isna()].sort_index()
    ndup = int(wide.index.duplicated().sum())
    wide = wide[~wide.index.duplicated(keep='last')]
    missing = [LABEL[v] for v in VARS if v not in wide.columns or wide[v].isna().all()]
    if missing:
        checks.append(['err', 'Variabel tidak ditemukan: ' + ', '.join(missing)])
    else:
        checks.append(['ok', 'Lima variabel lengkap: BI Rate, LF, DF, TBP, Deposito 1M'])
    checks.append(['warn' if ndup else 'ok', f'{ndup} tanggal duplikat — dipakai nilai terakhir' if ndup
                   else 'Tidak ada tanggal duplikat'])
    # skala: jika nilai dalam desimal (0,0575) ubah ke persen
    for v in wide.columns:
        s = wide[v].dropna()
        if len(s) and s.abs().median() < 0.2:
            wide[v] = wide[v] * 100
            checks.append(['warn', f'{LABEL[v]} tampak dalam desimal → dikonversi ke persen'])
    return wide, checks


def prepare(wide):
    """Forward-fill kausal, penyelarasan pada tanggal Deposito, agregasi bulanan."""
    series = {v: wide[v].dropna() for v in VARS}
    ff = wide[VARS].ffill()
    aligned = ff.dropna()
    aligned = aligned[aligned.index >= series[TARGET].index.min()]
    monthly = aligned.resample('ME').mean().dropna()
    return {'series': series, 'ff': ff, 'aligned': aligned, 'monthly': monthly}


def load(path):
    raw = read_any(path)
    wide, checks = to_wide(raw)
    if any(c[0] == 'err' for c in checks):
        return None, checks
    D = prepare(wide)
    al = D['aligned']
    checks.append(['ok', f"{len(al):,} hari observasi selaras · {al.index.min():%d %b %Y} – {al.index.max():%d %b %Y}".replace(',', '.')])
    last_q = al[al.index >= al.index.max() - pd.offsets.QuarterBegin(startingMonth=1)]
    checks.append(['ok' if len(al) >= 500 else 'warn',
                   'Panjang data memadai untuk rolling-origin' if len(al) >= 500
                   else 'Data < 500 hari: jumlah fold akan dikurangi otomatis'])
    return D, checks


def preview(D, n=10):
    al = D['aligned'].tail(n)
    rows = []
    for d, r in al.iterrows():
        rows.append([f'{d:%d %b %Y}', *(round(float(r[v]), 3) for v in VARS),
                     round(float((r[TARGET] - r['LPS_Rate']) * 100), 1)])
    return rows


def template_csv():
    d = pd.bdate_range('2020-01-02', periods=5)
    rows = []
    for x in d:
        for v, val in zip(['BI Rate', 'Lending Facility', 'Deposit Facility', 'LPS Rate', 'Deposito (1 Bln - Avg)'],
                          [5.00, 5.75, 4.25, 5.50, 4.85]):
            rows.append({'Tanggal': x.strftime('%Y-%m-%d'), 'Keterangan': v, 'Nilai': val})
    return pd.DataFrame(rows).to_csv(index=False)

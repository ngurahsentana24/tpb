# TBP Lab — CEEMDAN-LSTM Deposito 1M & Efektivitas TBP LPS

Dashboard web (Flask + HTML/CSS/JS + Chart.js) untuk paper **LPS Research Fair 2026**:
peramalan suku bunga Deposito 1 bulan dengan model hibrida **ARIMA-CEEMDAN-LSTM** dan implikasinya
terhadap Tingkat Bunga Penjaminan (TBP) LPS.

> Prototipe riset — bukan sistem resmi Lembaga Penjamin Simpanan.

## Dua mode hasil

| Mode | Sumber | Keterangan |
|---|---|---|
| **Hasil Paper** | `data/paper/results.json` | Angka resmi paper & lampiran: 6 fold rolling-origin 2025Q1–2026Q2, holdout 2026Q3, proyeksi ARIMA-CEEMDAN-LSTM tiga spesifikasi (level, gap, spread), ARDL/NARDL, efektivitas TBP, simulasi, rekomendasi. Tidak dihitung ulang. |
| **Training Ulang** | `data/runs/<id>/results.json` | Hasil dari data harian yang diunggah pengguna. Pipeline yang sama dijalankan di server (versi ringan, hiperparameter tetap tanpa Bayesian optimization). |

Pindah mode lewat tombol **Sumber hasil** di sidebar. Mode Training Ulang aktif setelah ada minimal satu run.

## Menu

Ringkasan · Data & Training Ulang · Dekomposisi · Evaluasi Model · Proyeksi · Efektivitas TBP & Transmisi ·
Simulasi Skenario · Rekomendasi · **Profil Peneliti** (dari `static/data/profile.json`).

## Struktur

```
app.py                      Flask API + halaman
core/
  io.py                     baca CSV/XLSX (format long atau wide), penyelarasan & forward-fill kausal
  stats.py                  statistik deskriptif, ADF/KPSS, siklus BI, efektivitas per siklus
  decomp.py                 CEEMDAN/EMD, Sample Entropy, co-IMF (K-Means), dekomposisi kausal bergulir
  models.py                 LSTM (PyTorch) / MLP fallback, ARIMA(0,1,0), Linear Trend, Rolling Mean, λ 1-SE
  evaluation.py             RMSE/MAE/MAPE, R²OOS, akurasi arah, Pesaran-Timmermann, Diebold-Mariano (HLN)
  transmission.py           ARDL/ECM (uji bounds PSS), NARDL, multiplier dinamis
  simulation.py             simulasi 5 langkah: shock BI → TBP → Deposito → gap → P(breach)
  pipeline.py               orkestrasi training ulang → results.json
  store.py                  penyimpanan run & antrean job latar belakang
templates/index.html        UI
static/css/style.css        tema hijau–kuning
static/js/app.js            logika frontend
static/img/lps-logo.png     logo
static/paper/*.png          gambar paper/lampiran (mode paper)
static/data/profile.json    isi tab Profil (edit sesuai kebutuhan)
data/paper/                 hasil paper (results.json + base_paths.npy)
data/sample/                contoh data SINTETIS untuk mencoba training ulang
scripts/build_paper_results.py   menyusun ulang data/paper dari angka paper + proyeksi M1/M2/M3
scripts/make_sample_data.py      membuat contoh data sintetis
```

## Menjalankan lokal

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # ringan (MLP); atau requirements-full.txt untuk LSTM PyTorch
python app.py                      # http://127.0.0.1:5000
```

## Deploy

**Render** — push repo ke GitHub → New → Blueprint (memakai `render.yaml`) atau Web Service dengan
Build `pip install -r requirements.txt`, Start `gunicorn app:app --workers 1 --threads 4 --timeout 600 --bind 0.0.0.0:$PORT`.

**Railway / Heroku-like** — otomatis membaca `Procfile` dan `runtime.txt`.

**Docker**
```bash
docker build -t tbp-lab .                    # ringan
docker build --build-arg FULL=1 -t tbp-lab . # dengan PyTorch
docker run -p 8000:8000 -e RETRAIN_TOKEN=rahasia tbp-lab
```

Catatan deploy:
- Gunakan **1 worker** (job training disimpan di memori proses); thread boleh lebih dari satu.
- Disk pada paket gratis biasanya **sementara**: run training ulang hilang saat server restart/redeploy.
  Mode Hasil Paper selalu tersedia karena ikut di repo.
- Training ulang default (6 fold, 3 seed, 60 epoch) memakan beberapa menit di CPU kecil. Untuk server gratis
  pakai mesin MLP, 2–3 fold, 1 seed.

## Variabel lingkungan

| Nama | Default | Fungsi |
|---|---|---|
| `RETRAIN_TOKEN` | kosong | Jika diisi, unggah/training/aktifkan/hapus run butuh token (kolom muncul di UI). |
| `MAX_UPLOAD_MB` | 20 | Batas ukuran file. |
| `PORT` | 5000 | Port lokal. |

## Format data untuk training ulang

Data **harian**, CSV/XLSX, salah satu format:

- **Long** (seperti `LPS.xlsx`): kolom `Tanggal, Keterangan, Nilai`, dengan Keterangan:
  `BI Rate`, `Lending Facility`, `Deposit Facility`, `LPS Rate`, `Deposito (1 Bln - Avg)`.
- **Wide**: kolom `Tanggal` + lima kolom variabel di atas.

Nilai dalam persen (5,75) atau desimal (0,0575 — dikonversi otomatis). Unduh template di menu Data.
Minimal ±500 hari Deposito 1M agar rolling-origin berjalan.

## Endpoint API

| Metode | Path | Keterangan |
|---|---|---|
| GET | `/api/health`, `/api/config` | status & konfigurasi |
| GET | `/api/results?source=paper\|run` | seluruh hasil (JSON) |
| POST | `/api/upload` | multipart `file` → validasi + pratinjau |
| POST | `/api/pipeline/run` | `{upload_id, config}` → `job_id` |
| GET | `/api/pipeline/status/<job_id>?since=n` | progres & log |
| GET/POST/DELETE | `/api/runs`, `/api/runs/<id>/activate`, `/api/runs/<id>` | riwayat run |
| POST | `/api/simulate` | `{source, shock, rule A\|C\|D, adj, lag, transmission ardl\|nardl, tol, B, stochastic}` |
| GET | `/api/export.xlsx?source=` | ekspor tabel ke Excel |
| GET | `/api/template`, `/api/sample`, `/api/profile` | template, contoh data sintetis, profil |

## Catatan metodologis mode Training Ulang

- Target `s` (level: D; gap: D − TBP; spread: D − BI) dimodelkan; ramalan dikembalikan ke level D.
- ARIMA(0,1,0) sebagai basis; koreksi LSTM atas co-IMF (dekomposisi kausal jendela W) dikalikan λ yang dipilih
  dengan aturan 1-SE pada blok kalibrasi 63 hari; early stopping pada blok 42 hari.
- LSTM murni: jaringan yang sama tanpa komponen CEEMDAN dan tanpa penyusutan λ.
- Proyeksi 6 bulan rekursif; PI 95% dan P(breach) dari bootstrap residual (B jalur);
  TBP minimum = kelipatan 25 bp terkecil agar P(breach) ≤ 10%.
- Simulasi mode paper memakai jalur dasar yang direkonstruksi dari PI bulanan (aproksimasi);
  tabel matriks resmi tetap angka paper.

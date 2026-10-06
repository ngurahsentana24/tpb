"""TBP Lab — Flask API & dashboard CEEMDAN-LSTM Deposito 1M dan efektivitas TBP LPS.

Dua sumber hasil:
  • paper — angka yang dilaporkan dalam paper & lampiran (data/paper/results.json, read-only)
  • run   — hasil training ulang dari data yang diunggah pengguna (data/runs/<id>/results.json)

Jalankan lokal:  python app.py   →  http://127.0.0.1:5000
Produksi:        gunicorn app:app --workers 1 --threads 4 --timeout 600
"""
import io
import json
import os
import uuid

from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from core import store
from core.io import load, preview, template_csv
from core.models import engine_name
from core.pipeline import DEFAULTS, STEPS

ROOT = os.path.dirname(os.path.abspath(__file__))
ALLOWED = {'.xlsx', '.xls', '.csv'}
TOKEN = os.environ.get('RETRAIN_TOKEN', '').strip()          # opsional: kunci fitur training ulang
MAX_MB = int(os.environ.get('MAX_UPLOAD_MB', '4' if store.SERVERLESS else '20'))   # Vercel: batas body 4,5 MB

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = MAX_MB * 1024 * 1024
app.json.ensure_ascii = False

# Batas konfigurasi agar training ulang tetap ringan di server web
LIMITS = dict(n_folds=(1, 8), window=(126, 504), K=(3, 8), trials=(5, 50), lookback=(3, 21), units=(8, 64),
              epochs=(5, 150), n_seeds=(1, 5), B=(200, 2000), n_train=(250, 1500))
CHOICES = dict(spec=('level', 'gap', 'spread'), engine=('auto', 'lstm', 'mlp'), decomp_causal=('emd', 'ceemdan'))
# Mode serverless (mis. Vercel): training dijalankan sinkron dalam satu request, jadi dibatasi agar selesai
# sebelum batas durasi fungsi. Ubah lewat env SERVERLESS_MAX_FOLDS dsb. bila paket hosting mengizinkan.
SERVERLESS_CAPS = dict(n_folds=int(os.environ.get('SERVERLESS_MAX_FOLDS', 2)), epochs=int(os.environ.get('SERVERLESS_MAX_EPOCHS', 20)),
                       n_seeds=1, B=int(os.environ.get('SERVERLESS_MAX_B', 500)), trials=10)


def _err(msg, code=400):
    return jsonify(ok=False, error=msg), code


def _authorised():
    if not TOKEN:
        return True
    return request.headers.get('X-Retrain-Token', '') == TOKEN or request.form.get('token', '') == TOKEN


def _source():
    s = request.args.get('source', 'paper')
    return s if s in ('paper', 'run') or s.startswith('20') else 'paper'


# ------------------------------------------------------------------------------------------- halaman
ASSET_VERSION = os.environ.get('VERCEL_GIT_COMMIT_SHA', '')[:8] or str(int(os.path.getmtime(os.path.join(ROOT, 'static', 'js', 'app.js'))))


@app.get('/')
def index():
    r = app.make_response(render_template('index.html', v=ASSET_VERSION))
    r.headers['Cache-Control'] = 'no-cache'
    return r


@app.get('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(ROOT, 'static', 'img'), 'lps-logo.png')


# ------------------------------------------------------------------------------------------- info
@app.get('/api/health')
def health():
    return jsonify(ok=True, engine=engine_name('auto'), paper=store.results('paper') is not None,
                   active_run=None if store.SERVERLESS else store.active_run(), token_required=bool(TOKEN),
                   max_upload_mb=MAX_MB, serverless=store.SERVERLESS, serverless_caps=SERVERLESS_CAPS)


@app.get('/api/config')
def config():
    return jsonify(defaults=DEFAULTS, limits=LIMITS, choices=CHOICES, steps=STEPS, engine=engine_name('auto'),
                   token_required=bool(TOKEN), serverless=store.SERVERLESS, serverless_caps=SERVERLESS_CAPS)


@app.get('/api/results')
def results():
    src = _source()
    R = store.results(src)
    if R is None:
        return _err('Belum ada hasil untuk sumber ini.' if src != 'paper' else
                    'data/paper/results.json tidak ditemukan — jalankan scripts/build_paper_results.py', 404)
    return jsonify(R)


@app.get('/api/profile')
def profile():
    return send_from_directory(os.path.join(ROOT, 'static', 'data'), 'profile.json')


# ------------------------------------------------------------------------------------------- data & training
@app.get('/api/template')
def template():
    return send_file(io.BytesIO(template_csv().encode('utf-8')), mimetype='text/csv', as_attachment=True,
                     download_name='template_data_LPS.csv')


@app.get('/api/sample')
def sample():
    return send_from_directory(os.path.join(ROOT, 'data', 'sample'), 'contoh_sintetis_LPS.csv', as_attachment=True)


@app.post('/api/upload')
def upload():
    if not _authorised():
        return _err('Token training ulang tidak valid.', 403)
    f = request.files.get('file')
    if not f or not f.filename:
        return _err('File tidak ditemukan.')
    name = secure_filename(f.filename) or 'data.csv'
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED:
        return _err('Format harus .xlsx, .xls, atau .csv')
    uid = uuid.uuid4().hex[:12]
    path = os.path.join(store.UPLOAD_DIR, f'{uid}{ext}')
    f.save(path)
    try:
        D, checks = load(path)
    except Exception as e:  # noqa
        os.remove(path)
        return _err(f'Data tidak dapat dibaca: {e}')
    a = D['aligned']
    try:
        with open(path + '.json', 'w') as fh:
            json.dump({'name': f.filename}, fh)
    except OSError:
        pass
    return jsonify(ok=True, upload_id=uid + ext, name=f.filename, rows=int(len(a)),
                   period=[a.index[0].strftime('%Y-%m-%d'), a.index[-1].strftime('%Y-%m-%d')],
                   checks=checks, preview=preview(D, 10))


def _clean_cfg(raw):
    cfg = {}
    for k, (lo, hi) in LIMITS.items():
        if k in raw:
            try:
                cfg[k] = int(min(max(int(raw[k]), lo), hi))
            except (TypeError, ValueError):
                pass
    for k, opts in CHOICES.items():
        if raw.get(k) in opts:
            cfg[k] = raw[k]
    if 'holdout' in raw:
        cfg['holdout'] = bool(raw['holdout'])
    if store.SERVERLESS:
        for k, cap in SERVERLESS_CAPS.items():
            cfg[k] = min(cfg.get(k, DEFAULTS.get(k, cap)), cap)
        cfg['engine'] = 'mlp' if engine_name(cfg.get('engine', 'auto')) != 'lstm' else cfg.get('engine', 'auto')
        cfg['decomp_causal'] = 'emd'
    return cfg


@app.post('/api/pipeline/run')
def run():
    if not _authorised():
        return _err('Token training ulang tidak valid.', 403)
    if store.SERVERLESS:
        return _run_sync()
    body = request.get_json(silent=True) or {}
    uid = secure_filename(str(body.get('upload_id', '')))
    path = os.path.join(store.UPLOAD_DIR, uid)
    if not uid or not os.path.exists(path):
        return _err('Unggah data terlebih dahulu.')
    if any(not j['done'] for j in store.JOBS.values()):
        return _err('Masih ada training yang berjalan. Tunggu hingga selesai.', 409)
    name = ''
    if os.path.exists(path + '.json'):
        name = json.load(open(path + '.json')).get('name', '')
    job = store.start_job(path, _clean_cfg(body.get('config', {})), name)
    return jsonify(ok=True, job_id=job['id'], run_id=job['run_id'], steps=STEPS)


def _run_sync():
    """Serverless: file dikirim ulang bersama konfigurasi, training sinkron, hasil dikembalikan ke browser."""
    f = request.files.get('file')
    if f and f.filename:
        ext = os.path.splitext(secure_filename(f.filename) or 'data.csv')[1].lower()
        if ext not in ALLOWED:
            return _err('Format harus .xlsx, .xls, atau .csv')
        try:
            raw = json.loads(request.form.get('config', '{}'))
        except ValueError:
            raw = {}
        name = f.filename
        path = os.path.join(store.UPLOAD_DIR, f'{uuid.uuid4().hex[:12]}{ext}')
        f.save(path)
    else:
        # kompatibel dengan frontend lama (JSON {upload_id}): pakai file di /tmp bila masih ada di instance ini
        body = request.get_json(silent=True) or {}
        raw = body.get('config', {}) or {}
        uid = secure_filename(str(body.get('upload_id', '')))
        path = os.path.join(store.UPLOAD_DIR, uid)
        if not uid or not os.path.exists(path):
            return _err('File unggahan tidak ditemukan di server (serverless). Muat ulang halaman (Ctrl+F5), '
                        'pilih file lagi, lalu jalankan training.')
        name = body.get('name') or uid
    try:
        R, base, logs = store.run_sync(path, _clean_cfg(raw), name)
    except Exception as e:  # noqa
        return _err(f'Training gagal: {e}', 500)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return jsonify(ok=True, sync=True, results=R, base_paths=base, log=logs)


@app.get('/api/pipeline/status/<jid>')
def status(jid):
    s = store.job_status(jid, int(request.args.get('since', 0)))
    if s is None:
        return _err('Job tidak ditemukan (server mungkin dimulai ulang).', 404)
    return jsonify(s)


@app.get('/api/runs')
def runs():
    return jsonify(runs=store.list_runs(), active=store.active_run())


@app.post('/api/runs/<rid>/activate')
def activate(rid):
    if not _authorised():
        return _err('Token training ulang tidak valid.', 403)
    rid = secure_filename(rid)
    if not os.path.exists(os.path.join(store.RUNS_DIR, rid, 'results.json')):
        return _err('Run tidak ditemukan.', 404)
    store.set_active(rid)
    return jsonify(ok=True, active=rid)


@app.delete('/api/runs/<rid>')
def delete(rid):
    if not _authorised():
        return _err('Token training ulang tidak valid.', 403)
    ok = store.delete_run(secure_filename(rid))
    return jsonify(ok=ok) if ok else _err('Run aktif tidak dapat dihapus / tidak ditemukan.')


# ------------------------------------------------------------------------------------------- simulasi
@app.post('/api/simulate')
def simulate():
    b = request.get_json(silent=True) or {}
    src = b.get('source', 'paper')
    if src == 'inline' and b.get('context') and b.get('base_paths'):
        sim = store.simulator_inline(b['context'], b['base_paths'])
    else:
        sim = store.simulator(src if src in ('paper', 'run') else 'paper')
    if sim is None:
        return _err('Belum ada hasil untuk sumber ini.', 404)
    try:
        out = sim.run(shock=int(b.get('shock', 0)), rule=str(b.get('rule', 'C'))[:1].upper(), adj=int(b.get('adj', 0)),
                      lag=int(b.get('lag', 1)), transmission='nardl' if b.get('transmission') == 'nardl' else 'ardl',
                      tol=int(b.get('tol', 10)), B=int(b.get('B', 1000)), stochastic=bool(b.get('stochastic', False)),
                      seed=42, with_heat=bool(b.get('heat', True)))
    except Exception as e:  # noqa
        return _err(f'Simulasi gagal: {e}', 500)
    return jsonify(out)


# ------------------------------------------------------------------------------------------- ekspor
def _xlsx(R):
    import pandas as pd
    buf = io.BytesIO()
    ev = R['evaluation']
    with pd.ExcelWriter(buf, engine='openpyxl') as xw:
        pd.DataFrame(R['descriptive']['table']).to_excel(xw, sheet_name='Deskriptif', index=False)
        pd.DataFrame(R['descriptive']['stationarity']).to_excel(xw, sheet_name='Stasioneritas', index=False)
        for k, lab in (('cv', 'Evaluasi_rolling'), ('ho', 'Evaluasi_holdout')):
            pd.DataFrame(ev[k]).T.rename_axis('model').reset_index().to_excel(xw, sheet_name=lab, index=False)
        pd.DataFrame(ev['dm']).to_excel(xw, sheet_name='Diebold_Mariano', index=False)
        rows = []
        for mname, lst in R['forecast']['monthly'].items():
            for h, r in enumerate(lst):
                rows.append(dict(model=mname, bulan=R['forecast']['months'][h], **r))
        pd.DataFrame(rows).to_excel(xw, sheet_name='Proyeksi_bulanan', index=False)
        pd.DataFrame(R['transmission']['ardl']).to_excel(xw, sheet_name='ARDL', index=False)
        pd.DataFrame(R['transmission']['nardl']).to_excel(xw, sheet_name='NARDL', index=False)
        pd.DataFrame(R['effectiveness']['rows']).to_excel(xw, sheet_name='Efektivitas', index=False)
        pd.DataFrame(R['simulation']['baseline']['table']).to_excel(xw, sheet_name='Simulasi_baseline', index=False)
        pd.DataFrame(R['simulation']['rules']).to_excel(xw, sheet_name='Aturan_TBP', index=False)
        pd.DataFrame(R['recommendations']['rows'],
                     columns=['Rekomendasi', 'Dasar empiris', 'Nilai kunci', 'Horizon']).to_excel(xw, sheet_name='Rekomendasi', index=False)
    buf.seek(0)
    return buf


XLSX = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'


@app.get('/api/export.xlsx')
def export():
    src = _source()
    R = store.results(src)
    if R is None:
        abort(404)
    name = 'hasil_paper_TBP_LPS.xlsx' if src == 'paper' else f'hasil_run_{R["meta"].get("run_id", "aktif")}.xlsx'
    return send_file(_xlsx(R), as_attachment=True, download_name=name, mimetype=XLSX)


@app.post('/api/export.xlsx')
def export_inline():
    R = request.get_json(silent=True)
    if not R or 'evaluation' not in R:
        return _err('Hasil tidak valid.')
    return send_file(_xlsx(R), as_attachment=True, download_name=f'hasil_run_{R["meta"].get("run_id", "lokal")}.xlsx',
                     mimetype=XLSX)


@app.errorhandler(413)
def too_large(_):
    return _err(f'File melebihi {MAX_MB} MB.', 413)


@app.errorhandler(Exception)
def unhandled(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        if request.path.startswith('/api/'):
            return _err(e.description or e.name, e.code)
        return e
    app.logger.exception('Kesalahan tak tertangani')
    return _err(f'Kesalahan server: {type(e).__name__}: {e}', 500)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=os.environ.get('FLASK_DEBUG') == '1')

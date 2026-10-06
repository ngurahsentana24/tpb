"""Penyimpanan hasil (paper vs run training ulang) dan antrean job latar belakang."""
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
import numpy as np

from .pipeline import run_pipeline, STEPS
from .simulation import Simulator

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
PAPER_DIR = os.path.join(BASE, 'paper')                     # read-only, ikut di repo

# Serverless (Vercel, AWS Lambda, dsb.): sistem berkas aplikasi read-only, hanya /tmp yang bisa ditulis,
# dan proses tidak boleh menjalankan thread latar belakang setelah respons dikirim.
SERVERLESS = bool(os.environ.get('VERCEL') or os.environ.get('AWS_LAMBDA_FUNCTION_NAME')
                  or os.environ.get('SERVERLESS', '').lower() in ('1', 'true', 'yes')
                  or not os.access(BASE, os.W_OK))          # folder aplikasi read-only → perlakukan sebagai serverless
WRITE_BASE = os.environ.get('DATA_DIR') or (os.path.join(tempfile.gettempdir(), 'tbp-lab') if SERVERLESS else BASE)
RUNS_DIR = os.path.join(WRITE_BASE, 'runs')
UPLOAD_DIR = os.path.join(WRITE_BASE, 'uploads')
ACTIVE = os.path.join(RUNS_DIR, 'active.txt')
for d in (RUNS_DIR, UPLOAD_DIR):
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:  # pragma: no cover
        pass

_cache = {}
_lock = threading.Lock()


def _dir(source):
    if source == 'paper':
        return PAPER_DIR
    rid = active_run() if source in (None, 'run', 'active') else source
    return os.path.join(RUNS_DIR, rid) if rid else None


def active_run():
    if os.path.exists(ACTIVE) and os.path.isdir(RUNS_DIR):
        rid = open(ACTIVE).read().strip()
        if rid and os.path.exists(os.path.join(RUNS_DIR, rid, 'results.json')):
            return rid
    return None


def set_active(rid):
    with open(ACTIVE, 'w') as f:
        f.write(rid)


def list_runs():
    out = []
    if not os.path.isdir(RUNS_DIR):
        return out
    for rid in sorted(os.listdir(RUNS_DIR), reverse=True):
        p = os.path.join(RUNS_DIR, rid, 'results.json')
        if os.path.exists(p):
            m = json.load(open(p)).get('meta', {})
            out.append({'id': rid, 'created': m.get('created'), 'data_end': m.get('data_end'), 'spec': m.get('spec_label'),
                        'engine': m.get('engine'), 'active': rid == active_run(), 'file': m.get('file')})
    return out


def delete_run(rid):
    p = os.path.join(RUNS_DIR, rid)
    if os.path.isdir(p) and rid != active_run():
        shutil.rmtree(p)
        return True
    return False


def results(source='paper'):
    d = _dir(source)
    if not d or not os.path.exists(os.path.join(d, 'results.json')):
        return None
    p = os.path.join(d, 'results.json')
    key = (p, os.path.getmtime(p))
    if key not in _cache:
        _cache[key] = json.load(open(p))
    return _cache[key]


def simulator_inline(context, base_paths):
    ctx = dict(context)
    ctx['base_paths'] = np.asarray(base_paths, float)
    return Simulator(ctx)


def run_sync(upload_path, cfg, original_name=''):
    """Training ulang sinkron (mode serverless): hasil dikembalikan langsung ke klien."""
    rid = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:4]
    out = os.path.join(RUNS_DIR, rid)
    logs = []
    log = lambda m: logs.append([time.strftime('%H:%M:%S'), m])
    try:
        run_pipeline(upload_path, out, cfg, log, lambda s, p: None)
        R = json.load(open(os.path.join(out, 'results.json')))
        R['meta']['run_id'] = rid
        R['meta']['file'] = original_name
        R['meta']['serverless'] = True
        base = np.load(os.path.join(out, 'base_paths.npy'))
        return R, np.round(base, 5).tolist(), logs
    finally:
        shutil.rmtree(out, ignore_errors=True)


def simulator(source='paper'):
    d = _dir(source)
    R = results(source)
    if R is None:
        return None
    ctx = dict(R['simulation']['context'])
    ctx['base_paths'] = np.load(os.path.join(d, 'base_paths.npy'))
    return Simulator(ctx)


# ---------------------------------------------------------------------------- job latar belakang
JOBS = {}


def start_job(upload_path, cfg, original_name=''):
    jid = uuid.uuid4().hex[:10]
    rid = time.strftime('%Y%m%d-%H%M%S') + '-' + jid[:4]
    job = {'id': jid, 'run_id': rid, 'step': 0, 'pct': 0.0, 'log': [], 'done': False, 'error': None,
           'steps': STEPS, 'started': time.time()}
    JOBS[jid] = job

    def log(m):
        job['log'].append([time.strftime('%H:%M:%S'), m])

    def prog(step, pct):
        job['step'], job['pct'] = step, round(pct * 100, 1)

    def work():
        out = os.path.join(RUNS_DIR, rid)
        try:
            with _lock:                        # satu training dalam satu waktu
                log(f'Mulai training ulang · run {rid}')
                run_pipeline(upload_path, out, cfg, log, prog)
                p = os.path.join(out, 'results.json')
                R = json.load(open(p))
                R['meta']['run_id'] = rid
                R['meta']['file'] = original_name
                json.dump(R, open(p, 'w'))
                set_active(rid)
            job['done'] = True
        except Exception as e:  # noqa
            job['error'] = str(e)
            job['done'] = True
            shutil.rmtree(out, ignore_errors=True)

    threading.Thread(target=work, daemon=True).start()
    return job


def job_status(jid, since=0):
    j = JOBS.get(jid)
    if not j:
        return None
    return {k: (v[since:] if k == 'log' else v) for k, v in j.items()} | {'log_len': len(j['log'])}

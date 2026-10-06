/* ==========================================================================
   TBP Lab — frontend (Chart.js 4) untuk Flask API
   Dua sumber hasil: "paper" (angka paper & lampiran) dan "run" (training ulang)
   ========================================================================== */
const API = {
  health: '/api/health', config: '/api/config', results: '/api/results', profile: '/api/profile',
  upload: '/api/upload', run: '/api/pipeline/run', status: '/api/pipeline/status/', runs: '/api/runs',
  simulate: '/api/simulate', export: '/api/export.xlsx'
};

/* ---------- util ---------- */
const $ = id => document.getElementById(id);
const isNum = x => x !== null && x !== undefined && x !== '' && isFinite(x);
const f2 = (x, d = 2) => isNum(x) ? Number(x).toLocaleString('id-ID', {minimumFractionDigits: d, maximumFractionDigits: d}) : '–';
const sgn = (x, d = 0) => isNum(x) ? (x > 0 ? '+' : '') + f2(x, d) : '–';
const pct = (x, d = 0) => isNum(x) ? f2(x * 100, d) + '%' : '–';
const pv = p => !isNum(p) ? '–' : p < 0.001 ? '&lt;0,001' : f2(p, 3);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const C = {g900: '#0D4A3E', g800: '#105C4D', g700: '#17705E', g500: '#2F8F78', g300: '#8CC7B5', g100: '#E4F2EC', y500: '#F2C14E',
  y400: '#F6D27A', red: '#D2584F', blue: '#4C86C6', purple: '#8B6BB8', grey: '#9AA6A0', orange: '#E08A3C'};
const MCOL = {'CEEMDAN-LSTM': C.red, 'LSTM': C.purple, 'ARIMA': C.blue, 'Linear Trend': C.grey, 'Rolling Mean (30)': C.g500,
  'CEEMDAN-LSTM (level)': C.red, 'CEEMDAN-LSTM (gap)': C.g700, 'CEEMDAN-LSTM (spread)': C.orange};
const mcol = m => MCOL[m] || C.g900;
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des'];
const ym = s => {const [y, m] = s.split('-'); return MON[+m - 1] + ' ' + y.slice(2)};

Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.color = '#6A7A73';
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.boxWidth = 8;
Chart.defaults.plugins.tooltip.backgroundColor = '#0D4A3E';
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.cornerRadius = 10;
Chart.defaults.elements.point.radius = 0;
Chart.defaults.elements.line.borderWidth = 2;
Chart.defaults.maintainAspectRatio = false;
Chart.defaults.scale.grid.color = '#EEF0EA';
Chart.defaults.scale.border.display = false;
const CH = {};
function mk(id, cfg) {if (CH[id]) CH[id].destroy(); const el = $(id); if (!el) return null; CH[id] = new Chart(el, cfg); return CH[id]}
function segBind(id, cb) {const el = $(id); el.querySelectorAll('button').forEach(b => b.onclick = () => {if (b.disabled) return; el.querySelectorAll('button').forEach(x => x.classList.remove('on')); b.classList.add('on'); cb(b.dataset)})}
function toast(msg) {$('toastMsg').textContent = msg; $('toast').classList.add('show'); setTimeout(() => $('toast').classList.remove('show'), 2800)}
function table(id, head, rows, numFrom = 1, bestRow = -1) {
  $(id).innerHTML = '<thead><tr>' + head.map((h, i) => `<th class="${i >= numFrom ? 'num' : ''}">${h}</th>`).join('') + '</tr></thead><tbody>' +
    (rows.length ? rows.map((r, ri) => `<tr class="${ri === bestRow ? 'best' : ''}">` + r.map((c, i) => `<td class="${i >= numFrom ? 'num' : ''}">${c ?? '–'}</td>`).join('') + '</tr>').join('')
      : `<tr><td class="empty" colspan="${head.length}">Tidak ada data.</td></tr>`) + '</tbody>';
}
const fig = (src, cap) => `<img class="fig" src="/static/${src}" alt="${esc(cap)}" loading="lazy"><div class="fig-cap">${cap}</div>`;
const cardH = (t, s) => `<div class="card-h"><div><h3>${t}</h3><div class="sub">${s || ''}</div></div></div>`;
async function getJSON(url, opt) {
  const r = await fetch(url, opt); let j = {};
  try {j = await r.json()} catch (e) {throw new Error('Respons server tidak valid')}
  if (!r.ok || j.ok === false) throw new Error(j.error || ('HTTP ' + r.status));
  return j;
}

/* ---------- state ---------- */
let SRC = 'paper', R = null, CFG = null, HEALTH = null;
let LOCAL = null;            // mode serverless: {results, base_paths} hasil training disimpan di browser
const SL = () => !!(HEALTH && HEALTH.serverless);
const RENDER = {}, done = {};
let CUR = 'overview';

/* ==========================================================================
   NAVIGASI & SUMBER HASIL
   ========================================================================== */
const TITLES = {
  overview: ['Ringkasan', 'Dinamika suku bunga, gap Deposito 1M − TBP, dan hasil peramalan'],
  data: ['Data & Training Ulang', 'Unggah data harian baru lalu jalankan ulang dekomposisi, model, proyeksi, dan simulasi'],
  decomp: ['Dekomposisi CEEMDAN', 'Struktur multi-frekuensi target dan gap serta pemetaannya ke horizon kebijakan'],
  model: ['Evaluasi Model', 'ARIMA-CEEMDAN-LSTM vs LSTM murni, ARIMA, dan pembanding sederhana'],
  forecast: ['Proyeksi 6 Bulan', 'Deposito 1M dengan PI 95% dan probabilitas breach saat TBP ditahan'],
  effect: ['Efektivitas TBP & Transmisi', 'Breach, posisi koridor, dan pass-through BI Rate per siklus kebijakan'],
  sim: ['Simulasi Skenario', 'Shock BI Rate → TBP → Deposito → gap → P(breach)'],
  rec: ['Rekomendasi Kebijakan', 'Angka kunci dan matriks rekomendasi bagi LPS'],
  profile: ['Profil Peneliti', 'Penulis paper dan pengembang dashboard']};
const NEEDS_R = ['overview', 'decomp', 'model', 'forecast', 'effect', 'sim', 'rec'];

function go(v) {
  CUR = v;
  document.querySelectorAll('.nav').forEach(b => b.classList.toggle('active', b.dataset.view === v));
  document.querySelectorAll('.view').forEach(s => s.classList.toggle('active', s.id === 'v-' + v));
  $('pageTitle').textContent = TITLES[v][0]; $('pageSub').textContent = TITLES[v][1];
  $('side').classList.remove('open');
  $('srcNotice').style.display = NEEDS_R.includes(v) ? '' : 'none';
  if (!done[v] && RENDER[v] && (R || !NEEDS_R.includes(v))) {
    try {RENDER[v]()} catch (e) {console.error(e); toast('Gagal menampilkan: ' + e.message)}
    done[v] = true;
  }
  window.scrollTo({top: 0, behavior: 'smooth'});
}
document.querySelectorAll('.nav').forEach(b => b.onclick = () => go(b.dataset.view));

async function loadResults(src) {
  try {
    if (src === 'run' && SL()) {
      if (!LOCAL) throw new Error('Belum ada hasil training ulang di sesi browser ini.');
      R = LOCAL.results;
    } else R = await getJSON(`${API.results}?source=${src}`);
    SRC = src;
  } catch (e) {
    toast(e.message);
    if (src !== 'paper') return loadResults('paper');
    R = null;
  }
  document.querySelectorAll('#srcSeg button').forEach(b => b.classList.toggle('on', b.dataset.s === SRC));
  NEEDS_R.forEach(v => delete done[v]);
  decorate();
  go(CUR);
}

function decorate() {
  const m = R ? R.meta : {};
  const paper = SRC === 'paper';
  $('modeChip').textContent = paper ? '● Hasil paper' : '● Training ulang';
  $('modeChip').className = 'chip ' + (paper ? 'yellow' : 'green');
  $('engineChip').textContent = paper ? 'LSTM · BO · 6 fold' : `${(m.engine || '').toUpperCase()} · ${m.spec_label || ''}`;
  $('footTitle').textContent = paper ? 'Hasil paper' : 'Run aktif';
  $('lastRun').textContent = paper ? 'LPS Research Fair 2026' : (m.created || '–');
  $('lastRunSub').textContent = `Data s.d. ${m.data_end || '–'}${paper ? '' : ' · ' + (m.file || '')}`;
  ['exportBtn', 'exportBtn2'].forEach(id => {$(id).href = `${API.export}?source=${SRC}`; $(id).onclick = exportClick});
  $('srcNotice').innerHTML = paper
    ? `<div class="notice"><b>Mode Hasil Paper.</b>&nbsp;Angka berasal dari paper & lampiran (hasil olahan penulis) dan tidak dihitung ulang di server. Grafik deret historis ditampilkan sebagai gambar paper.</div>`
    : `<div class="notice g"><b>Mode Training Ulang.</b>&nbsp;Hasil dihitung ulang dari <b>${esc(m.file || 'data unggahan')}</b> (${m.data_start} s.d. ${m.data_end}, ${m.n_days} hari) dengan konfigurasi ringan — dapat berbeda dari paper.</div>`;
}

async function exportClick(ev) {
  if (!(SRC === 'run' && SL() && LOCAL)) return;          // GET biasa untuk mode paper / server persisten
  ev.preventDefault();
  try {
    const r = await fetch(API.export, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(LOCAL.results)});
    if (!r.ok) throw new Error('Ekspor gagal');
    const url = URL.createObjectURL(await r.blob()), a = document.createElement('a');
    a.href = url; a.download = `hasil_run_${LOCAL.results.meta.run_id || 'lokal'}.xlsx`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 2000);
  } catch (e) {toast(e.message)}
}

async function refreshHealth() {
  try {HEALTH = await getJSON(API.health)} catch (e) {HEALTH = {}}
  const has = SL() ? !!LOCAL : !!HEALTH.active_run;
  $('srcRun').disabled = !has;
  $('srcRun').title = has ? '' : 'Belum ada hasil training ulang';
  $('ovRunTxt') && ($('ovRunTxt').textContent = has ? `Run aktif: ${SL() ? LOCAL.results.meta.run_id : HEALTH.active_run}` : 'Unggah data harian baru → pipeline yang sama dijalankan di server (versi ringan).');
  $('ovRunStep') && $('ovRunStep').classList.toggle('done', has);
  $('tokField').style.display = HEALTH.token_required ? '' : 'none';
  $('dropSub').textContent = `Excel/CSV · maks. ${HEALTH.max_upload_mb || 20} MB`;
}
segBind('srcSeg', d => loadResults(d.s));

/* ---------- plugin: bayangan siklus pengetatan ---------- */
const cycleShade = {id: 'cycleShade', beforeDatasetsDraw(ch, _, o) {
  if (!o || !o.cycles) return; const {ctx, chartArea: a, scales: {x}} = ch; const labs = ch.data.labels; ctx.save();
  o.cycles.filter(c => c.dir > 0).forEach(c => {
    const s = labs.findIndex(l => l >= c.start.slice(0, 7)); let e = labs.findLastIndex(l => l <= c.end.slice(0, 7));
    if (s < 0 || e < 0 || e < s) return;
    const x1 = x.getPixelForValue(s), x2 = x.getPixelForValue(e);
    ctx.fillStyle = 'rgba(210,88,79,.07)'; ctx.fillRect(x1, a.top, Math.max(x2 - x1, 2), a.bottom - a.top)});
  ctx.restore()}};

/* ==========================================================================
   1. RINGKASAN
   ========================================================================== */
function bestModel(ev) {return ev.models.reduce((b, m) => (ev.cv[m] && ev.cv[m].rmse < ev.cv[b].rmse ? m : b), ev.models[0])}
RENDER.overview = () => {
  const K = R.kpi, F = R.forecast, ev = R.evaluation;
  $('kDep').innerHTML = f2(K.dep_last, 3) + '<small>%</small>';
  $('kDepD').textContent = isNum(K.dep_prev_month) ? sgn((K.dep_last - K.dep_prev_month) * 100, 1) + ' bp vs rata-rata bulan lalu' : K.date_last;
  $('kTbp').innerHTML = f2(K.tbp_last) + '<small>% · BI ' + f2(K.bi_last) + '%</small>';
  $('kTbpD').textContent = 'Spread TBP − BI ' + sgn((K.tbp_last - K.bi_last) * 100) + ' bp';
  const g = K.gap_last_bp;
  $('kGap').innerHTML = sgn(g, 1) + '<small> bp</small>';
  $('kGapD').textContent = g > 0 ? 'Breach — deposito di atas TBP' : 'Aman — di bawah TBP';
  $('kGapD').className = 'delta ' + (g > 0 ? 'neg' : '');
  $('kPbLab').textContent = `P(breach) ${F.months[F.months.length - 1]}`;
  $('kPb').innerHTML = pct(K.pbreach_end, 1);

  const M = R.series && R.series.monthly;
  if (M) {
    $('ovSeg').style.display = '';
    $('ovLineBox').innerHTML = '<div class="cv h300"><canvas id="ovLine"></canvas></div>';
    const drawLine = range => {
      const N = M.dates.length, off = range === 'all' ? 0 : Math.max(0, N - (+range)), sl = a => a.slice(off);
      mk('ovLine', {type: 'line', data: {labels: sl(M.dates), datasets: [
        {label: 'BI Rate', data: sl(M.bi), borderColor: C.g900, stepped: true},
        {label: 'Lending Facility', data: sl(M.lf), borderColor: C.g300, stepped: true, borderDash: [4, 3], borderWidth: 1.5},
        {label: 'Deposit Facility', data: sl(M.df), borderColor: C.g300, stepped: true, borderDash: [4, 3], borderWidth: 1.5},
        {label: 'TBP LPS', data: sl(M.tbp), borderColor: C.y500, stepped: true, borderWidth: 2.6},
        {label: 'Deposito 1M', data: sl(M.dep), borderColor: C.red, tension: .35, borderWidth: 2.2, spanGaps: false}]},
        options: {interaction: {mode: 'index', intersect: false}, plugins: {cycleShade: {cycles: R.descriptive.cycles}, legend: {position: 'bottom'},
          tooltip: {callbacks: {title: t => ym(t[0].label)}}},
          scales: {x: {ticks: {maxTicksLimit: 10, callback(v) {return ym(this.getLabelForValue(v))}}}, y: {ticks: {callback: v => v + '%'}}}}, plugins: [cycleShade]});
    };
    drawLine('all'); segBind('ovSeg', d => drawLine(d.r));
    const gap = M.dates.map((_, i) => isNum(M.dep[i]) ? (M.dep[i] - M.tbp[i]) * 100 : null);
    $('ovGapBox').innerHTML = '<div class="cv h220"><canvas id="ovGap"></canvas></div>';
    mk('ovGap', {type: 'bar', data: {labels: M.dates, datasets: [{data: gap, backgroundColor: gap.map(x => x > 0 ? C.red : C.g300), borderRadius: 3}]},
      options: {plugins: {legend: {display: false}, tooltip: {callbacks: {title: t => ym(t[0].label), label: c => sgn(c.raw, 1) + ' bp'}}},
        scales: {x: {ticks: {maxTicksLimit: 10, callback(v) {return ym(this.getLabelForValue(v))}}, grid: {display: false}}, y: {ticks: {callback: v => v + ' bp'}}}}});
    const last12 = gap.filter(isNum).slice(-12);
    $('ovBreachChip').textContent = `Breach ${last12.filter(x => x > 0).length}/12 bulan terakhir`;
  } else {
    $('ovSeg').style.display = 'none';
    $('ovLineBox').innerHTML = fig('paper/g_series.png', 'Gambar 3 paper · Lima suku bunga harian 2017–2026 dan siklus kebijakan BI');
    $('ovGapBox').innerHTML = fig('paper/g_gap.png', 'Gambar 4 paper · Gap Deposito 1M − TBP dan episode breach');
    $('ovBreachChip').textContent = `Breach ${f2(R.descriptive.gap.pos_pct, 1)}% hari (2020–2026)`;
  }

  const best = bestModel(ev), b = ev.cv[best];
  $('ovBest').innerHTML = `<div style="display:flex;align-items:center;gap:12px;padding:12px;border-radius:16px;background:var(--y100)">
    <div class="ico" style="background:var(--y500);color:var(--g900)">★</div>
    <div><b style="font-size:15px">${best}</b><div class="muted" style="font-size:12px">RMSE ${f2(b.rmse)} bp · R²<sub>OOS</sub> ${f2(b.r2oos, 3)}</div></div>
    <span class="tag" style="margin-left:auto">${ev.folds.filter(f => !f.holdout).length} fold</span></div>`;
  mk('ovRmse', {type: 'bar', data: {labels: ev.models, datasets: [{data: ev.models.map(m => ev.cv[m].rmse), backgroundColor: ev.models.map(m => m === best ? C.y500 : C.g100), borderRadius: 8}]},
    options: {indexAxis: 'y', plugins: {legend: {display: false}}, scales: {x: {ticks: {callback: v => v + ' bp'}}, y: {grid: {display: false}}}}});
  const m = R.meta;
  $('ovMeta').innerHTML = `<div class="kv"><span>Periode</span><span>${m.data_start} s.d. ${m.data_end}</span><span>Target</span><span>${esc(m.spec_label)}</span>
    <span>Proyeksi</span><span>${F.months[0]} – ${F.months[F.months.length - 1]}</span><span>Catatan</span><span style="white-space:normal">${(m.notes || []).map(esc).join('<br>')}</span></div>`;
  refreshHealth();
};

/* ==========================================================================
   2. DATA & TRAINING ULANG
   ========================================================================== */
let UPLOAD = null, JOB = null, LOGN = 0;
function drawSteps(cur = -1, fin = false) {
  const st = (CFG && CFG.steps) || [];
  $('steps').innerHTML = st.map((s, i) => `<div class="step ${fin || i < cur ? 'done' : i === cur ? 'run' : ''}"><div class="n">${fin || i < cur ? '✓' : i + 1}</div><div><b>${s}</b></div></div>`).join('');
}
function logLine(t, ts) {const l = $('log'); ts = ts || new Date().toLocaleTimeString('id-ID'); l.innerHTML += `<br><span class="t">[${ts}]</span> ${esc(t)}`; l.scrollTop = l.scrollHeight}
function checks(list) {$('checks').innerHTML = list.map(c => `<div class="check ${c[0]}"><i>${c[0] === 'ok' ? '✓' : c[0] === 'warn' ? '!' : '·'}</i>${esc(c[1])}</div>`).join('')}
function tokenHdr() {const t = $('cfTok').value; return t ? {'X-Retrain-Token': t} : {}}

let FILE = null;
async function handleFile(file) {
  if (!file) return;
  FILE = file;
  $('fileBox').innerHTML = `<div class="file-row"><div class="fi">${esc(file.name.split('.').pop().toUpperCase())}</div><div><b>${esc(file.name)}</b><div class="muted" style="font-size:12px">${(file.size / 1024).toFixed(0)} KB · memvalidasi…</div></div></div>`;
  const fd = new FormData(); fd.append('file', file);
  try {
    const j = await getJSON(API.upload, {method: 'POST', body: fd, headers: tokenHdr()});
    UPLOAD = j;
    $('fileBox').innerHTML = `<div class="file-row"><div class="fi">${esc(file.name.split('.').pop().toUpperCase())}</div><div><b>${esc(j.name)}</b><div class="muted" style="font-size:12px">${j.rows} hari · ${j.period[0]} s.d. ${j.period[1]}</div></div><span class="tag" style="margin-left:auto">Valid</span></div>`;
    checks(j.checks);
    table('prevTbl', ['Tanggal', 'BI', 'LF', 'DF', 'TBP', 'Deposito 1M', 'Gap (bp)'], j.preview.map(r => [r[0], f2(r[1]), f2(r[2]), f2(r[3]), f2(r[4]), f2(r[5], 3), `<span class="tag ${r[6] > 0 ? 'r' : ''}">${sgn(r[6], 1)}</span>`]));
    $('runBtn').disabled = false; $('pipeSub').textContent = 'Siap dijalankan';
    logLine(`Data ${j.name} tervalidasi (${j.rows} hari).`);
  } catch (e) {
    UPLOAD = null; $('runBtn').disabled = true;
    $('fileBox').innerHTML = `<div class="file-row" style="background:var(--red100)"><div class="fi" style="background:var(--red);color:#fff">!</div><div><b>Gagal</b><div class="muted" style="font-size:12px">${esc(e.message)}</div></div></div>`;
  }
}
function cfg() {
  return {spec: $('cfSpec').value, n_folds: +$('cfFolds').value, window: +$('cfWin').value, decomp_causal: $('cfDec').value,
    engine: $('cfEng').value, epochs: +$('cfEp').value, n_seeds: +$('cfSeeds').value, B: +$('cfB').value, holdout: $('cfHo').checked};
}
async function runPipeline() {
  if (!UPLOAD) return toast('Unggah data terlebih dahulu');
  $('runBtn').disabled = true;
  if (SL()) return runSync();
  try {
    const j = await getJSON(API.run, {method: 'POST', headers: {'Content-Type': 'application/json', ...tokenHdr()}, body: JSON.stringify({upload_id: UPLOAD.upload_id, config: cfg()})});
    JOB = j.job_id; LOGN = 0; $('log').innerHTML = '<span class="t">$</span> training ulang dimulai…';
    drawSteps(0); poll();
  } catch (e) {toast(e.message); $('runBtn').disabled = false}
}
async function runSync() {
  const fd = new FormData(); fd.append('file', FILE); fd.append('config', JSON.stringify(cfg()));
  $('log').innerHTML = '<span class="t">$</span> training ulang (mode serverless, satu request)…';
  const caps = HEALTH.serverless_caps || {};
  logLine(`Konfigurasi dibatasi agar selesai dalam batas waktu fungsi: ≤ ${caps.n_folds} fold, ≤ ${caps.epochs} epoch, 1 seed, MLP, EMD.`);
  let k = 0; drawSteps(0); $('pipeSub').textContent = 'Berjalan di server…';
  const tick = setInterval(() => {k = Math.min(k + 1, 95); $('pipeBar').style.width = k + '%'; $('pipePct').textContent = k + '%';
    drawSteps(Math.min(Math.floor(k / 11), CFG.steps.length - 1))}, 900);
  try {
    const j = await getJSON(API.run, {method: 'POST', body: fd, headers: tokenHdr()});
    clearInterval(tick);
    j.log.forEach(([ts, t]) => logLine(t, ts));
    LOCAL = {results: j.results, base_paths: j.base_paths};
    $('pipeBar').style.width = '100%'; $('pipePct').textContent = '100%'; $('pipeSub').textContent = 'Selesai'; drawSteps(0, true);
    toast('Training ulang selesai — menampilkan hasil baru');
    await refreshHealth(); drawRuns(); loadResults('run');
  } catch (e) {
    clearInterval(tick); $('pipeSub').textContent = 'Gagal'; logLine('ERROR: ' + e.message);
    toast(/504|timeout|tidak valid/i.test(e.message) ? 'Server kehabisan waktu — kurangi fold/epoch atau deploy di Render/Railway' : e.message);
  } finally {$('runBtn').disabled = false}
}
async function poll() {
  if (!JOB) return;
  let s;
  try {s = await getJSON(`${API.status}${JOB}?since=${LOGN}`)} catch (e) {toast(e.message); $('runBtn').disabled = false; return}
  s.log.forEach(([ts, t]) => logLine(t, ts)); LOGN = s.log_len;
  $('pipeBar').style.width = s.pct + '%'; $('pipePct').textContent = Math.round(s.pct) + '%';
  $('pipeSub').textContent = s.done ? (s.error ? 'Gagal' : 'Selesai') : (CFG.steps[s.step] || 'Berjalan…');
  drawSteps(s.step, s.done && !s.error);
  if (!s.done) return setTimeout(poll, 1500);
  JOB = null; $('runBtn').disabled = false;
  if (s.error) {logLine('ERROR: ' + s.error); toast('Training gagal: ' + s.error); return}
  toast('Training ulang selesai — menampilkan hasil baru');
  await refreshHealth(); await drawRuns(); loadResults('run');
}
async function drawRuns() {
  if (SL()) {
    $('runsBox').innerHTML = LOCAL ? `<div class="runs-row"><div class="grow"><b class="mono">${esc(LOCAL.results.meta.run_id)}</b><div class="muted">${esc(LOCAL.results.meta.file || '')} · data s.d. ${LOCAL.results.meta.data_end}</div></div><span class="tag">Aktif</span></div>` : '';
    $('runsBox').innerHTML += '<div class="notice" style="margin:10px 0 0">Server berjalan serverless (mis. Vercel): hasil training disimpan di tab browser ini saja dan hilang saat halaman dimuat ulang. Unduh Excel untuk menyimpannya.</div>';
    return;
  }
  let j; try {j = await getJSON(API.runs)} catch (e) {return}
  if (!j.runs.length) {$('runsBox').innerHTML = '<div class="empty">Belum ada run.</div>'; return}
  $('runsBox').innerHTML = j.runs.map(r => `<div class="runs-row"><div class="grow"><b class="mono">${esc(r.id)}</b><div class="muted">${esc(r.file || '')} · data s.d. ${r.data_end} · ${esc(r.spec || '')} · ${esc(r.engine || '')}</div></div>
    ${r.active ? '<span class="tag">Aktif</span>' : `<button class="btn ghost sm" data-act="${r.id}">Aktifkan</button><button class="btn ghost sm" data-del="${r.id}">Hapus</button>`}</div>`).join('');
  $('runsBox').querySelectorAll('[data-act]').forEach(b => b.onclick = async () => {
    try {await getJSON(`${API.runs}/${b.dataset.act}/activate`, {method: 'POST', headers: tokenHdr()}); await refreshHealth(); drawRuns(); if (SRC === 'run') loadResults('run'); toast('Run diaktifkan')} catch (e) {toast(e.message)}});
  $('runsBox').querySelectorAll('[data-del]').forEach(b => b.onclick = async () => {
    if (!confirm('Hapus run ' + b.dataset.del + '?')) return;
    try {await getJSON(`${API.runs}/${b.dataset.del}`, {method: 'DELETE', headers: tokenHdr()}); drawRuns(); toast('Run dihapus')} catch (e) {toast(e.message)}});
}
RENDER.data = () => {
  const d = CFG ? CFG.defaults : {};
  if (d.n_folds) {$('cfFolds').value = d.n_folds; $('cfWin').value = d.window; $('cfEp').value = d.epochs; $('cfSeeds').value = d.n_seeds; $('cfB').value = d.B}
  if (SL()) {const c = HEALTH.serverless_caps; $('cfFolds').value = c.n_folds; $('cfFolds').max = c.n_folds; $('cfEp').value = c.epochs; $('cfEp').max = c.epochs;
    $('cfSeeds').value = 1; $('cfSeeds').max = 1; $('cfB').value = c.B; $('cfB').max = c.B; $('cfEng').value = 'mlp'; $('cfDec').value = 'emd'; $('cfDec').disabled = true}
  if (CFG && CFG.engine === 'mlp') $('cfEng').querySelector('[value=lstm]').textContent = 'LSTM (PyTorch tidak terpasang → MLP)';
  drawSteps(); drawRuns();
  const drop = $('drop'), inp = $('fileIn');
  inp.onchange = () => handleFile(inp.files[0]);
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => {e.preventDefault(); drop.classList.add('over')}));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => {e.preventDefault(); drop.classList.remove('over')}));
  drop.addEventListener('drop', e => handleFile(e.dataTransfer.files[0]));
  $('runBtn').onclick = runPipeline;
};

/* ==========================================================================
   3. DEKOMPOSISI
   ========================================================================== */
const catTag = c => c.startsWith('Derau') || c.startsWith('Noise') ? 'n' : c.startsWith('Siklus RDG') ? 'y' : c.startsWith('Tren') ? '' : 'r';
function drawDecomp(key) {
  const D = R.decomposition[key];
  const tbl = D.table, res = tbl.find(r => r.comp.startsWith('Res'));
  $('dKpis').innerHTML = [
    ['Jumlah IMF', `${tbl.length - 1} <small>+ residual</small>`, 'CEEMDAN deskriptif', ''],
    ['Orthogonality index', f2(D.io, 3), 'mendekati 0 = komponen terpisah baik', ''],
    ['Variansi residual (tren)', f2(res ? res.var_pct : null, 1) + '<small>%</small>', 'porsi variansi ternormalisasi', 'warn'],
    ['Siklus menengah', tbl.filter(r => r.category.startsWith('Siklus menengah')).length + ' <small>komponen</small>', 'sinyal yang perlu direspons TBP', '']]
    .map(([l, v, s, c], i) => `<div class="card ${i === 3 ? 'green' : ''} kpi"><span class="lab">${l}</span><div class="val">${v}</div><span class="${i === 3 ? 'muted' : 'delta ' + c}">${s}</span></div>`).join('');
  if (D.comps && D.dates) {
    $('dBody').innerHTML = `<div class="cv h180"><canvas id="dOrig"></canvas></div><div class="imf-grid mt" id="imfGrid"></div>`;
    mk('dOrig', {type: 'line', data: {labels: D.dates, datasets: [{label: D.label, data: D.orig, borderColor: C.g800, fill: {target: 'origin', above: 'rgba(23,112,94,.08)'}}]},
      options: {plugins: {legend: {display: false}}, scales: {x: {ticks: {maxTicksLimit: 8}}}}});
    $('imfGrid').innerHTML = D.comps.map((_, i) => `<div class="imf"><div class="imf-h"><b>${tbl[i] ? tbl[i].comp : 'C' + (i + 1)}</b><span class="muted">${tbl[i] && isNum(tbl[i].period_m) ? f2(tbl[i].period_m, 1) + ' bln · ' : ''}${tbl[i] ? f2(tbl[i].var_pct, 1) + '%' : ''}</span></div><div class="cv h90"><canvas id="imf${i}"></canvas></div></div>`).join('');
    const pal = [C.red, C.orange, C.y500, C.g500, C.blue, C.purple, C.g700, C.g900, C.grey];
    D.comps.forEach((c, i) => mk('imf' + i, {type: 'line', data: {labels: D.dates, datasets: [{data: c, borderColor: pal[i % pal.length], borderWidth: 1.3}]},
      options: {animation: false, plugins: {legend: {display: false}, tooltip: {enabled: false}}, scales: {x: {display: false}, y: {ticks: {maxTicksLimit: 3}}}}}));
  } else {
    $('dBody').innerHTML = fig(D.figure, key === 'target' ? 'Gambar 5 paper · Dekomposisi CEEMDAN Deposito 1M (spesifikasi level)' : 'Lampiran Gambar A1 · Dekomposisi CEEMDAN gap Deposito − TBP');
  }
  table('imfTbl', ['Komponen', 'Periode (bulan)', 'Variansi (%)', 'Korelasi dgn asli', 'Kategori horizon', 'Implikasi kebijakan'],
    tbl.map(r => [`<b>${r.comp}</b>`, f2(r.period_m, 2), f2(r.var_pct, 2), f2(r.corr, 3), `<span class="tag ${catTag(r.category)}">${r.category}</span>`, `<span style="white-space:normal">${r.implication}</span>`]), 1);
}
RENDER.decomp = () => {
  document.querySelectorAll('#dSeg button').forEach(b => b.classList.toggle('on', b.dataset.s === 'target'));
  $('dSeg').querySelector('[data-s=target]').textContent = SRC === 'paper' ? 'Deposito 1M' : 'Target';
  drawDecomp('target'); segBind('dSeg', d => drawDecomp(d.s));
  const coh = R.decomposition.coherence;
  $('cohCard').style.display = coh ? '' : 'none';
  if (coh) table('cohTbl', ['Komponen Deposito 1M', 'Korelasi lag 0', 'Lag optimum', 'Korelasi lag optimum', 'Arah'],
    coh.map(r => [`<b>${r.comp}</b>`, f2(r.c0), r.lag, f2(r.copt), `<span class="tag ${r.dir.startsWith('TBP') ? '' : 'y'}">${r.dir}</span>`]));
};

/* ==========================================================================
   4. EVALUASI MODEL
   ========================================================================== */
function drawEval(scheme) {
  const ev = R.evaluation, E = ev[scheme], base = E['ARIMA'] && E['ARIMA'].rmse;
  const best = ev.models.reduce((b, m) => E[m].rmse < E[b].rmse ? m : b, ev.models[0]);
  table('evalTbl', ['Model', 'RMSE (bp)', 'MAE (bp)', 'MAPE (%)', 'ΔRMSE vs ARIMA', 'R²<sub>OOS</sub>', 'Akurasi arah', 'n'],
    ev.models.map(m => {const e = E[m]; return [(m === best ? '★ ' : '') + `<b>${m}</b>`, f2(e.rmse), f2(e.mae), f2(e.mape, 3),
      base ? sgn((1 - e.rmse / base) * 100, 1) + '%' : '–', f2(e.r2oos, 3), isNum(e.dir_acc) ? f2(e.dir_acc, 1) + '%' : '–', e.n ?? '–']}), 1, ev.models.indexOf(best));
  const hoF = ev.folds.find(f => f.holdout);
  $('evalSub').textContent = scheme === 'cv' ? `Rolling-origin ${ev.folds.filter(f => !f.holdout).length} fold · ★ = RMSE terendah` : `${hoF ? hoF.name : 'Holdout'} · ★ = RMSE terendah`;
}
RENDER.model = () => {
  const ev = R.evaluation, cl = ev.cv['CEEMDAN-LSTM'], ar = ev.cv['ARIMA'], ls = ev.cv['LSTM'];
  const dmA = ev.dm.find(d => d.pair === 'CEEMDAN-LSTM vs ARIMA') || {};
  const kp = [
    ['RMSE CEEMDAN-LSTM', f2(cl.rmse) + '<small> bp</small>', `vs LSTM ${f2(ls.rmse)} · ARIMA ${f2(ar.rmse)}`, 'green'],
    ['Perbaikan vs ARIMA', sgn((1 - cl.rmse / ar.rmse) * 100, 1) + '<small>%</small>', 'rolling-origin', ''],
    ['Uji DM vs ARIMA', f2(dmA.dm_se) , 'p = ' + pv(dmA.p_se).replace('&lt;', '<'), ''],
    ['Akurasi arah', isNum(cl.dir_acc) ? f2(cl.dir_acc, 1) + '<small>%</small>' : '–', 'hari dengan perubahan', 'yellow']];
  $('mKpis').innerHTML = kp.map(([l, v, s, c]) => `<div class="card ${c} kpi"><span class="lab" ${c === 'yellow' ? 'style="color:var(--g900)"' : ''}>${l}</span><div class="val">${v}</div><span class="${c === 'green' ? 'muted' : 'delta'}">${s}</span></div>`).join('');
  drawEval('cv');
  document.querySelectorAll('#mSeg button').forEach(b => b.classList.toggle('on', b.dataset.s === 'cv'));
  segBind('mSeg', d => drawEval(d.s));
  const folds = ev.folds;
  mk('mFold', {type: 'line', data: {labels: folds.map(f => f.name.replace('Holdout', 'HO')), datasets: ev.models.map(m => ({label: m, data: folds.map(f => f.metrics[m] ? f.metrics[m].rmse : null),
      borderColor: mcol(m), backgroundColor: mcol(m), pointRadius: 3, borderWidth: m === 'CEEMDAN-LSTM' ? 2.8 : 1.6}))},
    options: {interaction: {mode: 'index', intersect: false}, plugins: {legend: {position: 'bottom'}}, scales: {y: {ticks: {callback: v => v + ' bp'}}, x: {ticks: {maxRotation: 0, autoSkip: true}}}}});
  const S = ev.sample;
  if (S && S.actual) {
    $('mRight').innerHTML = cardH('Aktual vs prediksi', `${S.name} · one-step`) + '<div class="cv h260"><canvas id="mPred"></canvas></div>';
    mk('mPred', {type: 'line', data: {labels: S.dates, datasets: [{label: 'Aktual', data: S.actual, borderColor: C.g900, borderWidth: 2.4},
      ...Object.entries(S.preds).map(([m, p]) => ({label: m, data: p, borderColor: mcol(m), borderWidth: 1.5, borderDash: m === 'ARIMA' ? [4, 3] : []}))]},
      options: {interaction: {mode: 'index', intersect: false}, plugins: {legend: {position: 'bottom'}}, scales: {x: {ticks: {maxTicksLimit: 6}}, y: {ticks: {callback: v => v.toFixed(2) + '%'}}}}});
  } else {
    $('mRight').innerHTML = cardH('Mengapa hibrida ARIMA-CEEMDAN-LSTM?', 'Ringkasan bagian 4.4 paper') + `<ul class="why">${(ev.why_hybrid || []).map(t => `<li>${t}</li>`).join('')}</ul>`;
  }
  $('dmTbl').classList.add('wrap');
  table('dmTbl', ['Pasangan', 'DM (SE)', 'p', 'DM (AE)', 'p', 'Kesimpulan'],
    ev.dm.map(d => [`<b>${d.pair}</b>`, f2(d.dm_se), pv(d.p_se), f2(d.dm_ae), pv(d.p_ae),
      d.p_se < 0.05 ? `<span class="tag">${d.dm_se < 0 ? d.pair.split(' vs ')[0] : d.pair.split(' vs ')[1]} lebih akurat</span>` : '<span class="tag n">Tidak berbeda</span>']));
  table('foldTbl', ['Fold', 'Hari', 'λ', ...ev.models.slice(0, 3).map(m => m.replace('CEEMDAN-LSTM', 'C-LSTM'))],
    folds.map(f => [`<b>${f.name}</b>`, f.n_test, f2(f.lambda, 1), ...ev.models.slice(0, 3).map(m => f2(f.metrics[m] && f.metrics[m].rmse))]));
  $('horCard').style.display = ev.horizon ? '' : 'none';
  if (ev.horizon) table('horTbl', ['Horizon', 'RMSE CEEMDAN-LSTM', 'RMSE ARIMA', 'PI 95% (± bp)', '% |galat| > 12,5 bp', 'Kesesuaian kisi CL', 'Kesesuaian kisi ARIMA'],
    ev.horizon.map(h => [`<b>${h.h}</b>`, f2(h.cl, 1), f2(h.ar, 1), '±' + f2(h.pi, 1), f2(h.big, 1) + '%', f2(h.grid_cl, 1) + '%', f2(h.grid_ar, 1) + '%']));
};

/* ==========================================================================
   5. PROYEKSI
   ========================================================================== */
function drawForecast(key) {
  const F = R.forecast, s = F.series[key], mo = F.monthly[key];
  mk('fMain', {type: 'line', data: {labels: F.dates, datasets: [
    {label: 'PI 95% atas', data: s.hi, borderColor: 'transparent', backgroundColor: 'rgba(210,88,79,.12)', fill: '+1', pointRadius: 0},
    {label: 'PI 95% bawah', data: s.lo, borderColor: 'transparent', pointRadius: 0},
    {label: s.label, data: s.pred, borderColor: mcol(key), borderWidth: 2.6},
    {label: `TBP ditahan ${f2(F.tbp)}%`, data: F.dates.map(() => F.tbp), borderColor: C.y500, borderDash: [6, 4], borderWidth: 2},
    {label: `BI Rate ${f2(F.bi)}%`, data: F.dates.map(() => F.bi), borderColor: C.g900, borderDash: [2, 3], borderWidth: 1.4}]},
    options: {interaction: {mode: 'index', intersect: false}, plugins: {legend: {position: 'bottom', labels: {filter: l => !l.text.startsWith('PI 95% bawah')}}},
      scales: {x: {ticks: {maxTicksLimit: 8}}, y: {ticks: {callback: v => v.toFixed(2) + '%'}}}}});
  table('fTbl', ['Bulan', 'Hari kerja', 'Proyeksi', 'PI 95%', 'Gap vs TBP', 'P(breach)', 'TBP min', 'Penyesuaian'],
    F.months.map((m, h) => {const r = mo[h], g = (r.v - F.tbp) * 100; return [`<b>${m}</b>`, F.days ? F.days[h] : '–', f2(r.v, 3) + '%', `${f2(r.lo)}–${f2(r.hi)}`,
      `<span class="tag ${g > 0 ? 'r' : ''}">${sgn(g, 1)} bp</span>`, `<b>${pct(r.pb, 1)}</b>`, f2(r.tbpmin) + '%', sgn((r.tbpmin - F.tbp) * 100) + ' bp']}));
  const last = mo[mo.length - 1];
  $('fKpis').innerHTML = [
    ['Proyeksi ' + F.months[F.months.length - 1], f2(last.v, 3) + '<small>%</small>', `PI 95% ${f2(last.lo)}–${f2(last.hi)}%`, 'green'],
    ['P(breach) akhir', pct(last.pb, 1), `TBP ditahan ${f2(F.tbp)}%`, 'yellow'],
    ['TBP minimum maks.', f2(Math.max(...mo.map(r => r.tbpmin))) + '<small>%</small>', 'P(breach) ≤ 10% · kelipatan 25 bp', ''],
    ['λ koreksi LSTM', f2(F.lambda, 1), 'aturan 1-SE pada blok kalibrasi', '']]
    .map(([l, v, s, c]) => `<div class="card ${c} kpi"><span class="lab" ${c === 'yellow' ? 'style="color:var(--g900)"' : ''}>${l}</span><div class="val">${v}</div><span class="${c === 'green' ? 'muted' : 'delta'}">${s}</span></div>`).join('');
}
RENDER.forecast = () => {
  const F = R.forecast, keys = Object.keys(F.series);
  $('fSeg').innerHTML = keys.map(k => `<button class="${k === F.main ? 'on' : ''}" data-m="${esc(k)}">${esc(k.replace('CEEMDAN-LSTM ', 'C-L '))}</button>`).join('');
  drawForecast(F.main); segBind('fSeg', d => drawForecast(d.m));
  mk('fPb', {type: 'bar', data: {labels: F.months, datasets: keys.map(k => ({label: k, data: F.monthly[k].map(r => r.pb * 100), backgroundColor: mcol(k), borderRadius: 4}))},
    options: {plugins: {legend: {position: 'bottom'}}, scales: {y: {min: 0, max: 100, ticks: {callback: v => v + '%'}}, x: {grid: {display: false}}}}});
  $('fTblSub').textContent = (F.note ? F.note + ' ' : '') + 'TBP minimum = kelipatan 25 bp terkecil agar P(breach) ≤ 10%.';
  if (F.figure) {$('fFigCard').style.display = ''; $('fFigCard').innerHTML = fig(F.figure, 'Gambar 7 paper · Proyeksi harian Deposito 1M CEEMDAN-LSTM tiga spesifikasi dan PI 95%')}
  else $('fFigCard').style.display = 'none';
};

/* ==========================================================================
   6. EFEKTIVITAS & TRANSMISI
   ========================================================================== */
RENDER.effect = () => {
  const E = R.effectiveness.rows, all = E[0], pr = E.find(r => r.period.startsWith('Proyeksi'));
  $('eKpis').innerHTML = [
    ['Breach historis', f2(all.breach, 1) + '<small>% hari</small>', all.period, 'green'],
    ['Rata-rata gap', sgn(all.gap, 1) + '<small> bp</small>', `${all.episodes} episode · maks ${f2(all.depth, 1)} bp`, ''],
    ['Posisi koridor TBP', f2(all.corridor, 2), '0 = DF · 1 = LF', ''],
    ['Breach proyeksi', pr ? f2(pr.breach, 1) + '<small>% hari</small>' : '–', 'TBP ditahan', 'yellow']]
    .map(([l, v, s, c]) => `<div class="card ${c} kpi"><span class="lab" ${c === 'yellow' ? 'style="color:var(--g900)"' : ''}>${l}</span><div class="val">${v}</div><span class="${c === 'green' ? 'muted' : 'delta'}">${s}</span></div>`).join('');
  table('effTbl', ['Periode', 'Hari', 'Gap (bp)', 'Breach (% hari)', 'Episode', 'Kedalaman maks (bp)', 'Posisi koridor', 'ΔTBP/ΔBI', 'ΔDep/ΔBI'],
    E.map(r => [`<b>${r.period}</b>`, r.days, sgn(r.gap, 1), `<span class="tag ${r.breach > 10 ? 'r' : r.breach > 0 ? 'y' : ''}">${f2(r.breach, 1)}</span>`, r.episodes, f2(r.depth, 1), f2(r.corridor), f2(r.tbp_bi), f2(r.dep_bi)]));
  const M = R.series && R.series.monthly;
  if (M) {
    $('eLeft').innerHTML = cardH('Posisi TBP dalam koridor BI', '(TBP − DF)/(LF − DF) bulanan') + '<div class="cv h260"><canvas id="eCor"></canvas></div>';
    mk('eCor', {type: 'line', data: {labels: M.dates, datasets: [{label: 'Posisi koridor', data: R.series.corridor, borderColor: C.g800, stepped: true, fill: {target: {value: 0}, above: 'rgba(23,112,94,.08)', below: 'rgba(210,88,79,.12)'}},
      {label: 'BI Rate', data: M.dates.map(() => 0.4286), borderColor: C.y500, borderDash: [5, 4], borderWidth: 1.4}]},
      options: {plugins: {legend: {position: 'bottom'}, tooltip: {callbacks: {title: t => ym(t[0].label)}}}, scales: {x: {ticks: {maxTicksLimit: 8, callback(v) {return ym(this.getLabelForValue(v))}}}}}});
    $('eRight').innerHTML = cardH('Siklus kebijakan BI', 'Teridentifikasi dari perubahan BI Rate') + '<div class="tbl-wrap"><table id="cycTbl"></table></div>';
    table('cycTbl', ['Siklus', 'Periode', 'n', 'Total (bp)'], R.descriptive.cycles.map(c => [`<b>${c.name}</b>`, `${c.start.slice(0, 7)} – ${c.end.slice(0, 7)}`, c.n, sgn(c.total_bp)]));
  } else {
    $('eLeft').innerHTML = cardH('Posisi TBP dalam koridor BI', 'Lampiran Gambar C5') + fig('paper/g_corridor.png', 'Posisi TBP dalam koridor suku bunga BI');
    $('eRight').innerHTML = cardH('Timeline episode breach', 'Lampiran Gambar C6') + fig('paper/g_timeline.png', 'Timeline episode breach');
  }
  const T = R.transmission;
  table('ardlTbl', ['ARDL/ECM · jalur', '(p, q)', 'F bounds', 'Kesimpulan', 'θ jangka panjang', 'p θ', 'γ₀', 'ECT α', 'Half-life (bln)'],
    T.ardl.map(a => a.error ? [`<b>${a.path}</b>`, '–', '–', esc(a.error), '–', '–', '–', '–', '–'] :
      [`<b>${a.path}</b>`, `(${a.p}, ${a.q})`, f2(a.F, 2), a.coint ? '<span class="tag">Kointegrasi</span>' : a.inconclusive ? '<span class="tag y">Tidak konklusif</span>' : '<span class="tag n">Tidak ada</span>',
        f2(a.theta, 3), pv(a.p_theta), f2(a.g0, 3), f2(a.alpha, 3), f2(a.hl, 1)]));
  table('nardlTbl', ['NARDL · jalur', '(p, q)', 'F bounds', 'Kesimpulan', 'θ⁺ (BI naik)', 'θ⁻ (BI turun)', 'Wald LR (p)', 'ECT α', 'Half-life (bln)'],
    T.nardl.map(a => a.error ? [`<b>${a.path}</b>`, '–', '–', esc(a.error), '–', '–', '–', '–', '–'] :
      [`<b>${a.path}</b>`, `(${a.p}, ${a.q})`, f2(a.F, 2), a.coint ? '<span class="tag">Kointegrasi</span>' : a.inconclusive ? '<span class="tag y">Tidak konklusif</span>' : '<span class="tag n">Tidak ada</span>',
        f2(a.theta_pos, 3), f2(a.theta_neg, 3), pv(a.lr_p), f2(a.alpha, 3), f2(a.hl, 1)]));
  const mu = T.multipliers, lab = mu.ardl.map((_, i) => 'Bulan ' + i);
  mk('eMult', {type: 'line', data: {labels: lab, datasets: [
    {label: 'ARDL', data: mu.ardl, borderColor: C.g800, pointRadius: 3}, {label: 'NARDL θ⁺ (naik)', data: mu.nardl_pos, borderColor: C.red, pointRadius: 3},
    {label: 'NARDL θ⁻ (turun)', data: mu.nardl_neg, borderColor: C.blue, pointRadius: 3}]},
    options: {plugins: {legend: {position: 'bottom'}}, scales: {y: {ticks: {callback: v => f2(v, 2)}}}}});
  $('eFig2').innerHTML = T.figure ? cardH('Rolling pass-through kumulatif', 'Lampiran Gambar C3 · jendela 24 bulan') + fig(T.figure, 'Pass-through BI Rate → TBP & Deposito')
    : cardH('Interpretasi', 'Dibaca dari hasil ARDL') + `<p class="muted" style="font-size:13px">Half-life menunjukkan berapa bulan yang dibutuhkan untuk menutup separuh deviasi dari keseimbangan jangka panjang. Multiplier di kiri menjadi input langkah 3 simulasi.</p>`;
};

/* ==========================================================================
   7. SIMULASI (dihitung di server)
   ========================================================================== */
const S = {shock: 0, rule: 'C', adj: 0, lag: 1, tr: 'ardl', tol: 10, B: 1000, stoch: false};
let simTimer, simSeq = 0;
function runSim() {clearTimeout(simTimer); simTimer = setTimeout(doSim, 220)}
async function doSim() {
  const seq = ++simSeq;
  let o;
  try {
    o = await getJSON(API.simulate, {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({source: SRC === 'run' && SL() ? 'inline' : SRC, context: SRC === 'run' && SL() ? R.simulation.context : undefined,
        base_paths: SRC === 'run' && SL() ? LOCAL.base_paths : undefined, shock: S.shock, rule: S.rule, adj: S.adj, lag: S.lag, transmission: S.tr, tol: S.tol, B: S.B, stochastic: S.stoch, heat: true})});
  } catch (e) {return toast(e.message)}
  if (seq !== simSeq) return;
  const tol = S.tol / 100, H = o.months.length;
  const p = o.pmax;
  const col = p > 0.5 ? C.red : p > tol ? C.y500 : C.g500;
  $('gauge').style.background = `conic-gradient(${col} ${p * 360}deg, var(--g100) 0)`;
  $('gVal').textContent = pct(p); $('gSub').textContent = o.pmax_month;
  $('gTag').textContent = p > 0.5 ? 'Risiko tinggi' : p > tol ? 'Di atas ambang' : 'Di bawah ambang';
  $('gTag').className = 'delta ' + (p > 0.5 ? 'neg' : p > tol ? 'warn' : '');
  $('tolLab').textContent = S.tol;
  $('sTbpMin').innerHTML = f2(o.tbp_min_max) + '<small style="color:#BFE0D4">%</small>';
  $('sTbpMinSub').textContent = 'puncak kebutuhan · ' + o.tbp_min_month;
  $('sAdjNeed').textContent = sgn(o.adj_needed_bp) + ' bp';
  $('sGapEnd').innerHTML = sgn(o.gap_end_bp, 1) + '<small> bp</small>';
  $('sGapTag').textContent = o.gap_end_bp > 0 ? 'Breach' : 'Aman'; $('sGapTag').className = 'delta ' + (o.gap_end_bp > 0 ? 'neg' : '');
  $('sTbpEnd').textContent = f2(o.tbp_end) + '%';
  mk('sFan', {type: 'line', data: {labels: o.months, datasets: [
    {label: 'P95', data: o.q95, borderColor: 'transparent', backgroundColor: 'rgba(210,88,79,.10)', fill: '+3'},
    {label: 'P75', data: o.q75, borderColor: 'transparent', backgroundColor: 'rgba(210,88,79,.20)', fill: '+1'},
    {label: 'P25', data: o.q25, borderColor: 'transparent'},
    {label: 'P05', data: o.q05, borderColor: 'transparent'},
    {label: 'Median Deposito', data: o.q50, borderColor: C.red, borderWidth: 2.4, pointRadius: 3},
    {label: 'TBP skenario', data: o.tbp_med, borderColor: C.y500, stepped: true, borderWidth: 2.4}]},
    options: {animation: {duration: 250}, plugins: {legend: {position: 'bottom', labels: {filter: l => !/^P\d/.test(l.text)}}}, scales: {y: {ticks: {callback: v => v.toFixed(2) + '%'}}}}});
  mk('sPb', {type: 'bar', data: {labels: o.months, datasets: [
    {type: 'line', label: 'Ambang', data: Array(H).fill(S.tol), borderColor: C.g900, borderDash: [5, 4], borderWidth: 1.5},
    {label: 'P(breach)', data: o.pb.map(x => x * 100), backgroundColor: o.pb.map(x => x > 0.5 ? C.red : x > tol ? C.y500 : C.g500), borderRadius: 10}]},
    options: {animation: {duration: 250}, plugins: {legend: {display: false}}, scales: {y: {min: 0, max: 100, ticks: {callback: v => v + '%'}}, x: {grid: {display: false}}}}});
  table('simTbl', ['Bulan', 'BI', 'TBP', 'Deposito', 'Gap (bp)', 'P(breach)', 'TBP min'],
    o.table.map(r => [`<b>${r.month}</b>`, f2(r.bi), f2(r.tbp), f2(r.dep, 3), `<span class="tag ${r.gap > 0 ? 'r' : ''}">${sgn(r.gap, 1)}</span>`, pct(r.pb), f2(r.tbp_min)]));
  if (o.heat) drawHeat(o.heat);
}
function drawHeat(h) {
  const lab = {A: 'A · Spread', C: 'C · Tahan', D: 'D · Dep+25'};
  let s = '<div></div>' + h.rules.map(r => `<div class="hd">${lab[r] || r}</div>`).join('');
  h.shocks.forEach((sh, i) => {
    s += `<div class="rl">${isNum(sh) ? (sh > 0 ? '+' : '') + sh + ' bp' : sh}</div>`;
    h.rules.forEach((rl, j) => {const p = h.pmax[i][j];
      const bg = p > 0.5 ? `rgba(210,88,79,${0.25 + p * 0.75})` : p > S.tol / 100 ? `rgba(242,193,78,${0.4 + p})` : `rgba(47,143,120,${0.25 + (1 - p) * 0.5})`;
      const on = sh === S.shock && rl === S.rule ? 'outline:3px solid #0D4A3E;outline-offset:-3px;' : '';
      s += `<div class="cell" style="background:${bg};color:${p > 0.5 ? '#fff' : '#16241F'};${on}">${f2(p * 100, 0)}%</div>`})});
  $('heat').innerHTML = s;
}
function rangeFill(el) {const p = (el.value - el.min) / (el.max - el.min) * 100; el.style.setProperty('--p', p + '%')}
let simBound = false;
function bindSim() {
  if (simBound) return; simBound = true;
  const map = [['sShock', 'shock', v => (v > 0 ? '+' : '') + v + ' bp'], ['sAdj', 'adj', v => (v >= 0 ? '+' : '') + v + ' bp'], ['sLag', 'lag', v => 'bulan ke-' + v], ['sTol', 'tol', v => v + '%'], ['sB', 'B', v => v]];
  map.forEach(([id, key, fmt]) => {const el = $(id), out = $('o' + id.slice(1)); const upd = () => {S[key] = +el.value; out.textContent = fmt(el.value); rangeFill(el); runSim()}; el.oninput = upd; rangeFill(el)});
  document.querySelectorAll('#tbpOpts .opt').forEach(b => b.onclick = () => {document.querySelectorAll('#tbpOpts .opt').forEach(x => x.classList.remove('on')); b.classList.add('on'); S.rule = b.dataset.o; runSim()});
  segBind('trSeg', d => {S.tr = d.t; runSim()});
  $('sStoch').onchange = e => {S.stoch = e.target.checked; runSim()};
}
function resetSim() {
  Object.assign(S, {shock: 0, rule: 'C', adj: 0, lag: 1, tr: 'ardl', tol: 10, B: 1000, stoch: false});
  [['sShock', 0], ['sAdj', 0], ['sLag', 1], ['sTol', 10], ['sB', 1000]].forEach(([id, v]) => {$(id).value = v; rangeFill($(id))});
  $('oShock').textContent = '0 bp'; $('oAdj').textContent = '+0 bp'; $('oLag').textContent = 'bulan ke-1'; $('oTol').textContent = '10%'; $('oB').textContent = '1000';
  $('sStoch').checked = false;
  document.querySelectorAll('#tbpOpts .opt').forEach(x => x.classList.toggle('on', x.dataset.o === 'C'));
  document.querySelectorAll('#trSeg button').forEach(x => x.classList.toggle('on', x.dataset.t === 'ardl'));
  runSim();
}
RENDER.sim = () => {
  bindSim();
  const Sm = R.simulation;
  $('simNotice').innerHTML = Sm.approx ? `<div class="notice"><b>Catatan mode paper.</b>&nbsp;Jalur bootstrap paper tidak dipublikasikan, sehingga simulator interaktif memakai jalur dasar yang direkonstruksi dari rata-rata & PI 95% bulanan proyeksi level (aproksimasi). Angka resmi ada pada matriks di bawah (Tabel 13 / Lampiran C11).</div>` : '';
  $('mxSub').textContent = Sm.approx ? 'Angka paper · Tabel 13 dan Lampiran C11' : 'Dihitung saat training ulang · transmisi ARDL';
  table('mxTbl', ['Skenario BI', 'A · BI + spread', 'C · Ditahan', 'D · Dep bln lalu + 25 bp', 'TBP minimum akhir', 'Δ vs TBP kini'],
    Sm.matrix.map(r => {const sh = isNum(r.shock) ? (r.shock > 0 ? '+' : '') + r.shock + ' bp' : 'Stokastik (CIR)';
      const c = x => `${f2(x.end * 100, 0)} / ${f2(x.max * 100, 0)}%`;
      return [`<b>${sh}</b>`, c(r.A), c(r.C), c(r.D), f2(r.tbp_min_end) + '%', sgn((r.tbp_min_end - Sm.context.tbp_last) * 100) + ' bp']}));
  runSim();
};

/* ==========================================================================
   8. REKOMENDASI
   ========================================================================== */
RENDER.rec = () => {
  const m = R.recommendations.main, a = R.transmission.ardl.find(x => x.path === 'BI Rate → Deposito 1M' && !x.error);
  $('rKpis').innerHTML = `
    <div class="card green kpi"><span class="lab">Rekomendasi utama</span><div class="val" style="font-size:24px">TBP ${f2(m.tbp_now)}% (${sgn(m.adj_now_bp)} bp)</div><span class="muted">pada penetapan terdekat agar P(breach) ≤ 10%</span></div>
    <div class="card kpi"><span class="lab">Puncak kebutuhan TBP</span><div class="val">${f2(m.tbp_peak)}<small>% · ${m.peak_month}</small></div><span class="delta warn">${sgn(m.adj_peak_bp)} bp dari TBP kini</span></div>
    <div class="card yellow kpi"><span class="lab" style="color:var(--g900)">Half-life BI → Deposito</span><div class="val">${a ? f2(a.hl, 1) : '–'} <small style="color:var(--g800)">bulan</small></div><span class="delta" style="background:rgba(13,74,62,.12)">θ = ${a ? f2(a.theta, 2) : '–'} · ARDL-ECM</span></div>`;
  $('recRows').innerHTML = R.recommendations.rows.map((r, i) => `<div class="rec-row"><span class="n">${i + 1}</span><div><b>${esc(r[0])}</b></div><div><small style="font-size:12.5px">${esc(r[1])}</small></div><div><span class="tag ${i === 0 ? 'y' : ''}" style="white-space:normal">${esc(r[2])}</span></div><div><small style="font-size:12.5px;color:var(--ink);font-weight:600">${esc(r[3])}</small></div></div>`).join('');
  const rules = R.simulation.rules;
  table('ruleTbl', ['Aturan', 'Breach (% bulan)', 'Rata-rata gap (bp)', 'Perubahan TBP (kali)', 'TBP akhir horizon', 'Δ vs TBP kini'],
    rules.map(r => [`<b>${r.rule}</b>`, f2(r.breach, 1), sgn(r.gap, 1), r.changes, f2(r.rec) + '%', sgn(r.delta) + ' bp']), 1,
    rules.reduce((b, r, i) => r.breach < rules[b].breach ? i : b, 0));
};

/* ==========================================================================
   9. PROFIL (sumber: static/data/profile.json)
   ========================================================================== */
RENDER.profile = async () => {
  let P; try {P = await getJSON(API.profile)} catch (e) {$('pfBody').innerHTML = '<div class="empty">Profil tidak dapat dimuat.</div>'; return}
  const list = (t, s, items, fn) => items && items.length ? `<div class="card">${cardH(t, s)}<div class="pf-list">${items.map(fn).join('')}</div></div>` : '';
  const link = (u, t) => u ? `<a href="${esc(u)}" target="_blank" rel="noopener">${t}</a>` : t;
  const hero = `<div class="card green"><div class="pf-hero">
    <div class="pf-photo" id="pfPhoto">${P.portrait ? `<img src="${esc(P.portrait)}" alt="${esc(P.name)}" onerror="this.parentNode.textContent='${esc(P.initials || 'NS')}'">` : esc(P.initials || '')}</div>
    <div><h2>${esc(P.name)}</h2><div class="role">${esc(P.role)} · ${esc(P.location)}</div>${(P.about || []).map(p => `<p>${esc(p)}</p>`).join('')}
      <div class="pf-links">${(P.contacts || []).map(c => `<a href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.label)}</a>`).join('')}</div></div></div></div>`;
  const facts = `<div class="grid g4 mt">${(P.facts || []).map((f, i) => `<div class="card ${i === 3 ? 'yellow' : ''} kpi"><span class="lab" ${i === 3 ? 'style="color:var(--g900)"' : ''}>${esc(f.label)}</span><div class="val" style="font-size:18px;letter-spacing:-.2px">${esc(f.value)}</div></div>`).join('')}</div>`;
  const exp = `<div class="card">${cardH('Pengalaman', 'Riset, industri, dan layanan publik')}<div class="tl">${(P.experience || []).map(e => `<div class="tl-i"><span class="per">${esc(e.period)}</span><b>${esc(e.title)}</b><small>${esc(e.org)}</small>${e.desc ? `<ul class="why" style="margin-top:6px">${e.desc.map(d => `<li style="font-size:12.3px">${esc(d)}</li>`).join('')}</ul>` : ''}</div>`).join('')}</div></div>`;
  const edu = `<div class="card">${cardH('Pendidikan', '')}<div class="tl">${(P.education || []).map(e => `<div class="tl-i"><span class="per">${esc(e.period)}</span><b>${esc(e.degree)}</b><small>${esc(e.school)}${e.note ? ' · ' + esc(e.note) : ''}</small></div>`).join('')}</div></div>`;
  const skills = P.skills && P.skills.length ? `<div class="card mt">${cardH('Keahlian', 'Tingkat profisiensi')}${P.skills.map(g => `<div style="margin-bottom:10px"><div class="nav-label" style="color:var(--g700);padding:6px 0">${esc(g.group)}</div>${g.items.map(s => `<div class="skill"><div class="ctrl-h"><span>${esc(s.name)}</span><output>${s.pct}%</output></div><div class="bar y"><span style="width:${s.pct}%"></span></div></div>`).join('')}</div>`).join('')}</div>` : '';
  const interests = P.interests && P.interests.length ? `<div class="card mt">${cardH('Minat penelitian', '')}<div style="display:flex;flex-wrap:wrap;gap:6px">${P.interests.map(t => `<span class="tag">${esc(t)}</span>`).join('')}</div></div>` : '';
  const pubs = list('Publikasi ilmiah', `${(P.publications || []).length} karya`, P.publications, p => `<div class="it"><b>${link(p.url, esc(p.title))}</b><small>${esc(p.venue)} · ${esc(p.year)}${p.authors ? ' · ' + esc(p.authors) : ''}</small>${p.badge ? ` <span class="tag y">${esc(p.badge)}</span>` : ''}</div>`);
  const projs = P.projects && P.projects.length ? `<div class="card mt">${cardH('Proyek, aplikasi & dashboard', '')}<div class="grid g3">${P.projects.map(p => `<div class="imf"><div class="imf-h"><b>${link(p.url, esc(p.title))}</b><span class="muted">${esc(p.year)}</span></div><div class="muted" style="font-size:12.3px;margin:4px 0 8px">${esc(p.desc)}</div><div style="display:flex;gap:4px;flex-wrap:wrap">${(p.tags || []).map(t => `<span class="tag n">${esc(t)}</span>`).join('')}</div></div>`).join('')}</div></div>` : '';
  const certs = list('Sertifikasi', '', P.certifications, c => `<div class="it"><b>${esc(c.name)}</b><small>${esc(c.issuer)} · ${esc(c.year)}</small></div>`);
  const awards = list('Penghargaan', '', P.awards, c => `<div class="it"><b>${esc(c.name)}</b><small>${esc(c.issuer)} · ${esc(c.year)}</small></div>`);
  const orgs = list('Organisasi', '', P.organizations, o => `<div class="it"><b>${esc(o.role)}</b><small>${esc(o.org)} · ${esc(o.year)}</small></div>`);
  const vol = list('Kegiatan sukarela', '', P.volunteering, o => `<div class="it"><b>${esc(o.role)}</b><small>${esc(o.org)} · ${esc(o.year)}</small></div>`);
  const speak = list('Topik pembicara', 'Webinar · kuliah tamu · workshop', (P.speaking || []).map(t => ({t})), o => `<div class="it"><b>${esc(o.t)}</b></div>`);
  $('pfBody').innerHTML = hero + facts + `<div class="grid g21 mt"><div>${exp}</div><div>${edu}${skills}${interests}</div></div>`
    + `<div class="mt">${pubs}</div>` + projs + `<div class="grid g2 mt"><div>${certs}</div><div>${awards}<div class="mt">${orgs}</div><div class="mt">${vol}</div></div></div>`
    + `<div class="mt">${speak}</div>`;
};

/* ==========================================================================
   INIT
   ========================================================================== */
(async () => {
  try {CFG = await getJSON(API.config)} catch (e) {CFG = {steps: [], defaults: {}}}
  await refreshHealth();
  const want = new URLSearchParams(location.search).get('source');
  await loadResults(want === 'run' && HEALTH.active_run ? 'run' : 'paper');
})();

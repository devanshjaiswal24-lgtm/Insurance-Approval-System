const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct = x => x == null ? "–" : (x * 100).toFixed(1) + "%";
const f3 = x => x == null ? "–" : Number(x).toFixed(3);
let INFO = null, charts = {}, loaded = {};
const NAVY = "#0b2a55", BLUE = "#2f6db5", RED = "#b3261e", GREEN = "#1b7a3d";

function toast(msg, kind = "") { const d = document.createElement("div"); d.className = kind; d.textContent = msg; $("#toast").append(d); setTimeout(() => d.remove(), 4500); }
async function api(path, opts) {
  let r; try { r = await fetch("/api" + path, opts); } catch { throw new Error("Cannot reach the backend. Is `python app.py` running?"); }
  const j = await r.json().catch(() => null);
  if (!j || !j.success) { const e = new Error(j?.error?.message || "Request failed"); e.details = j?.error?.details; throw e; }
  return j.data;
}
const post = (p, body) => api(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
function chart(id, cfg) { charts[id]?.destroy(); charts[id] = new Chart($("#" + id), cfg); }
function barH(id, labels, vals, colors, title) {
  chart(id, { type: "bar", data: { labels, datasets: [{ data: vals, backgroundColor: colors }] },
    options: { indexAxis: "y", plugins: { legend: { display: false }, title: { display: !!title, text: title } }, scales: { x: { grid: { display: false } } } } });
}

/* ---------- routing ---------- */
function route() {
  const id = (location.hash || "#home").slice(1); const page = document.getElementById(id) ? id : "home";
  document.querySelectorAll(".page").forEach(p => p.classList.toggle("on", p.id === page));
  document.querySelectorAll("nav a").forEach(a => a.classList.toggle("on", a.getAttribute("href") === "#" + page));
  ({ home: loadHome, xai: loadXai, notice: loadSelectors, fairness: loadFair, evaluation: loadEval, predict: () => {} })[page]();
}
window.addEventListener("hashchange", route);

/* ---------- home ---------- */
async function ensureInfo() { if (!INFO) INFO = await api("/model-info"); return INFO; }
async function loadHome() {
  try {
    const m = await ensureInfo(), t = m.test_metrics, d = m.dataset, h = await api("/health");
    const card = (v, l) => `<div class="card"><b>${esc(v)}</b><span>${esc(l)}</span></div>`;
    $("#cards").innerHTML = card(h.model_available ? "Ready" : "Missing", "Model status") + card(m.selected_model.replace(" (baseline)", ""), "Selected model") + card(d.records_clean, "Records (clean)") +
      card(t.confusion_matrix.flat().reduce((a, b) => a + b), "Test records evaluated") + card(pct(t.accuracy), "Test accuracy") + card(f3(t.roc_auc), "Test ROC-AUC") + card(f3(t.recall), "Recall (claims)") + card(f3(t.precision), "Precision (claims)");
    $("#dsSummary").innerHTML = `${d.records_raw} raw → ${d.records_clean} records after removing ${d.duplicates_removed} duplicates; ${d.columns} columns; claim rate ${pct(d.claim_rate)} (imbalanced).<br>Train ${d.train_size} / test ${d.test_size}. <b>${esc(m.data_note)}</b>`;
    const g = await api("/global-explanation"); const k = Object.keys(g.importance);
    barH("homeImp", k.map(x => g.labels[x]), k.map(x => g.importance[x]), NAVY);
  } catch (e) { toast(e.message, "e"); $("#cards").innerHTML = `<div class="card err">${esc(e.message)}</div>`; }
}

/* ---------- predict form ---------- */
const EXAMPLE = { customer_age: 24, vehicle_age: 8, vehicle_type: "Sports", annual_mileage: 16000, driving_experience: 5, previous_claims: 2, previous_accidents: 2, years_insured: 2, vehicle_value: 22000, policy_type: "Comprehensive", region_risk: "Medium" };
async function buildForm() {
  const m = await ensureInfo(); let h = "";
  for (const f of m.features) {
    if (m.categories[f]) h += `<div><label for="${f}">${esc(m.labels[f])}</label><select id="${f}" name="${f}" required>${m.categories[f].map(o => `<option>${esc(o)}</option>`).join("")}</select></div>`;
    else { const [lo, hi] = m.ranges[f]; h += `<div><label for="${f}">${esc(m.labels[f])}</label><input id="${f}" name="${f}" type="number" min="${lo}" max="${hi}" step="any" required><div class="hint">${lo}–${hi}</div></div>`; }
  }
  $("#fields").innerHTML = h; fill(EXAMPLE);
}
const fill = o => Object.entries(o).forEach(([k, v]) => { const el = document.getElementById(k); if (el) el.value = v; });
$("#exBtn").onclick = () => fill(EXAMPLE);
$("#form").addEventListener("submit", async ev => {
  ev.preventDefault(); const btn = $("#goBtn"), err = $("#formErr"); err.textContent = "";
  document.querySelectorAll("input").forEach(i => i.classList.remove("bad"));
  const body = {}; let bad = false;
  for (const f of INFO.features) { const el = document.getElementById(f); body[f] = el.value; if (el.type === "number" && (el.value === "" || +el.value < el.min || +el.value > el.max)) { el.classList.add("bad"); bad = true; } }
  if (bad) { err.textContent = "Please correct the highlighted fields (values must be within the stated ranges)."; return; }
  btn.disabled = true; btn.innerHTML = '<span class="spin"></span> Predicting…';
  try { const r = await post("/predict", body); showResult(r); toast("Prediction complete", "s"); loaded = {}; }
  catch (e) { err.innerHTML = esc(e.message) + (e.details ? "<ul>" + e.details.map(d => `<li>${esc(d)}</li>`).join("") + "</ul>" : ""); toast(e.message, "e"); }
  finally { btn.disabled = false; btn.textContent = "Predict claim risk"; }
});
function contribChart(id, rows, unit) {
  const r = rows.slice(0, 8);
  barH(id, r.map(x => `${x.label} = ${x.value}`), r.map(x => x.contribution), r.map(x => x.contribution > 0 ? RED : GREEN), `SHAP contribution (${unit}); red = raises risk, green = lowers`);
}
function showResult(r) {
  $("#result").innerHTML = `<h2>Result <span class="tag">#${r.id}</span></h2>
  <div class="row" style="align-items:center"><div class="big">${pct(r.probability)}</div><span class="pill ${r.category}">${r.category} risk</span></div>
  <p class="small">Estimated probability of a claim in the next 12 months. Baseline (average customer): ${pct(r.base_probability)}.<br>Category rule: Low &lt; ${pct(r.risk_thresholds.low_below)}, Moderate ${pct(r.risk_thresholds.low_below)}–${pct(r.risk_thresholds.high_from)}, High ≥ ${pct(r.risk_thresholds.high_from)} (configurable in config.py).</p>
  <p><b>Recommendation:</b> ${esc(r.recommendation)}</p>
  <h2>Top 3 influential factors</h2><ol>${r.top_factors.map(f => `<li><b>${esc(f.label)}</b> = ${esc(f.value)} — <span class="${f.contribution > 0 ? "up" : "down"}">${f.direction} predicted risk</span> (${f.contribution > 0 ? "+" : ""}${f.contribution.toFixed(2)})</li>`).join("")}</ol>
  <canvas id="resChart" height="230" aria-label="Local feature contributions" role="img"></canvas>
  <h2>Plain-language explanation</h2><p>${esc(r.explanation)}</p>
  <p class="small">Predicted at ${esc(r.timestamp)}. ${esc(r.disclaimer)}</p><a class="btn alt" href="#notice" style="text-decoration:none;display:inline-block">Open explanation report</a>`;
  contribChart("resChart", r.contributions, r.unit); localStorage.setItem("lastPred", r.id);
}

/* ---------- selectors (history) ---------- */
async function loadSelectors() {
  const list = await api("/predictions").catch(() => []);
  for (const id of ["#xaiSel", "#ntSel"]) $(id).innerHTML = list.length ? list.map(p => `<option value="${p.id}">#${p.id} · ${pct(p.probability)} · ${p.category} · ${p.timestamp}</option>`).join("") : "<option value=''>No predictions yet</option>";
}
async function loadXai() {
  await loadSelectors();
  if (loaded.xai) return; 
  try {
    const g = await api("/global-explanation"), k = Object.keys(g.importance);
    barH("gImp", k.map(x => g.labels[x]), k.map(x => g.importance[x]), NAVY, `Mean |SHAP| (${g.unit})`);
    $("#shapImg").src = g.images.summary + "?t=" + Date.now();
    $("#gDir").innerHTML = "<h2>Direction of effect</h2><ul>" + Object.entries(g.direction).map(([f, d]) => d.type === "numeric" ? `<li><b>${esc(g.labels[f])}</b>: ${esc(d.summary)}</li>` :
      `<li><b>${esc(g.labels[f])}</b>: ` + Object.entries(d.mean_shap_by_level).map(([l, v]) => `${esc(l)} ${v > 0 ? "+" : ""}${v.toFixed(2)}`).join(", ") + "</li>").join("") + `</ul><p class="small">${esc(g.note)}</p>`;
    loaded.xai = true;
  } catch (e) { toast(e.message, "e"); }
}
$("#xaiBtn").onclick = async () => {
  const id = $("#xaiSel").value; if (!id) return toast("Make a prediction first", "e");
  const b = $("#xaiBtn"); b.disabled = true; b.innerHTML = '<span class="spin"></span> Computing…';
  try {
    const r = await post("/explain", { prediction_id: +id, include_lime: true });
    $("#xaiOut").innerHTML = `<p><b>Prediction #${r.id}:</b> ${pct(r.prediction.probability)} (${r.prediction.category})</p><div class="grid2">
      <div><h2>Local SHAP <span class="tag alt">${esc(r.shap.unit)}</span></h2><canvas id="locShap" height="240"></canvas></div>
      <div><h2>LIME <span class="tag alt">surrogate R² ${r.lime.r2_of_surrogate.toFixed(2)}</span></h2><canvas id="locLime" height="240"></canvas>
      <p class="small">LIME local prediction ${pct(r.lime.local_prediction)}; weights are on the probability scale.</p></div></div>
      <h2>Customer's most influential features</h2><table><tr><th>Feature<th>Value<th>SHAP<th>Direction</tr>${r.shap.contributions.slice(0, 5).map(c => `<tr><td>${esc(c.label)}<td>${esc(c.value)}<td>${c.contribution.toFixed(3)}<td>${c.direction} risk</tr>`).join("")}</table>
      <p class="small"><b>Global vs local:</b> global explanations summarise the model over many customers; local explanations explain this one prediction. ${esc(r.method_note)}</p>`;
    contribChart("locShap", r.shap.contributions, r.shap.unit);
    barH("locLime", r.lime.weights.map(w => w.rule), r.lime.weights.map(w => w.weight), r.lime.weights.map(w => w.weight > 0 ? RED : GREEN));
  } catch (e) { toast(e.message, "e"); } finally { b.disabled = false; b.textContent = "Explain (SHAP + LIME)"; }
};

/* ---------- notice ---------- */
$("#ntBtn").onclick = async () => {
  const id = $("#ntSel").value; if (!id) return toast("Make a prediction first", "e");
  try {
    const n = await api("/notice/" + id);
    const li = xs => xs.length ? xs.map(x => `<li><b>${esc(x.feature)}</b> (${esc(x.value)}): ${esc(x.explanation)}</li>`).join("") : "<li>None identified.</li>";
    $("#ntOut").innerHTML = `<h2>Decision explanation — prediction #${n.id}</h2><p class="small">Generated ${esc(n.generated_at)}</p>
      <p><span class="pill ${n.category}">${n.category}</span> &nbsp;<b>${pct(n.probability)}</b> estimated claim risk</p><p>${esc(n.summary)}</p>
      <p><b>Recommendation:</b> ${esc(n.recommendation)}</p><h2>Factors that increased the estimated risk</h2><ul>${li(n.adverse_factors)}</ul>
      <h2>Factors that lowered the estimated risk</h2><ul>${li(n.favourable_factors)}</ul>
      <p class="small">Not used as customer-facing reasons: ${esc(n.excluded_from_reasons.join(", "))}.</p><h2>Disclaimer</h2><p class="small">${esc(n.disclaimer)}</p>
      <div class="row"><button class="btn" onclick="window.print()">Print</button><a class="btn alt" style="text-decoration:none" href="/api/notice/${n.id}/download">Download HTML</a></div>`;
  } catch (e) { toast(e.message, "e"); }
};

/* ---------- fairness ---------- */
async function loadFair() {
  if (loaded.fair) return;
  try {
    const f = await api("/fairness-report"), o = f.overall; let h = `<div class="panel"><b>Overall (test set):</b> n=${o.n}, predicted-positive ${pct(o.predicted_positive_rate)}, TPR ${pct(o.tpr)}, FPR ${pct(o.fpr)}, precision ${pct(o.precision)}.
      <p class="small">${esc(f.disclaimer)} Disparities below are prompts for human investigation, not legal findings.</p><a href="${f.downloads.html}" target="_blank">HTML report</a> · <a href="${f.downloads.csv}">CSV</a></div><div class="grid2">`;
    Object.keys(f.attributes).forEach(a => h += `<div class="panel"><h2>${esc(a)}</h2><canvas id="fc_${a}" height="200"></canvas></div>`);
    h += `</div><div class="panel"><h2>Group metrics</h2><table><tr><th>Attribute<th>Group<th>n<th>Actual rate<th>Pred. positive<th>TPR (95% CI)<th>FPR (95% CI)<th>Precision<th></tr>`;
    for (const [a, gs] of Object.entries(f.attributes)) for (const [g, m] of Object.entries(gs)) {
      const ci = c => c && c[0] != null ? ` (${pct(c[0])}–${pct(c[1])})` : "";
      h += `<tr><td>${esc(a)}<td>${esc(g)}<td>${m.n}<td>${pct(m.actual_claim_rate)}<td>${pct(m.predicted_positive_rate)}<td>${pct(m.tpr)}${ci(m.ci_tpr)}<td>${pct(m.fpr)}${ci(m.ci_fpr)}<td>${pct(m.precision)}<td>${m.small_sample ? "small sample" : ""}</tr>`; }
    h += `</table></div><div class="panel"><h2>Gaps (max − min across groups)</h2><table><tr><th>Attribute<th>Pred. positive gap<th>TPR gap<th>FPR gap<th>Selection-rate ratio</tr>${Object.entries(f.gaps).map(([a, g]) => `<tr><td>${esc(a)}<td>${pct(g.predicted_positive_rate_gap)}<td>${pct(g.tpr_gap)}<td>${pct(g.fpr_gap)}<td>${g.selection_rate_ratio_min_max.toFixed(2)}</tr>`).join("")}</table></div>
      <div class="panel"><h2>Flags for investigation (${f.flags.length})</h2><ul>${f.flags.map(x => `<li>${esc(x.attribute)} = <b>${esc(x.group)}</b>: ${esc(x.metric)} ${pct(x.value)} vs overall ${pct(x.overall)} (CI ${pct(x.ci[0])}–${pct(x.ci[1])})${x.small_sample ? " — small sample" : ""}</li>`).join("") || "<li>None</li>"}</ul>
      <h2>Proxy &amp; data-quality checks</h2><p class="small">${esc(f.proxy_checks.note)}<br>Mean mileage by gender: ${Object.entries(f.proxy_checks.mean_annual_mileage_by_gender).map(([k, v]) => `${esc(k)} ${Math.round(v)}`).join(", ")}; sports-car share: ${Object.entries(f.proxy_checks.sports_car_share_by_gender).map(([k, v]) => `${esc(k)} ${pct(v)}`).join(", ")}.</p>
      <h2>Limitations</h2><ul>${f.limitations.map(x => `<li>${esc(x)}</li>`).join("")}</ul><h2>Recommended mitigation</h2><ul>${f.mitigation.map(x => `<li>${esc(x)}</li>`).join("")}</ul></div>`;
    $("#fairOut").innerHTML = h;
    for (const [a, gs] of Object.entries(f.attributes)) { const k = Object.keys(gs);
      chart("fc_" + a, { type: "bar", data: { labels: k, datasets: [{ label: "Pred. positive rate", data: k.map(x => gs[x].predicted_positive_rate), backgroundColor: "#9ec5e8" }, { label: "TPR", data: k.map(x => gs[x].tpr), backgroundColor: BLUE }, { label: "FPR", data: k.map(x => gs[x].fpr), backgroundColor: NAVY }] }, options: { scales: { y: { max: 1 } } } }); }
    loaded.fair = true;
  } catch (e) { $("#fairOut").innerHTML = `<div class="panel err">${esc(e.message)}</div>`; }
}

/* ---------- evaluation ---------- */
async function loadEval() {
  if (loaded.eval) return;
  try {
    const e = await api("/evaluation"), t = e.test_metrics, d = e.dataset, cm = t.confusion_matrix;
    $("#evalOut").innerHTML = `<div class="banner">${esc(e.data_note)}</div><div class="cards">${[["Accuracy", pct(t.accuracy)], ["Precision", f3(t.precision)], ["Recall", f3(t.recall)], ["F1", f3(t.f1)], ["ROC-AUC", f3(t.roc_auc)], ["PR-AUC", f3(t.pr_auc)], ["Brier", f3(t.brier)]].map(([l, v]) => `<div class="card"><b>${v}</b><span>${l}</span></div>`).join("")}</div>
    <p class="small">Selected: <b>${esc(e.selected_model)}</b>. Decision threshold ${f3(e.decision_threshold)} (maximises F1 on out-of-fold training predictions). Only ${pct(d.claim_rate)} of customers file claims, so a model that always predicts "no claim" would already score ${pct(e.majority_class_accuracy)} accuracy — accuracy alone is misleading; look at recall, precision and PR-AUC.</p>
    <div class="grid2"><div class="panel"><h2>Confusion matrix (test)</h2><table><tr><th><th>Pred. no claim<th>Pred. claim</tr><tr><th>Actual no claim<td>${cm[0][0]}<td>${cm[0][1]}</tr><tr><th>Actual claim<td>${cm[1][0]}<td>${cm[1][1]}</tr></table>
      <h2 style="margin-top:1rem">Split &amp; method</h2><p class="small">${esc(d.split)}. Train ${d.train_size}, test ${d.test_size}. Preprocessing fitted only inside training folds (no leakage).</p></div>
      <div class="panel"><h2>Calibration (test)</h2><canvas id="calC" height="200"></canvas></div></div>
    <div class="panel"><h2>Model comparison (held-out test set)</h2><table><tr><th>Model<th>Accuracy<th>Precision<th>Recall<th>F1<th>ROC-AUC<th>PR-AUC<th>Brier<th>CV ROC-AUC</tr>${e.model_comparison.map(m => `<tr><td>${esc(m.model)}<td>${pct(m.accuracy)}<td>${f3(m.precision)}<td>${f3(m.recall)}<td>${f3(m.f1)}<td>${f3(m.roc_auc)}<td>${f3(m.pr_auc)}<td>${f3(m.brier)}<td>${f3(e.cv_results.find(c => c.model === m.model)?.cv_roc_auc)}</tr>`).join("")}</table>
      <p class="small">Model selection used cross-validated ROC-AUC on the training data only; an interpretable baseline is kept unless a complex model gains ≥ 0.005 AUC.</p><img class="full" src="/reports/evaluation.png" alt="Evaluation plots"></div>`;
    chart("calC", { type: "line", data: { labels: e.calibration.predicted.map(x => x.toFixed(2)), datasets: [{ label: "Observed claim rate", data: e.calibration.observed, borderColor: NAVY }, { label: "Perfect", data: e.calibration.predicted, borderColor: "#999", borderDash: [4, 4], pointRadius: 0 }] }, options: { scales: { x: { title: { display: true, text: "Mean predicted probability" } } } } });
    loaded.eval = true;
  } catch (er) { $("#evalOut").innerHTML = `<div class="panel err">${esc(er.message)}</div>`; }
}

buildForm().then(route).catch(e => { toast(e.message, "e"); route(); });

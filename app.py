"""InsureXplain Flask backend. Loads the trained artefacts once; never trains at request time.
Run:  python app.py   ->  http://127.0.0.1:5000"""
import json, itertools, os
from datetime import datetime, timezone
from collections import OrderedDict
from flask import Flask, jsonify, request, send_from_directory, Response
import config as C
from explainability.engine import InsureEngine
from explainability.notice import build_notice, notice_html

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
ENGINE = None
HISTORY, _ids = OrderedDict(), itertools.count(1)          # in-memory demo history (last 50 predictions)

def engine():
    global ENGINE
    if ENGINE is None: ENGINE = InsureEngine()
    return ENGINE

def now(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
def ok(data, code=200): return jsonify(success=True, data=data, error=None, timestamp=now()), code
def fail(msg, code=400, details=None): return jsonify(success=False, data=None, error=dict(message=msg, details=details), timestamp=now()), code

def validate(payload):
    """Whitelist validation: only known fields, correct types, allowed ranges/categories. Nothing is executed or loaded."""
    if not isinstance(payload, dict): return None, ["Body must be a JSON object."]
    errs, rec = [], {}
    for f in C.NUMERIC:
        if f not in payload or payload[f] in ("", None): errs.append(f"{C.LABELS[f]} is required."); continue
        try: v = float(payload[f])
        except (TypeError, ValueError): errs.append(f"{C.LABELS[f]} must be a number."); continue
        lo, hi = C.RANGES[f]
        if not lo <= v <= hi: errs.append(f"{C.LABELS[f]} must be between {lo} and {hi}."); continue
        rec[f] = int(v) if f != "annual_mileage" and f != "vehicle_value" and v == int(v) else v
    for f in C.CATEGORICAL:
        if payload.get(f) not in C.CATEGORY_VALUES[f]: errs.append(f"{C.LABELS[f]} must be one of {C.CATEGORY_VALUES[f]}.")
        else: rec[f] = payload[f]
    if not errs and C.DATA_SOURCE == "synthetic":
        if rec["driving_experience"] > rec["customer_age"] - 16: errs.append("Driving experience cannot exceed (age - 16).")
        if rec["years_insured"] > rec["driving_experience"]: errs.append("Years insured cannot exceed driving experience.")
        if rec["previous_claims"] > rec["previous_accidents"] + 5: errs.append("Previous claims look inconsistent with previous accidents.")
    return (None, errs) if errs else (rec, [])

def store(rec):
    e = engine(); pred = e.predict(rec); sh = e.local_shap(rec); pid = next(_ids)
    item = dict(id=pid, input=rec, prediction=pred, shap=sh, timestamp=now(), notice=build_notice(rec, pred, sh, now()))
    HISTORY[pid] = item
    while len(HISTORY) > 50: HISTORY.popitem(last=False)
    return item

def get_item(pid):
    it = HISTORY.get(int(pid)) if str(pid).isdigit() else None
    return it

@app.get("/")
def index(): return send_from_directory("static", "index.html")

@app.get("/reports/<path:name>")
def reports(name):  # send_from_directory blocks path traversal
    return send_from_directory(C.REPORT_DIR, name)

@app.get("/api/health")
def health():
    ready = (C.MODEL_DIR / "trained_model.joblib").exists()
    return ok(dict(backend="up", model_available=ready, explainers_loaded=ENGINE is not None))

@app.get("/api/model-info")
def model_info():
    m = engine().metrics
    return ok(dict(selected_model=m["selected_model"], data_note=m["data_note"], dataset=m["dataset"], test_metrics=m["test_metrics"],
        decision_threshold=m["decision_threshold"], risk_thresholds=C.RISK_THRESHOLDS, features=C.FEATURES, labels=C.LABELS,
        categories=C.CATEGORY_VALUES, ranges=C.RANGES, explainability_unit=engine().space))

@app.get("/api/evaluation")
def evaluation():
    m = engine().metrics
    return ok({k: m[k] for k in ["selected_model", "test_metrics", "model_comparison", "cv_results", "classification_report", "calibration",
                                 "majority_class_accuracy", "dataset", "decision_threshold", "data_note"]})

@app.post("/api/predict")
def predict():
    rec, errs = validate(request.get_json(silent=True))
    if errs: return fail("Validation failed.", 422, errs)
    it = store(rec)
    sh = it["shap"]["contributions"]
    return ok(dict(id=it["id"], timestamp=it["timestamp"], **it["prediction"], risk_thresholds=C.RISK_THRESHOLDS,
        top_factors=sh[:3], contributions=sh, base_probability=it["shap"]["base_probability"], unit=it["shap"]["unit"],
        explanation=it["notice"]["summary"], disclaimer="Decision-support output from a model trained on synthetic data; not an automatic decision."))

@app.post("/api/explain")
def explain():
    body = request.get_json(silent=True) or {}
    it = get_item(body.get("prediction_id")) if body.get("prediction_id") else None
    if it is None:
        rec, errs = validate(body.get("input", body))
        if errs: return fail("Provide a valid prediction_id or input.", 422, errs)
        it = store(rec)
    out = dict(id=it["id"], input=it["input"], prediction=it["prediction"], shap=it["shap"])
    if body.get("include_lime", True): out["lime"] = engine().local_lime(it["input"])
    out["method_note"] = ("SHAP and LIME can disagree: SHAP gives additive, model-faithful attributions relative to a background "
                          "average; LIME fits a sparse local surrogate on random perturbations and is sensitive to sampling and discretisation. "
                          "Neither shows causation.")
    return ok(out)

@app.get("/api/predictions")
def predictions():
    return ok([dict(id=i["id"], timestamp=i["timestamp"], probability=i["prediction"]["probability"], category=i["prediction"]["category"]) for i in reversed(HISTORY.values())])

@app.get("/api/global-explanation")
def global_exp():
    g = engine().global_explanation()
    if not g: return fail("Run explainability/build_global.py first.", 404)
    g["images"] = dict(importance="/reports/global_importance.png", summary="/reports/shap_summary.png")
    g["lime_note"] = "LIME is a local method; the global view uses SHAP aggregated over a training sample."
    return ok(g)

@app.get("/api/fairness-report")
def fairness():
    p = C.REPORT_DIR / "fairness_report.json"
    if not p.exists(): return fail("Run training/fairness_analysis.py first.", 404)
    d = json.load(open(p)); d["downloads"] = dict(html="/reports/fairness_report.html", csv="/reports/fairness_groups.csv")
    d["disclaimer"] = "Statistical fairness metrics alone cannot establish legal compliance or non-compliance."
    return ok(d)

@app.get("/api/notice/<pid>")
def notice(pid):
    it = get_item(pid)
    return ok(dict(id=it["id"], input=it["input"], **it["notice"])) if it else fail("Unknown prediction id.", 404)

@app.get("/api/notice/<pid>/download")
def notice_dl(pid):
    it = get_item(pid)
    if not it: return fail("Unknown prediction id.", 404)
    return Response(notice_html(it["notice"], it["input"]), mimetype="text/html",
                    headers={"Content-Disposition": f"attachment; filename=insurexplain_report_{pid}.html"})

@app.errorhandler(404)
def nf(e): return fail("Not found.", 404)
@app.errorhandler(500)
def ise(e): return fail("Internal server error.", 500)

if __name__ == "__main__":
    engine(); app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)

"""Group fairness audit on the held-out test predictions.  Run after train_model.py.
Audit attributes are used ONLY here; 'gender' is never a model input."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import config as C

def wilson(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float((c - w) / d), float((c + w) / d))

def group_metrics(g):
    y, yh = g.y_true.values, g.y_pred.values
    tp, fp, fn, tn = ((yh == 1) & (y == 1)).sum(), ((yh == 1) & (y == 0)).sum(), ((yh == 0) & (y == 1)).sum(), ((yh == 0) & (y == 0)).sum()
    n = len(g); pos, neg = tp + fn, fp + tn
    out = dict(n=int(n), actual_claim_rate=float(y.mean()), mean_predicted_prob=float(g.y_prob.mean()), predicted_positive_rate=float(yh.mean()),
        tpr=float(tp / pos) if pos else None, fpr=float(fp / neg) if neg else None, precision=float(tp / (tp + fp)) if tp + fp else None,
        recall=float(tp / pos) if pos else None, actual_positives=int(pos), actual_negatives=int(neg))
    out["ci_predicted_positive_rate"] = wilson(int(yh.sum()), n); out["ci_tpr"] = wilson(int(tp), int(pos)); out["ci_fpr"] = wilson(int(fp), int(neg))
    out["small_sample"] = bool(n < 100 or pos < 20)
    return out

def main():
    t = pd.read_csv(C.MODEL_DIR / "test_predictions.csv")
    if C.DATA_SOURCE == "synthetic":
        t["age_band"] = pd.cut(t.customer_age, [17, 25, 40, 60, 120], labels=["18-25", "26-40", "41-60", "61+"])
        t["region_risk"] = t.region_risk.fillna("Missing")
    overall = group_metrics(t); report = dict(overall=overall, attributes={}, flags=[])
    rows = []
    attrs = ["age_band", "gender", "region_risk"] if C.DATA_SOURCE == "synthetic" else []
    for attr in attrs:
        res = {str(k): group_metrics(g) for k, g in t.groupby(attr, observed=True)}
        report["attributes"][attr] = res
        for k, m in res.items():
            rows.append(dict(attribute=attr, group=k, **{a: m[a] for a in ["n", "actual_claim_rate", "mean_predicted_prob", "predicted_positive_rate", "tpr", "fpr", "precision", "recall"]}, small_sample=m["small_sample"]))
            for metric, ci in [("predicted_positive_rate", "ci_predicted_positive_rate"), ("tpr", "ci_tpr"), ("fpr", "ci_fpr")]:
                v, (lo, hi) = m[metric], m[ci]
                if v is None or lo is None: continue
                ref = overall[metric]
                if (ref < lo or ref > hi) and abs(v - ref) >= .05:
                    report["flags"].append(dict(attribute=attr, group=k, metric=metric, value=v, overall=ref, ci=[lo, hi], small_sample=m["small_sample"],
                        note="Group value differs from overall by >=5 points and the 95% CI excludes the overall value. Investigate; this is not a legal finding."))
        vals = [m for m in res.values() if m["tpr"] is not None and m["fpr"] is not None]
        report.setdefault("gaps", {})[attr] = dict(
            predicted_positive_rate_gap=float(max(m["predicted_positive_rate"] for m in res.values()) - min(m["predicted_positive_rate"] for m in res.values())),
            tpr_gap=float(max(m["tpr"] for m in vals) - min(m["tpr"] for m in vals)), fpr_gap=float(max(m["fpr"] for m in vals) - min(m["fpr"] for m in vals)),
            selection_rate_ratio_min_max=float(min(m["predicted_positive_rate"] for m in res.values()) / max(m["predicted_positive_rate"] for m in res.values())))
    # proxy / data-quality checks on the cleaned dataset
    d = pd.read_csv(C.DATA_FILE)
    report["proxy_checks"] = {} if C.DATA_SOURCE != "synthetic" else dict(
        mean_annual_mileage_by_gender={k: float(v) for k, v in d.groupby("gender").annual_mileage.mean().items()},
        sports_car_share_by_gender={k: float(v) for k, v in d.groupby("gender").vehicle_type.apply(lambda s: (s == "Sports").mean()).items()},
        claim_rate_by_gender={k: float(v) for k, v in d.groupby("gender")[C.TARGET].mean().items()},
        missing_rate_by_gender={k: {c: float(v) for c, v in x.items()} for k, x in d.groupby("gender")[["annual_mileage", "vehicle_value", "region_risk", "vehicle_type"]].apply(lambda g: g.isna().mean()).iterrows()},
        note="Proxy checks are available for the synthetic dataset; the Analytics Vidhya dataset has no audit attribute configured.")
    report["limitations"] = [
        "Synthetic data: any disparity reflects the generator's assumptions, not real customers.",
        "Some groups are small; confidence intervals are wide and flags on small samples are unreliable.",
        "Predicted-positive-rate, TPR and FPR gaps cannot all be equalised at once when base rates differ (impossibility results); choosing a fairness criterion is a policy decision.",
        "Age is a model input and a rating factor in many markets; whether that is permitted depends on jurisdiction.",
        "Only single attributes are audited; intersectional groups (e.g. age x gender) are not.",
        "Statistical fairness metrics alone cannot establish legal compliance or non-compliance."]
    report["mitigation"] = [
        "Review flagged groups with domain experts and compliance counsel before any action.",
        "Examine proxy features (annual mileage, vehicle type, region) and test removal or monotonic constraints.",
        "Consider group-aware threshold review or re-weighting only where legally permitted.",
        "Collect more data for small groups and re-run with intersectional slices.",
        "Monitor these metrics after deployment and keep a human in the loop for adverse outcomes."]
    json.dump(report, open(C.REPORT_DIR / "fairness_report.json", "w"), indent=2)
    pd.DataFrame(rows).to_csv(C.REPORT_DIR / "fairness_groups.csv", index=False)
    f = lambda x: "-" if x is None else f"{x:.3f}"
    tr = "".join(f"<tr><td>{r['attribute']}</td><td>{r['group']}</td><td>{r['n']}</td><td>{f(r['actual_claim_rate'])}</td><td>{f(r['predicted_positive_rate'])}</td><td>{f(r['tpr'])}</td><td>{f(r['fpr'])}</td><td>{f(r['precision'])}</td><td>{'small' if r['small_sample'] else ''}</td></tr>" for r in rows)
    fl = "".join(f"<li>{x['attribute']}={x['group']}: {x['metric']} {x['value']:.3f} vs overall {x['overall']:.3f}</li>" for x in report["flags"]) or "<li>No flags.</li>"
    html = f"""<!doctype html><meta charset=utf-8><title>InsureXplain Fairness Report</title><style>body{{font:14px system-ui;max-width:900px;margin:2rem auto;color:#12263f}}td,th{{border:1px solid #cfe0f3;padding:4px 8px}}table{{border-collapse:collapse}}</style>
<h1>Fairness &amp; Compliance Report (synthetic data)</h1><p>Overall: n={overall['n']}, predicted positive rate {f(overall['predicted_positive_rate'])}, TPR {f(overall['tpr'])}, FPR {f(overall['fpr'])}.</p>
<table><tr><th>Attribute<th>Group<th>n<th>Actual rate<th>Pred. pos. rate<th>TPR<th>FPR<th>Precision<th>Note</tr>{tr}</table>
<h2>Flags for human review</h2><ul>{fl}</ul><h2>Limitations</h2><ul>{''.join(f'<li>{x}</li>' for x in report['limitations'])}</ul>
<h2>Mitigation</h2><ul>{''.join(f'<li>{x}</li>' for x in report['mitigation'])}</ul><p><b>Statistical fairness metrics alone cannot establish legal compliance.</b></p>"""
    open(C.REPORT_DIR / "fairness_report.html", "w").write(html)
    print(pd.DataFrame(rows).round(3).to_string()); print("flags:", len(report["flags"]))
if __name__ == "__main__": main()

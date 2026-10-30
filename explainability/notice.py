"""Plain-language, ILLUSTRATIVE adverse-underwriting explanation built from genuine SHAP contributions."""
import html, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

MIN_EFFECT = 0.02   # contributions smaller than this are treated as negligible and never listed as reasons

def _fmt(f, v):
    if f == "vehicle_value": return f"{v:,.0f}"
    if f == "annual_mileage": return f"{v:,.0f} per year"
    if f in ("customer_age", "vehicle_age", "driving_experience", "years_insured"): return f"{v:g} years"
    return str(v)

PHRASE = {
    "previous_claims": "the recorded number of previous claims ({v})",
    "previous_accidents": "the recorded number of previous accidents ({v})",
    "annual_mileage": "the estimated annual mileage ({v})",
    "vehicle_age": "the age of the vehicle ({v})",
    "vehicle_type": "the vehicle type ({v})",
    "driving_experience": "the amount of driving experience ({v})",
    "years_insured": "the number of years insured ({v})",
    "vehicle_value": "the recorded vehicle value ({v})",
    "policy_type": "the selected policy type ({v})",
    "customer_age": "the customer's age band ({v})",
    "region_risk": "the geographic risk category ({v})",
}

def build_notice(rec, pred, shap_local, now):
    eligible = [c for c in shap_local["contributions"] if c["feature"] not in C.NOTICE_EXCLUDED and abs(c["contribution"]) >= MIN_EFFECT]
    adverse = sorted([c for c in eligible if c["contribution"] > 0], key=lambda c: -c["contribution"])[:3]
    favour = sorted([c for c in eligible if c["contribution"] < 0], key=lambda c: c["contribution"])[:3]
    def phrase(c):
        return PHRASE.get(c["feature"], f"the {c['label'].lower()} ({_fmt(c['feature'], c['value'])})")
    def line(c, up):
        what = phrase(c).format(v=_fmt(c["feature"], c["value"]))
        return (f"{what[0].upper()+what[1:]} was associated with a {'higher' if up else 'lower'} estimated claim-risk score in this model.")
    cat = pred["category"]
    head = {"High": "Your application has been referred for further human review.",
            "Moderate": "Your application may benefit from additional review by an underwriter.",
            "Low": "The model's assessment places this profile in the lower claim-risk range."}[cat]
    if adverse:
        names = [phrase(c).split(" (")[0] for c in adverse]
        body = ("The model's assessment was most influenced by " + ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1] + ".")
    else:
        body = "No eligible factor increased the estimated risk by a meaningful amount, so no adverse factors are listed."
    note = ("" if len(adverse) == 3 else f" Only {len(adverse)} adverse factor(s) met the reliability threshold; none were added to fill the list.")
    summary = (f"{head} {body}{note} These factors contributed to an estimated claim-risk score of {pred['probability']:.1%} "
               f"(category: {cat}). This is an automated model assessment and does not, by itself, establish that you are "
               "ineligible for insurance or that any statement you made is inaccurate.")
    return dict(generated_at=now, summary=summary, category=cat, probability=pred["probability"], recommendation=pred["recommendation"],
        adverse_factors=[dict(feature=c["label"], value=_fmt(c["feature"], c["value"]), contribution=c["contribution"], explanation=line(c, True)) for c in adverse],
        favourable_factors=[dict(feature=c["label"], value=_fmt(c["feature"], c["value"]), contribution=c["contribution"], explanation=line(c, False)) for c in favour],
        excluded_from_reasons=[C.LABELS[f] for f in C.NOTICE_EXCLUDED],
        disclaimer=(f"ILLUSTRATIVE ACADEMIC OUTPUT built on {C.DATA_NOTE.lower()} This is an explanation of a model result, not a legally approved adverse "
                    "action notice. Legal requirements depend on jurisdiction, decision context and applicable insurance regulation. Model "
                    "contributions describe statistical association, not causation, and do not imply fault or dishonesty. Final decisions "
                    "must be made by qualified humans."))

def notice_html(n, rec):
    li = lambda xs: "".join(f"<li><b>{html.escape(x['feature'])}</b> ({html.escape(str(x['value']))}): {html.escape(x['explanation'])}</li>" for x in xs) or "<li>None identified.</li>"
    inputs = "".join(f"<tr><td>{html.escape(C.LABELS[k])}</td><td>{html.escape(str(rec[k]))}</td></tr>" for k in C.FEATURES)
    return f"""<!doctype html><html lang=en><meta charset=utf-8><title>InsureXplain explanation report</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#12263f}}h1{{color:#0b2a55}}
td{{border:1px solid #cfe0f3;padding:4px 10px}}table{{border-collapse:collapse}}.box{{background:#eef5fc;padding:1rem;border-radius:8px}}small{{color:#555}}</style>
<h1>InsureXplain — Decision Explanation (illustrative)</h1><p><small>Generated {html.escape(n['generated_at'])}</small></p>
<div class=box><p>{html.escape(n['summary'])}</p><p><b>Recommendation:</b> {html.escape(n['recommendation'])}</p></div>
<h2>Factors that increased the estimated risk</h2><ul>{li(n['adverse_factors'])}</ul>
<h2>Factors that lowered the estimated risk</h2><ul>{li(n['favourable_factors'])}</ul>
<p><small>Not used as customer-facing reasons (sensitive or proxy-prone): {html.escape(', '.join(n['excluded_from_reasons']))}.</small></p>
<h2>Information used</h2><table>{inputs}</table><h2>Disclaimer</h2><p>{html.escape(n['disclaimer'])}</p></html>"""

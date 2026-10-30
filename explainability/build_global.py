"""Compute GLOBAL SHAP explanations on a training sample and save plots + JSON.  Run after train_model.py."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, shap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config as C
from explainability.engine import InsureEngine

eng = InsureEngine()
S = eng.train.sample(1500, random_state=C.SEED).reset_index(drop=True)
agg = eng._aggregate(eng._raw_shap(S))
imp = agg.abs().mean().sort_values(ascending=False)

# direction summary: numeric -> correlation of value with SHAP; categorical -> mean SHAP per level
direction = {}
for f in C.FEATURES:
    if f in C.NUMERIC:
        v = S[f].fillna(S[f].median()); r = float(np.corrcoef(v, agg[f])[0, 1]) if v.std() > 0 and agg[f].std() > 0 else 0.0
        direction[f] = dict(type="numeric", correlation=r, summary=("Higher values tend to INCREASE" if r > .1 else "Higher values tend to DECREASE" if r < -.1 else "No consistent direction for") + " predicted risk")
    else:
        direction[f] = dict(type="categorical", mean_shap_by_level={k: float(v) for k, v in agg[f].groupby(S[f].fillna("Missing")).mean().items()})
json.dump(dict(unit=eng.space, n_samples=len(S), importance={k: float(v) for k, v in imp.items()},
    labels=C.LABELS, direction=direction, base_value=eng.base,
    note="SHAP values describe how the MODEL uses each feature. They do not establish causation."),
    open(C.REPORT_DIR / "global_shap.json", "w"), indent=2)

# plots
plt.figure(figsize=(7, 4.5)); imp[::-1].rename(index=C.LABELS).plot.barh(color="#0b2a55")
plt.xlabel(f"mean |SHAP| ({eng.space})"); plt.title("Global feature importance"); plt.tight_layout(); plt.savefig(C.REPORT_DIR / "global_importance.png", dpi=120); plt.close()
codes = eng._to_codes(S)[C.FEATURES]
shap.summary_plot(agg.values, codes.rename(columns=C.LABELS), show=False, max_display=11)
plt.title("SHAP summary (categorical features shown as category codes)"); plt.tight_layout(); plt.savefig(C.REPORT_DIR / "shap_summary.png", dpi=120, bbox_inches="tight"); plt.close()
print("Global SHAP saved. Top features:\n", imp.head(6).round(3))

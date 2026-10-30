"""Explainability engine: genuine SHAP (global + local) and LIME (local) computed from the trained pipeline."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, joblib, shap
from lime.lime_tabular import LimeTabularExplainer
import config as C
from training.features import ENGINEERED_OWNERS

def _sigmoid(x): return 1 / (1 + np.exp(-x))

class InsureEngine:
    def __init__(self):
        self.model = joblib.load(C.MODEL_DIR / "trained_model.joblib")
        self.pre, self.clf = self.model.named_steps["pre"], self.model.named_steps["clf"]
        self.metrics = json.load(open(C.MODEL_DIR / "model_metrics.json"))
        self.train = pd.read_csv(C.MODEL_DIR / "train_features.csv")
        # The nested encoder was fitted after FeatureBuilder added its columns.
        # Ask it directly so sklearn does not re-check the original input names.
        encoder = self.pre.named_steps["encode"] if hasattr(self.pre, "named_steps") else self.pre
        self.fnames = list(encoder.get_feature_names_out())
        self.owner = [self._owner(n) for n in self.fnames]           # transformed column -> original feature
        bg = self.pre.transform(self.train.sample(300, random_state=C.SEED))
        if hasattr(self.clf, "coef_"):                               # linear model -> exact linear SHAP
            self.shap_exp = shap.LinearExplainer(self.clf, bg); self.space = "log-odds"
        else:                                                        # tree model -> TreeExplainer
            self.shap_exp = shap.TreeExplainer(self.clf)
            self.space = "log-odds" if "Gradient" in type(self.clf).__name__ else "probability"
        self.base = float(np.ravel(self.shap_exp.expected_value)[-1])
        self._init_lime()

    def _owner(self, name):
        n = name.split("__", 1)[1]
        if n in ENGINEERED_OWNERS:
            return ENGINEERED_OWNERS[n]
        for c in C.CATEGORICAL:
            if n.startswith(c + "_"): return c
        return n

    # ---------- prediction ----------
    @staticmethod
    def category(p):
        t = C.RISK_THRESHOLDS
        return "Low" if p < t["low_below"] else ("High" if p >= t["high_from"] else "Moderate")

    def frame(self, rec): return pd.DataFrame([{k: rec[k] for k in C.FEATURES}])

    def predict(self, rec):
        p = float(self.model.predict_proba(self.frame(rec))[0, 1]); cat = self.category(p)
        rec_txt = {"Low": "Standard processing; no referral indicated by the model alone.",
                   "Moderate": "Optional underwriter review recommended before a final decision.",
                   "High": "Refer to a human underwriter for review. This is not an automatic decline."}[cat]
        return dict(probability=p, category=cat, recommendation=rec_txt,
                    binary_flag=bool(p >= self.metrics["decision_threshold"]))

    # ---------- SHAP ----------
    def _raw_shap(self, X):
        sv = self.shap_exp.shap_values(self.pre.transform(X))
        if isinstance(sv, list): sv = sv[-1]
        sv = np.asarray(sv); return sv[..., -1] if sv.ndim == 3 else sv

    def _aggregate(self, sv):
        """Sum SHAP values of one-hot columns back onto the original feature (exact: SHAP is additive)."""
        out = pd.DataFrame(0.0, index=range(sv.shape[0]), columns=C.FEATURES)
        for j, o in enumerate(self.owner): out[o] += sv[:, j]
        return out

    def local_shap(self, rec):
        X = self.frame(rec); agg = self._aggregate(self._raw_shap(X)).iloc[0]
        rows = [dict(feature=f, label=C.LABELS[f], value=rec[f], contribution=float(agg[f]),
                     direction="increases" if agg[f] > 0 else "decreases") for f in C.FEATURES]
        rows.sort(key=lambda r: -abs(r["contribution"]))
        return dict(method="SHAP", unit=self.space, base_value=self.base,
                    base_probability=float(_sigmoid(self.base)) if self.space == "log-odds" else self.base, contributions=rows)

    # ---------- LIME ----------
    def _init_lime(self):
        t = self.train.copy()
        self.num_med = t[C.NUMERIC].median(); self.cat_names = [C.CATEGORY_VALUES[c] for c in C.CATEGORICAL]
        self.lime_cols = C.FEATURES
        enc = self._to_codes(t)
        self.lime = LimeTabularExplainer(enc.values, feature_names=self.lime_cols, class_names=["No claim", "Claim"],
            categorical_features=[self.lime_cols.index(c) for c in C.CATEGORICAL],
            categorical_names={self.lime_cols.index(c): C.CATEGORY_VALUES[c] for c in C.CATEGORICAL},
            mode="classification", discretize_continuous=True, random_state=C.SEED)

    def _to_codes(self, df):
        d = df[C.FEATURES].copy()
        d[C.NUMERIC] = d[C.NUMERIC].fillna(self.num_med)
        for c in C.CATEGORICAL:
            vals = C.CATEGORY_VALUES[c]; d[c] = d[c].fillna(d[c].mode()[0] if d[c].notna().any() else vals[0]).map(vals.index)
        return d.astype(float)

    def _lime_predict(self, arr):
        d = pd.DataFrame(arr, columns=self.lime_cols)
        for c in C.CATEGORICAL:
            vals = C.CATEGORY_VALUES[c]; d[c] = d[c].round().clip(0, len(vals) - 1).astype(int).map(lambda i: vals[i])
        return self.model.predict_proba(d)

    def local_lime(self, rec, num_features=6, num_samples=3000):
        row = self._to_codes(self.frame(rec)).values[0]
        e = self.lime.explain_instance(row, self._lime_predict, num_features=num_features, num_samples=num_samples)
        return dict(method="LIME", unit="local surrogate weight (probability scale)", local_prediction=float(e.local_pred[0]),
                    intercept=float(e.intercept[1]), r2_of_surrogate=float(e.score),
                    weights=[dict(rule=r, weight=float(w), direction="increases" if w > 0 else "decreases") for r, w in e.as_list()])

    # ---------- global ----------
    def global_explanation(self):
        p = C.REPORT_DIR / "global_shap.json"
        return json.load(open(p)) if p.exists() else None

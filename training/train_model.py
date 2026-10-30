"""Train, tune, evaluate and save the InsureXplain claim-risk model.  Run:  python training/train_model.py"""
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report, brier_score_loss, precision_recall_curve,
    roc_curve)
from sklearn.calibration import calibration_curve
import config as C
from training.features import FeatureBuilder, ENGINEERED_NUMERIC
warnings.filterwarnings("ignore")
C.MODEL_DIR.mkdir(exist_ok=True); C.REPORT_DIR.mkdir(exist_ok=True)

# ---------- 1. load + inspect ----------
raw = pd.read_csv(C.DATA_FILE)
print("Records, columns:", raw.shape); print(raw.head()); raw.info()
missing = raw.isna().sum(); print("\nMissing values:\n", missing[missing > 0])
dups = int(raw.duplicated().sum()); df = raw.drop_duplicates().reset_index(drop=True)
print(f"Duplicates removed: {dups}  -> {len(df)} records")
rate = df[C.TARGET].mean(); print(f"Claim rate {rate:.3%} (imbalanced -> accuracy alone is misleading)")
df.to_csv(C.DATA_DIR / "cleaned_dataset.csv", index=False)   # de-duplicated; imputation happens inside the pipeline

# ---------- 2. EDA plots ----------
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
df[C.TARGET].value_counts().sort_index().plot.bar(ax=ax[0], color=["#9ec5e8", "#0b2a55"]); ax[0].set_title("Target distribution")
hist_col = "annual_mileage" if "annual_mileage" in df else "population_density"
group_col = "previous_claims" if "previous_claims" in df else "age_of_car"
sns.histplot(df[hist_col].dropna(), ax=ax[1], color="#0b2a55"); ax[1].set_title(hist_col)
sns.barplot(x=group_col, y=C.TARGET, data=df, ax=ax[2], color="#2f6db5"); ax[2].set_title(f"Claim rate by {group_col}")
plt.tight_layout(); plt.savefig(C.REPORT_DIR / "eda.png", dpi=110); plt.close()

# ---------- 3. split (stratified) - audit attributes kept aside, never model inputs ----------
X, y, audit = df[C.FEATURES], df[C.TARGET], df[C.AUDIT_ONLY]
X_tr, X_te, y_tr, y_te, a_tr, a_te = train_test_split(X, y, audit, test_size=.2, stratify=y, random_state=C.SEED)

def make_pre(scale=True):  # preprocessing is fitted only on training folds (inside Pipeline) -> no leakage
    num = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    numeric_features = C.NUMERIC if C.DATA_SOURCE == "analytics_vidhya" else C.NUMERIC + ENGINEERED_NUMERIC
    encoder = ColumnTransformer([("num", Pipeline(num), numeric_features),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                               categories=[C.CATEGORY_VALUES[c] for c in C.CATEGORICAL]))]), C.CATEGORICAL)])
    return Pipeline(([("features", FeatureBuilder())] if C.DATA_SOURCE != "analytics_vidhya" else []) + [("encode", encoder)])

cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
if C.DATA_SOURCE == "analytics_vidhya":
    candidates = {
        "Logistic Regression (baseline)": (Pipeline([("pre", make_pre(True)), ("clf", LogisticRegression(max_iter=1000))]),
            {"clf__C": [.1, 1]}),
        "Random Forest": (Pipeline([("pre", make_pre(False)), ("clf", RandomForestClassifier(random_state=C.SEED, n_jobs=-1))]),
            {"clf__n_estimators": [150], "clf__max_depth": [8], "clf__min_samples_leaf": [10]}),
        "Gradient Boosting": (Pipeline([("pre", make_pre(False)), ("clf", GradientBoostingClassifier(random_state=C.SEED))]),
            {"clf__n_estimators": [100], "clf__max_depth": [2], "clf__learning_rate": [.05], "clf__subsample": [.8]}),
    }
else:
    candidates = {
     "Logistic Regression (baseline)": (Pipeline([("pre", make_pre(True)), ("clf", LogisticRegression(max_iter=2000))]),
            {"clf__C": [.1, 1, 10]}),
     "Random Forest": (Pipeline([("pre", make_pre(False)), ("clf", RandomForestClassifier(random_state=C.SEED, n_jobs=-1))]),
            {"clf__n_estimators": [200], "clf__max_depth": [5, 8], "clf__min_samples_leaf": [10, 30]}),
     "Gradient Boosting": (Pipeline([("pre", make_pre(False)), ("clf", GradientBoostingClassifier(random_state=C.SEED))]),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3], "clf__learning_rate": [.05, .1], "clf__subsample": [.8]}),
    }
searched, cv_table = {}, []
for name, (pipe, grid) in candidates.items():
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1).fit(X_tr, y_tr)
    searched[name] = gs; cv_table.append({"model": name, "cv_roc_auc": gs.best_score_, "best_params": gs.best_params_})
    print(f"{name}: CV ROC-AUC {gs.best_score_:.4f}  {gs.best_params_}")

# selection: best CV ROC-AUC among *tree* models (SHAP TreeExplainer) unless baseline is within 0.005
best_name = max(searched, key=lambda k: searched[k].best_score_)
base_auc = searched["Logistic Regression (baseline)"].best_score_
if best_name != "Logistic Regression (baseline)" and searched[best_name].best_score_ - base_auc < .005:
    print("Gain over baseline < 0.005 AUC -> keeping interpretable baseline"); best_name = "Logistic Regression (baseline)"
final = searched[best_name].best_estimator_; print("Selected:", best_name)

# decision threshold for the binary label: maximise F1 on out-of-fold TRAIN predictions (test set untouched)
oof = cross_val_predict(final, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
p, r, t = precision_recall_curve(y_tr, oof); f1s = 2 * p * r / (p + r + 1e-9)
thr = float(t[np.argmax(f1s[:-1])]); print(f"Decision threshold (OOF F1-max): {thr:.3f}")

# ---------- 4. final test evaluation ----------
def evaluate(model, name, threshold):
    pr = model.predict_proba(X_te)[:, 1]; pred = (pr >= threshold).astype(int)
    return dict(model=name, threshold=threshold, accuracy=accuracy_score(y_te, pred), precision=precision_score(y_te, pred, zero_division=0),
        recall=recall_score(y_te, pred), f1=f1_score(y_te, pred), roc_auc=roc_auc_score(y_te, pr),
        pr_auc=average_precision_score(y_te, pr), brier=brier_score_loss(y_te, pr), confusion_matrix=confusion_matrix(y_te, pred).tolist()), pr, pred
comparison, probs = [], {}
for name, gs in searched.items():
    m = gs.best_estimator_; o = cross_val_predict(m, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    pp, rr, tt = precision_recall_curve(y_tr, o); th = float(tt[np.argmax((2*pp*rr/(pp+rr+1e-9))[:-1])])
    res, pr_, _ = evaluate(m, name, th); comparison.append(res); probs[name] = pr_
res, test_prob, test_pred = evaluate(final, best_name, thr)
print(classification_report(y_te, test_pred, target_names=["No claim", "Claim"]))
print(pd.DataFrame(comparison)[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]].round(3))
majority_acc = 1 - y_te.mean()

# plots: confusion matrix, ROC, PR, calibration
fig, ax = plt.subplots(1, 4, figsize=(19, 4))
sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax[0], xticklabels=["No claim", "Claim"], yticklabels=["No claim", "Claim"]); ax[0].set_title("Confusion matrix (test)")
for n_, pr_ in probs.items():
    fpr, tpr, _ = roc_curve(y_te, pr_); ax[1].plot(fpr, tpr, label=n_.split(" (")[0]); pp_, rr_, _ = precision_recall_curve(y_te, pr_); ax[2].plot(rr_, pp_, label=n_.split(" (")[0])
ax[1].plot([0, 1], [0, 1], "k--"); ax[1].set_title("ROC"); ax[1].legend(); ax[2].set_title("Precision-recall"); ax[2].set_xlabel("Recall"); ax[2].legend()
fr, mp = calibration_curve(y_te, test_prob, n_bins=8, strategy="quantile"); ax[3].plot(mp, fr, "o-"); ax[3].plot([0, .5], [0, .5], "k--"); ax[3].set_title("Calibration"); ax[3].set_xlabel("Mean predicted"); ax[3].set_ylabel("Observed rate")
plt.tight_layout(); plt.savefig(C.REPORT_DIR / "evaluation.png", dpi=110); plt.close()
cal_fr, cal_mp = calibration_curve(y_te, test_prob, n_bins=8, strategy="quantile")

# ---------- 5. save artefacts ----------
joblib.dump(final, C.MODEL_DIR / "trained_model.joblib")                      # full pipeline (preprocess + model)
joblib.dump(final.named_steps["pre"], C.MODEL_DIR / "preprocessing_pipeline.joblib")
X_tr.to_csv(C.MODEL_DIR / "train_features.csv", index=False)                  # background data for SHAP/LIME
pd.DataFrame({"y_true": y_te.values, "y_prob": test_prob, "y_pred": test_pred}).join(a_te.reset_index(drop=True)).join(
    X_te.reset_index(drop=True)).to_csv(C.MODEL_DIR / "test_predictions.csv", index=False)
metrics = dict(
    selected_model=best_name, decision_threshold=thr, risk_thresholds=C.RISK_THRESHOLDS, seed=C.SEED,
    data_note=C.DATA_NOTE,
    dataset=dict(records_raw=len(raw), records_clean=len(df), duplicates_removed=dups, columns=raw.shape[1], claim_rate=float(rate),
        missing_values={k: int(v) for k, v in missing[missing > 0].items()}, train_size=len(X_tr), test_size=len(X_te),
        split="80/20 stratified; 5-fold stratified CV on train for tuning & threshold", features=C.FEATURES, audit_only=C.AUDIT_ONLY),
    test_metrics=res, majority_class_accuracy=float(majority_acc), model_comparison=comparison,
    cv_results=[{**r_, "best_params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in r_["best_params"].items()}} for r_ in cv_table],
    calibration=dict(observed=cal_fr.tolist(), predicted=cal_mp.tolist()),
    classification_report=classification_report(y_te, test_pred, target_names=["No claim", "Claim"], output_dict=True))
json.dump(metrics, open(C.MODEL_DIR / "model_metrics.json", "w"), indent=2, default=float)
print("Saved model + metrics to", C.MODEL_DIR)

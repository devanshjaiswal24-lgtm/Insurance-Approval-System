# InsureXplain — Explainable Insurance Risk Intelligence

Academic project: **motor-insurance claim-risk prediction** with genuine SHAP + LIME explanations, a plain-language
adverse-underwriting explanation generator, a fairness audit and a Flask + Chart.js dashboard.

> **Read this first.** The default data source is the **Analytics Vidhya vehicle-insurance claim dataset**. The project also retains the original synthetic generator as an offline fallback. Labels and dataset provenance should be considered when interpreting metrics.
> `training/generate_data.py`), so the model learns *those rules*, not real insurance behaviour. Metrics demonstrate the
> workflow only. The system predicts **claim risk** (likelihood of a future claim). It does not adjudicate claims, detect
> fraud, or make automatic decisions; a high score never means a customer should be denied cover. The explanation
> report is **illustrative, not a legally compliant adverse action notice**.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py          # generates data, trains, builds SHAP + fairness reports (about 1-2 min)
python app.py              # open http://127.0.0.1:5000
```
Training is separate from serving: `app.py` only loads `models/*.joblib`. Set `INSUREXPLAIN_DATA_SOURCE=synthetic` to use the original fallback generator.

## Run the deployed app locally
```powershell
python -m pip install -r requirements.txt
python app.py
```
The server listens on `http://127.0.0.1:5000` locally and uses Render's `PORT` environment variable when deployed.

## Deploy on Render
Push this repository to GitHub, then create a Render **Web Service** from the repository. Render can use the included `render.yaml`, or configure:
* Build command: `pip install -r requirements.txt`
* Start command: `gunicorn app:app`
* Environment variable: `INSUREXPLAIN_DATA_SOURCE=analytics_vidhya`

The trained model, reports, and Analytics Vidhya training data are included in the repository so deployment does not retrain the model. Do not commit credentials or private data.

## Structure
```
app.py                      Flask API + serves the dashboard
config.py                   features, ranges, risk thresholds, notice exclusions, seed
run_all.py                  runs the whole offline pipeline
data/                       analytics_vidhya/train.csv (default) · insurance_dataset.csv (synthetic fallback) · cleaned_dataset.csv
training/                   generate_data.py · train_model.py · evaluate_model.py · fairness_analysis.py
explainability/             engine.py (SHAP + LIME) · build_global.py · notice.py (plain-language generator)
models/                     trained_model.joblib · preprocessing_pipeline.joblib · model_metrics.json · test_predictions.csv
reports/                    eda.png · evaluation.png · global_shap.json · shap_summary.png · fairness_report.{json,html} · fairness_groups.csv
static/                     index.html · styles.css · app.js
notebooks/                  model_development.ipynb
```

## Method summary
* **Features (11):** customer_age, vehicle_age, vehicle_type, annual_mileage, driving_experience, previous_claims, previous_accidents, years_insured, vehicle_value, policy_type, region_risk. Target: `claim_next_12m`. Annual premium is deliberately **excluded** (it would be derived from risk and leak the answer). `gender` exists only for fairness auditing and is **never** a model input.
* **Preparation:** duplicates removed, ~3% missing values imputed *inside* the sklearn pipeline (median / most-frequent), one-hot encoding, scaling for the linear model, and leakage-safe derived features for young/senior age bands and combined experience/claim history. The split is stratified 80/20, with 5-fold CV on the training set only and fixed seed 42.
* **Models:** Logistic Regression baseline vs Random Forest vs Gradient Boosting, tuned with GridSearchCV (ROC-AUC). The interpretable baseline is kept unless a complex model gains ≥0.005 AUC. The run that produced the delivered files selected whichever model won on real CV scores; see `models/model_metrics.json`.
* **Imbalance:** ~9% of records have a claim, so accuracy alone misleads (always predicting "no claim" scores ≈91%). The binary flag threshold maximises F1 on out-of-fold training predictions; the test set is touched once.
* **Risk categories:** Low < 8%, Moderate 8–20%, High ≥ 20% (`RISK_THRESHOLDS` in `config.py`; chosen relative to the 9% base rate and the F1-optimal cut-off).
* **SHAP:** `LinearExplainer` (linear model) or `TreeExplainer` (tree models) on the real fitted model. One-hot columns are summed back onto the original feature (exact, because SHAP is additive). Global = mean |SHAP| over 1,500 training rows. Values are in log-odds.
* **LIME:** `LimeTabularExplainer` over the original 11 features (categoricals as codes) with a `predict_fn` that maps back to a DataFrame and calls the real pipeline. SHAP and LIME may disagree: SHAP is additive and model-faithful relative to a background average; LIME fits a sparse surrogate on random perturbations. They are shown side by side, never forced to agree.
* **Interactions:** Logistic Regression is additive, so there are no learned interactions to explain. If a tree model is selected, `shap_interaction_values` could be added.
* **Causation:** SHAP/LIME describe how the *model* uses features, not what causes claims.
* **Notice generator:** takes the top three *adverse* (risk-increasing) SHAP contributions among eligible features with |contribution| ≥ 0.02; if fewer qualify it lists fewer. `customer_age` and `region_risk` are excluded as customer-facing reasons by default (`NOTICE_EXCLUDED`) because they are sensitive/proxy-prone in many jurisdictions; they still appear in the technical explanation.
* **Fairness:** per-group n, actual rate, predicted-positive rate, TPR, FPR, precision, Wilson 95% CIs, gaps, flags (≥5-point difference *and* CI excluding the overall value), proxy checks, limitations and mitigation. Groups: age band, gender (audit-only), region. Large age-band gaps appear because age genuinely drives the synthetic labels, and age is a model input; whether that is acceptable is a legal/policy question.

## API (all JSON: `{success, data, error, timestamp}`)
| Endpoint | Purpose |
|---|---|
| `GET /api/health` | backend/model status |
| `GET /api/model-info` | model, dataset summary, test metrics, schema, ranges |
| `POST /api/predict` | validate input → probability, category, recommendation, SHAP factors, explanation |
| `POST /api/explain` | `{prediction_id}` or `{input}` → SHAP + LIME (`include_lime` optional) |
| `GET /api/global-explanation` | global SHAP importances, directions, image links |
| `GET /api/fairness-report` | fairness JSON + download links |
| `GET /api/evaluation` | metrics, comparison, calibration |
| `GET /api/predictions` | in-memory history (last 50) |
| `GET /api/notice/<id>` and `/download` | explanation report (JSON / HTML download) |

Example:
```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" -d '{
 "customer_age":24,"vehicle_age":8,"vehicle_type":"Sports","annual_mileage":16000,"driving_experience":5,
 "previous_claims":2,"previous_accidents":2,"years_insured":2,"vehicle_value":22000,
 "policy_type":"Comprehensive","region_risk":"Medium"}'
```
Response (abridged; numbers vary with retraining): `{"success":true,"data":{"id":1,"probability":0.4,"category":"High","recommendation":"Refer to a human underwriter...","top_factors":[{"label":"Previous claim count","contribution":0.9,"direction":"increases"},...],"explanation":"..."}}`

Invalid input returns HTTP 422 with `error.details` listing each problem. The API accepts only whitelisted fields, never loads user-supplied paths and never executes user input.

## Known limitations
Synthetic data; single train/test split; in-memory prediction history (resets on restart); no authentication (local demo); only single-attribute fairness slices; Chart.js loads from a CDN (needs internet); fairness and notice modules are educational, not legal advice. Life insurance can be added by supplying another dataset/config and reusing `engine.py`.

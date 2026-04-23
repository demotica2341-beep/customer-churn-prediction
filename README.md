# Customer Churn Prediction — Complete Project

A production-ready churn prediction system for SaaS/Telecom.
Predicts which customers will cancel within the next 30 days.

---

## Project Structure

```
churn_project/
├── data/
│   ├── customers.csv          ← Raw customer dataset (5,000 rows)
│   └── scored_customers.csv   ← Model output with churn scores
├── src/
│   ├── generate_data.py       ← STEP 1: Synthetic data generation
│   ├── features.py            ← STEP 2: Feature engineering pipeline
│   ├── train.py               ← STEP 3: XGBoost training + evaluation
│   └── inference.py           ← STEP 4: Batch scoring / inference
├── models/
│   ├── xgb_model.pkl          ← Trained XGBoost classifier
│   ├── preprocessor.pkl       ← Fitted sklearn ColumnTransformer
│   ├── feature_engineer.pkl   ← Fitted feature engineering transformer
│   └── metadata.pkl           ← Threshold, AUC scores, feature names
├── notebooks/
│   └── evaluation_plots.png   ← ROC, PR, confusion matrix, feature importance
├── dashboard/
│   └── index.html             ← Interactive churn dashboard (open in browser)
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
```

### 2. Run the full pipeline
```bash
# Step 1: Generate data (replace with your real CSV in production)
python src/generate_data.py

# Step 2 + 3: Train model (feature engineering is built-in)
python src/train.py

# Step 4: Score all customers
python src/inference.py

# View dashboard
open dashboard/index.html
```

---

## Data Schema

| Column | Type | Description |
|--------|------|-------------|
| customer_id | string | Unique identifier |
| tenure_days | int | Days since signup |
| plan | categorical | starter / growth / pro / enterprise |
| monthly_mrr | int | Monthly recurring revenue |
| sessions_30d | int | Login sessions last 30 days |
| sessions_60d | int | Login sessions previous 30-60 days |
| features_used | int | Number of product features used (0-10) |
| api_calls_30d | int | API calls last 30 days |
| days_since_login | int | Days since last login |
| support_tickets_90d | int | Support tickets last 90 days |
| payment_failures_6m | int | Payment failures last 6 months |
| had_downgrade | binary | Downgraded plan in last 90 days |
| had_upgrade | binary | Upgraded plan in last 90 days |
| nps_score | int | Net Promoter Score (0-10) |
| churned | binary | **TARGET**: churned within 30 days |

---

## Engineered Features

| Feature | Formula | Why it matters |
|---------|---------|----------------|
| `usage_trend` | sessions_30d − sessions_60d | Declining usage is the #1 churn signal |
| `engagement_score` | Weighted blend of sessions, features, API, recency | Single 0-1 health score |
| `is_dormant` | days_since_login > 14 | Binary flag for at-risk users |
| `is_declining` | usage_trend < −5 | Sharp drop in activity |
| `billing_issues` | payment_failures > 0 | Failed payments predict churn |
| `nps_detractor` | nps_score ≤ 3 | Dissatisfied customers |
| `tenure_bucket` | Binned tenure | Non-linear tenure effects |

---

## Model Results

| Metric | Value |
|--------|-------|
| ROC-AUC (test) | 0.6275 |
| PR-AUC (test) | 0.1607 |
| Optimal threshold | 0.46 |
| Class ratio (neg:pos) | 13.5:1 |

**Note**: PR-AUC matters more than ROC-AUC for imbalanced churn datasets.
A high ROC-AUC can mask poor precision on the minority (churner) class.

---

## Production Deployment

### Batch scoring (daily cron job)
```python
from src.inference import score_customers
import pandas as pd

# Pull fresh data from your warehouse
df = pd.read_sql("SELECT * FROM customers WHERE snapshot_date = CURRENT_DATE", conn)

# Score
scored = score_customers(df)

# Push to CRM / Salesforce / Hubspot
upload_to_crm(scored[scored.risk_tier.isin(['CRITICAL', 'HIGH'])])
```

### Model drift monitoring
- Monitor `mean(churn_probability)` weekly — should stay near training base rate
- Re-train monthly with a rolling 12-month window
- Alert if ROC-AUC drops below 0.60 on a holdout set

### Threshold tuning
- Lower threshold → more recalls, more false positives (more retention calls)
- Higher threshold → fewer alerts, fewer missed churners
- Rule of thumb: set threshold where `cost_of_FN / cost_of_FP` ratio is balanced
  - If a churner costs $500 to lose and a retention call costs $20, FN is 25× more costly
  - Lower the threshold to bias toward recall

---

## Improving the Model

1. **Add SHAP explanations** — per-customer "why is this person at risk"
2. **LTV weighting** — weight training samples by customer lifetime value
3. **Sequence features** — rolling 7/30/90-day windows, time-series features
4. **Interaction features** — (tenure × NPS), (plan × usage_trend)
5. **Ensemble** — blend XGBoost + LightGBM + Logistic Regression
6. **Calibration** — use Platt scaling so predicted probabilities are accurate
7. **Real-time scoring** — wrap inference.py in a FastAPI endpoint

---

## Files to Replace for Production

| File | Replace with |
|------|-------------|
| `src/generate_data.py` | Your real data extraction query |
| `data/customers.csv` | Live export from your warehouse |
| `models/*.pkl` | Retrained models on real data |

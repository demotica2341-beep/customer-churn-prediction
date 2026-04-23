"""
STEP 4 — INFERENCE / SCORING
Loads saved model and scores new customers.
In production, run this daily as a batch job (cron / Airflow / Lambda).
Outputs a scored CSV with churn_probability, risk_tier, and recommended_action.
"""

import pandas as pd
import numpy as np
import joblib, sys, os
sys.path.insert(0, '/home/claude/churn_project/src')
from features import prepare_features


# ── Load artefacts ────────────────────────────────────────────────────────────

def load_model(model_dir='/home/claude/churn_project/models'):
    xgb          = joblib.load(f'{model_dir}/xgb_model.pkl')
    preprocessor = joblib.load(f'{model_dir}/preprocessor.pkl')
    metadata     = joblib.load(f'{model_dir}/metadata.pkl')
    return xgb, preprocessor, metadata


# ── Score customers ───────────────────────────────────────────────────────────

def score_customers(df: pd.DataFrame, model_dir='/home/claude/churn_project/models') -> pd.DataFrame:
    """
    Input:  raw customer DataFrame (same schema as training data)
    Output: DataFrame with churn_probability, risk_tier, mrr_at_risk, recommended_action
    """
    xgb, preprocessor, metadata = load_model(model_dir)
    threshold = metadata['threshold']

    # Keep customer_id for output
    ids = df['customer_id'].copy() if 'customer_id' in df.columns else pd.RangeIndex(len(df))

    X, y, _ = prepare_features(df)
    X_pp    = preprocessor.transform(X)
    probs   = xgb.predict_proba(X_pp)[:, 1]

    # Risk tiers
    def tier(p):
        if p >= 0.70: return 'CRITICAL'
        if p >= 0.45: return 'HIGH'
        if p >= 0.25: return 'MEDIUM'
        return 'LOW'

    # Recommended action per tier
    action_map = {
        'CRITICAL': 'Immediate CSM outreach + retention offer',
        'HIGH':     'Schedule health-check call this week',
        'MEDIUM':   'Send engagement email + feature tips',
        'LOW':      'No action needed (monitor next cycle)',
    }

    results = pd.DataFrame({
        'customer_id':       ids.values,
        'churn_probability': np.round(probs, 4),
        'churn_predicted':   (probs >= threshold).astype(int),
        'risk_tier':         [tier(p) for p in probs],
        'monthly_mrr':       df['monthly_mrr'].values,
        'mrr_at_risk':       np.round(df['monthly_mrr'].values * probs, 2),
        'recommended_action': [action_map[tier(p)] for p in probs],
    }).sort_values('churn_probability', ascending=False)

    # Summary
    total_mrr_at_risk = results['mrr_at_risk'].sum()
    critical = (results['risk_tier'] == 'CRITICAL').sum()
    high     = (results['risk_tier'] == 'HIGH').sum()

    print(f"\nScoring summary  ({len(results):,} customers)")
    print(f"  CRITICAL: {critical:4d}  |  HIGH: {high:4d}")
    print(f"  Total MRR at risk: ${total_mrr_at_risk:,.0f}/mo")
    print(f"  Model threshold: {threshold:.2f}")
    return results


if __name__ == '__main__':
    # Score the full dataset as a demo
    df = pd.read_csv('/home/claude/churn_project/data/customers.csv')
    scored = score_customers(df)
    scored.to_csv('/home/claude/churn_project/data/scored_customers.csv', index=False)

    print("\nTop 10 at-risk customers:")
    print(scored[['customer_id','churn_probability','risk_tier',
                  'monthly_mrr','mrr_at_risk','recommended_action']].head(10).to_string(index=False))

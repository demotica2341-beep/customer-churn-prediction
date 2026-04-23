"""
STEP 1 — DATA GENERATION
Generates a realistic synthetic SaaS customer dataset with ~8% churn rate.
In production, replace this with your actual CRM + billing + usage exports.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
N = 5000  # customers

def generate_churn_dataset(n=N):
    today = datetime(2024, 4, 1)

    # ── Demographics & Plan ──────────────────────────────────────────────────
    tenure_days   = np.random.exponential(scale=400, size=n).clip(7, 1500).astype(int)
    plan          = np.random.choice(['starter', 'growth', 'pro', 'enterprise'],
                                     p=[0.35, 0.30, 0.25, 0.10], size=n)
    plan_price    = {'starter': 29, 'growth': 79, 'pro': 199, 'enterprise': 499}
    monthly_mrr   = np.array([plan_price[p] for p in plan])
    region        = np.random.choice(['NA', 'EU', 'APAC', 'LATAM'],
                                     p=[0.45, 0.30, 0.15, 0.10], size=n)
    company_size  = np.random.choice(['1-10', '11-50', '51-200', '200+'],
                                     p=[0.30, 0.35, 0.25, 0.10], size=n)

    # ── Usage Signals ────────────────────────────────────────────────────────
    # Sessions per month (last 30 days)
    sessions_30d  = np.random.poisson(lam=18, size=n).clip(0, 120)
    # Sessions previous 30-60 day window
    sessions_60d  = (sessions_30d * np.random.uniform(0.6, 1.8, size=n)).clip(0, 120).astype(int)
    # Feature adoption breadth (0-10 features used)
    features_used = np.random.binomial(10, 0.4, size=n)
    # API calls last 30 days
    api_calls_30d = np.random.exponential(scale=500, size=n).clip(0, 20000).astype(int)
    # Days since last login
    days_since_login = np.random.exponential(scale=5, size=n).clip(0, 90).astype(int)

    # ── Support & Billing Signals ────────────────────────────────────────────
    support_tickets_90d   = np.random.poisson(lam=1.2, size=n).clip(0, 20)
    payment_failures_6m   = np.random.binomial(6, 0.05, size=n)
    had_downgrade         = np.random.binomial(1, 0.08, size=n)
    had_upgrade           = np.random.binomial(1, 0.15, size=n)
    nps_score             = np.random.choice(range(0, 11),
                                              p=[0.03,0.02,0.03,0.05,0.07,0.10,
                                                 0.12,0.18,0.18,0.12,0.10], size=n)

    # ── Engineered Signals ───────────────────────────────────────────────────
    usage_trend = sessions_30d - sessions_60d           # negative = declining
    engagement_score = (
        0.3 * (sessions_30d / sessions_30d.max()) +
        0.3 * (features_used / 10) +
        0.2 * (api_calls_30d / api_calls_30d.max()) +
        0.2 * (1 - days_since_login / 90)
    ).clip(0, 1)

    # ── Churn Label Construction ─────────────────────────────────────────────
    # Build logistic churn probability from real signals
    log_odds = (
        -3.5                                        # base (keeps churn ~8%)
        + 0.8  * (days_since_login > 14).astype(int)
        + 1.2  * (usage_trend < -5).astype(int)
        + 0.6  * (sessions_30d < 3).astype(int)
        + 0.5  * (features_used < 2).astype(int)
        + 0.7  * (payment_failures_6m > 0).astype(int)
        + 0.9  * (had_downgrade == 1)
        - 0.4  * (had_upgrade == 1)
        + 0.4  * (support_tickets_90d > 3).astype(int)
        - 0.3  * (nps_score >= 8).astype(int)
        + 0.5  * (nps_score <= 3).astype(int)
        - 0.5  * (engagement_score > 0.6).astype(int)
        + np.random.normal(0, 0.3, size=n)
    )
    churn_prob = 1 / (1 + np.exp(-log_odds))
    churned    = (np.random.uniform(size=n) < churn_prob).astype(int)

    df = pd.DataFrame({
        'customer_id':          [f'CUST_{i:05d}' for i in range(n)],
        'tenure_days':          tenure_days,
        'plan':                 plan,
        'monthly_mrr':          monthly_mrr,
        'region':               region,
        'company_size':         company_size,
        'sessions_30d':         sessions_30d,
        'sessions_60d':         sessions_60d,
        'features_used':        features_used,
        'api_calls_30d':        api_calls_30d,
        'days_since_login':     days_since_login,
        'support_tickets_90d':  support_tickets_90d,
        'payment_failures_6m':  payment_failures_6m,
        'had_downgrade':        had_downgrade,
        'had_upgrade':          had_upgrade,
        'nps_score':            nps_score,
        'churned':              churned
    })

    print(f"Dataset: {len(df):,} customers | Churn rate: {churned.mean():.1%}")
    print(f"Plan distribution:\n{df['plan'].value_counts().to_string()}")
    return df


if __name__ == '__main__':
    df = generate_churn_dataset()
    df.to_csv('/home/claude/churn_project/data/customers.csv', index=False)
    print(f"\nSaved to data/customers.csv  shape={df.shape}")

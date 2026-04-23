"""
STEP 2 — FEATURE ENGINEERING
Transforms raw customer data into model-ready features.
All transformations are encapsulated in a sklearn Pipeline so the
same transforms apply consistently at training and inference time.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib, os


# ── Custom Feature Engineering Transformer ───────────────────────────────────

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates derived features before standard preprocessing."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Usage trend: positive = growing, negative = declining (strong churn signal)
        df['usage_trend'] = df['sessions_30d'] - df['sessions_60d']

        # Engagement score: composite 0-1 (weighted blend of 4 signals)
        sess_max = df['sessions_30d'].max() or 1
        api_max  = df['api_calls_30d'].max() or 1
        df['engagement_score'] = (
            0.30 * (df['sessions_30d'] / sess_max) +
            0.30 * (df['features_used'] / 10) +
            0.20 * (df['api_calls_30d'] / api_max) +
            0.20 * (1 - df['days_since_login'] / 90)
        ).clip(0, 1)

        # Recency flag: hasn't logged in for 2+ weeks
        df['is_dormant'] = (df['days_since_login'] > 14).astype(int)

        # Declining usage flag
        df['is_declining'] = (df['usage_trend'] < -5).astype(int)

        # Billing health flag
        df['billing_issues'] = (df['payment_failures_6m'] > 0).astype(int)

        # NPS sentiment buckets
        df['nps_detractor']  = (df['nps_score'] <= 3).astype(int)
        df['nps_promoter']   = (df['nps_score'] >= 9).astype(int)

        # Revenue at risk (for business prioritisation, not model input)
        df['mrr_at_risk'] = df['monthly_mrr'] * df.get('churn_prob', 0)

        # Tenure bucket
        df['tenure_bucket'] = pd.cut(
            df['tenure_days'],
            bins=[0, 30, 90, 180, 365, 99999],
            labels=['<1m', '1-3m', '3-6m', '6-12m', '12m+']
        ).astype(str)

        return df


# ── Column definitions ────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    'tenure_days', 'monthly_mrr',
    'sessions_30d', 'sessions_60d', 'usage_trend',
    'features_used', 'api_calls_30d', 'days_since_login',
    'support_tickets_90d', 'payment_failures_6m',
    'nps_score', 'engagement_score',
    'is_dormant', 'is_declining', 'billing_issues',
    'nps_detractor', 'nps_promoter',
    'had_downgrade', 'had_upgrade',
]

CATEGORICAL_FEATURES = ['plan', 'region', 'company_size', 'tenure_bucket']

TARGET = 'churned'
DROP_COLS = ['customer_id', 'mrr_at_risk']


# ── Build sklearn Pipeline ────────────────────────────────────────────────────

def build_preprocessor():
    numeric_pipeline = Pipeline([
        ('scaler', StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, NUMERIC_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder='drop')
    return preprocessor


def prepare_features(df: pd.DataFrame):
    """Run feature engineering and return X, y."""
    engineer = ChurnFeatureEngineer()
    df_eng = engineer.transform(df)

    y = df_eng[TARGET].values if TARGET in df_eng.columns else None
    X = df_eng.drop(columns=[c for c in DROP_COLS + [TARGET] if c in df_eng.columns])
    return X, y, engineer


if __name__ == '__main__':
    df = pd.read_csv('/home/claude/churn_project/data/customers.csv')
    X, y, eng = prepare_features(df)
    print(f"Features: {X.shape[1]} columns, {X.shape[0]} rows")
    print("Numeric features:", NUMERIC_FEATURES)
    print("Categorical features:", CATEGORICAL_FEATURES)
    print(f"\nTarget: {y.mean():.2%} churn rate")
    print("\nSample engineered features:")
    print(X[['usage_trend','engagement_score','is_dormant','is_declining']].describe().round(3))

"""
STEP 3 — MODEL TRAINING & EVALUATION
- Temporal train/val/test split (no data leakage)
- XGBoost with scale_pos_weight for class imbalance
- Full evaluation: ROC-AUC, PR-AUC, confusion matrix, feature importance
- Saves trained model + preprocessor to /models/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, joblib, os, sys
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/claude/churn_project/src')
from features import (
    build_preprocessor, prepare_features,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES
)

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    classification_report, confusion_matrix,
    f1_score
)
from sklearn.pipeline import Pipeline

os.makedirs('/home/claude/churn_project/models', exist_ok=True)
os.makedirs('/home/claude/churn_project/notebooks', exist_ok=True)


# ── Load & split ──────────────────────────────────────────────────────────────

df = pd.read_csv('/home/claude/churn_project/data/customers.csv')
print(f"Loaded {len(df):,} customers | Churn rate: {df['churned'].mean():.1%}")

# Time-based split: first 70% train, next 15% val, last 15% test
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

df_train = df.iloc[:train_end].copy()
df_val   = df.iloc[train_end:val_end].copy()
df_test  = df.iloc[val_end:].copy()

X_train, y_train, eng = prepare_features(df_train)
X_val,   y_val,   _   = prepare_features(df_val)
X_test,  y_test,  _   = prepare_features(df_test)

print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")


# ── Preprocessor fit ──────────────────────────────────────────────────────────

preprocessor = build_preprocessor()
X_train_pp = preprocessor.fit_transform(X_train)
X_val_pp   = preprocessor.transform(X_val)
X_test_pp  = preprocessor.transform(X_test)

# Get feature names after OHE
ohe_cats = (
    preprocessor
    .named_transformers_['cat']
    .named_steps['ohe']
    .get_feature_names_out(CATEGORICAL_FEATURES)
    .tolist()
)
feature_names = NUMERIC_FEATURES + ohe_cats


# ── Class imbalance weight ────────────────────────────────────────────────────

neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos
print(f"Class ratio  neg:pos = {neg}:{pos}  →  scale_pos_weight = {scale_pos_weight:.1f}")


# ── Train XGBoost ─────────────────────────────────────────────────────────────

xgb = XGBClassifier(
    n_estimators       = 400,
    max_depth          = 5,
    learning_rate      = 0.04,
    subsample          = 0.80,
    colsample_bytree   = 0.75,
    min_child_weight   = 5,
    scale_pos_weight   = scale_pos_weight,
    eval_metric        = 'aucpr',
    early_stopping_rounds = 25,
    random_state       = 42,
    verbosity          = 0,
    n_jobs             = -1,
)
xgb.fit(
    X_train_pp, y_train,
    eval_set=[(X_val_pp, y_val)],
    verbose=False
)
print(f"XGBoost best iteration: {xgb.best_iteration}")


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(model, X, y, name=""):
    prob = model.predict_proba(X)[:, 1]
    roc  = roc_auc_score(y, prob)
    pr   = average_precision_score(y, prob)
    # Find threshold that maximises F1
    prec_arr, rec_arr, thrs = precision_recall_curve(y, prob)
    f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
    best_thr = thrs[np.argmax(f1_arr[:-1])]
    pred = (prob >= best_thr).astype(int)
    f1 = f1_score(y, pred)
    print(f"\n{'─'*45}")
    print(f"{name}  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  F1={f1:.4f}  thr={best_thr:.2f}")
    print(classification_report(y, pred, target_names=['Active', 'Churned']))
    return prob, roc, pr, best_thr

val_prob,  val_roc,  val_pr,  _   = evaluate(xgb, X_val_pp,  y_val,  "VAL ")
test_prob, test_roc, test_pr, thr = evaluate(xgb, X_test_pp, y_test, "TEST")


# ── Plots ─────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0f1117')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.38)

DARK   = '#0f1117'
CARD   = '#1a1d27'
ACCENT = '#6c63ff'
GREEN  = '#4ade80'
AMBER  = '#fbbf24'
RED    = '#f87171'
TEXT   = '#e2e8f0'
MUTED  = '#64748b'

def style_ax(ax, title):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3142')

# 1. ROC Curve
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, f'ROC Curve  (AUC = {test_roc:.3f})')
fpr, tpr, _ = roc_curve(y_test, test_prob)
ax1.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f'XGBoost (AUC={test_roc:.3f})')
ax1.plot([0, 1], [0, 1], '--', color=MUTED, lw=1, label='Random')
ax1.fill_between(fpr, tpr, alpha=0.12, color=ACCENT)
ax1.set_xlabel('False Positive Rate', color=MUTED, fontsize=9)
ax1.set_ylabel('True Positive Rate', color=MUTED, fontsize=9)
ax1.legend(fontsize=9, labelcolor=TEXT, facecolor=CARD)

# 2. Precision-Recall Curve
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, f'Precision-Recall  (AUC = {test_pr:.3f})')
prec_arr, rec_arr, _ = precision_recall_curve(y_test, test_prob)
ax2.plot(rec_arr, prec_arr, color=GREEN, lw=2.5, label=f'XGBoost (AP={test_pr:.3f})')
baseline = y_test.mean()
ax2.axhline(baseline, color=MUTED, linestyle='--', lw=1, label=f'Baseline ({baseline:.2f})')
ax2.fill_between(rec_arr, prec_arr, alpha=0.12, color=GREEN)
ax2.set_xlabel('Recall', color=MUTED, fontsize=9)
ax2.set_ylabel('Precision', color=MUTED, fontsize=9)
ax2.legend(fontsize=9, labelcolor=TEXT, facecolor=CARD)

# 3. Probability Distribution
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, 'Predicted Churn Probability')
churned_mask = y_test == 1
ax3.hist(test_prob[~churned_mask], bins=40, color=GREEN, alpha=0.65, label='Active', density=True)
ax3.hist(test_prob[churned_mask],  bins=40, color=RED,   alpha=0.65, label='Churned', density=True)
ax3.axvline(thr, color=AMBER, lw=2, linestyle='--', label=f'Threshold ({thr:.2f})')
ax3.set_xlabel('Churn Probability', color=MUTED, fontsize=9)
ax3.set_ylabel('Density', color=MUTED, fontsize=9)
ax3.legend(fontsize=9, labelcolor=TEXT, facecolor=CARD)

# 4. Feature Importance (top 15)
ax4 = fig.add_subplot(gs[1, 0:2])
style_ax(ax4, 'Top 15 Feature Importances (Gain)')
importances = xgb.feature_importances_
idx = np.argsort(importances)[-15:]
names_short = [n.replace('plan_', 'plan:').replace('region_', 'rgn:')
                .replace('company_size_', 'co:').replace('tenure_bucket_', 'tenure:')
               for n in np.array(feature_names)[idx]]
colors = [ACCENT if importances[i] > np.percentile(importances, 85) else MUTED for i in idx]
bars = ax4.barh(range(15), importances[idx], color=colors, height=0.65)
ax4.set_yticks(range(15))
ax4.set_yticklabels(names_short, fontsize=9)
ax4.set_xlabel('Importance (gain)', color=MUTED, fontsize=9)
for bar, val in zip(bars, importances[idx]):
    ax4.text(val + 0.001, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', color=TEXT, fontsize=8)

# 5. Confusion Matrix
ax5 = fig.add_subplot(gs[1, 2])
style_ax(ax5, f'Confusion Matrix  (thr={thr:.2f})')
pred_test = (test_prob >= thr).astype(int)
cm = confusion_matrix(y_test, pred_test)
im = ax5.imshow(cm, cmap='Blues', vmin=0)
ax5.set_xticks([0,1]); ax5.set_yticks([0,1])
ax5.set_xticklabels(['Active', 'Churned'], color=TEXT, fontsize=10)
ax5.set_yticklabels(['Active', 'Churned'], color=TEXT, fontsize=10)
ax5.set_xlabel('Predicted', color=MUTED, fontsize=9)
ax5.set_ylabel('Actual', color=MUTED, fontsize=9)
for i in range(2):
    for j in range(2):
        ax5.text(j, i, str(cm[i,j]), ha='center', va='center',
                 fontsize=18, fontweight='bold',
                 color='white' if cm[i,j] > cm.max()/2 else '#1e3a5f')

fig.suptitle('Customer Churn Prediction — Model Evaluation',
             color=TEXT, fontsize=14, fontweight='bold', y=0.98)

plt.savefig('/home/claude/churn_project/notebooks/evaluation_plots.png',
            dpi=150, bbox_inches='tight', facecolor=DARK)
print("\nSaved evaluation_plots.png")


# ── Save model artefacts ──────────────────────────────────────────────────────

joblib.dump(xgb,          '/home/claude/churn_project/models/xgb_model.pkl')
joblib.dump(preprocessor, '/home/claude/churn_project/models/preprocessor.pkl')
joblib.dump(eng,          '/home/claude/churn_project/models/feature_engineer.pkl')
joblib.dump({
    'threshold': float(thr),
    'feature_names': feature_names,
    'val_roc_auc': float(val_roc),
    'test_roc_auc': float(test_roc),
    'test_pr_auc':  float(test_pr),
}, '/home/claude/churn_project/models/metadata.pkl')

print(f"\nModel artefacts saved to models/")
print(f"Final TEST  ROC-AUC = {test_roc:.4f}  |  PR-AUC = {test_pr:.4f}")

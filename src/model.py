import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
MODEL_PATH    = os.path.join(MODELS_DIR, "fifa_model.pkl")
SHAP_PATH     = os.path.join(MODELS_DIR, "shap_importance.png")

# ── Feature sets ──────────────────────────────────────────────────────────────
BASE_FEATURES = [
    "rank_diff",
    "team1_win_rate",
    "team2_win_rate",
    "head_to_head",
    "stage_num",
    "team1_avg_goals",
    "team2_avg_goals",
    "team1_is_host",
    "coach_exp_diff",
]

NEW_FEATURES = BASE_FEATURES + [
    "team1_conf_strength",
    "team2_conf_strength",
    "conf_strength_diff",
    "team1_recent_form",
    "team2_recent_form",
    "form_diff",
]

TARGET = "home_win"

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(FEATURES_PATH)
y  = df[TARGET]

print(f"Dataset: {len(df)} rows")
print(f"Class balance — home_win=1: {y.mean():.1%}  home_win=0: {(1-y).mean():.1%}\n")

# ── Shared train/test split (same random state for fair comparison) ───────────
X_base = df[BASE_FEATURES]
X_new  = df[NEW_FEATURES]

_, _, y_train, y_test = train_test_split(X_base, y, test_size=0.2,
                                          random_state=42, stratify=y)
X_base_train, X_base_test = train_test_split(X_base, test_size=0.2,
                                               random_state=42, stratify=y)
X_new_train,  X_new_test  = train_test_split(X_new,  test_size=0.2,
                                               random_state=42, stratify=y)

# ── Model factory ─────────────────────────────────────────────────────────────
def make_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100, eval_metric="logloss",
            use_label_encoder=False, random_state=42, verbosity=0,
        ),
    }

def train_eval(X_train, X_test, y_train, y_test):
    results, trained = {}, {}
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC-ROC":  roc_auc_score(y_test, y_proba),
        }
        trained[name] = model
    return results, trained

# ── Train BEFORE (base features) ─────────────────────────────────────────────
print("Training with BASE features (9 features) …")
before_results, _ = train_eval(X_base_train, X_base_test, y_train, y_test)

# ── Train AFTER (base + confederation features) ───────────────────────────────
print("Training with NEW features (+confederation, 12 features) …\n")
after_results, after_trained = train_eval(X_new_train, X_new_test, y_train, y_test)

# ── Before / After comparison table ──────────────────────────────────────────
print("=" * 70)
print(f"{'Model':<22}  {'Before Acc':>10}  {'After Acc':>10}  "
      f"{'Before AUC':>10}  {'After AUC':>10}  {'ΔAUC':>7}")
print("-" * 70)
for name in before_results:
    b = before_results[name]
    a = after_results[name]
    delta = a["AUC-ROC"] - b["AUC-ROC"]
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
    print(f"{name:<22}  {b['Accuracy']:>10.4f}  {a['Accuracy']:>10.4f}  "
          f"{b['AUC-ROC']:>10.4f}  {a['AUC-ROC']:>10.4f}  "
          f"{arrow}{abs(delta):>5.4f}")
print("=" * 70)

# ── Select & save best new model ──────────────────────────────────────────────
best_name  = max(after_results, key=lambda n: after_results[n]["AUC-ROC"])
best_model = after_trained[best_name]
best_acc   = after_results[best_name]["Accuracy"]
best_auc   = after_results[best_name]["AUC-ROC"]

joblib.dump(best_model, MODEL_PATH)
print(f"\nBest model : {best_name}")
print(f"  Accuracy : {best_acc:.4f}")
print(f"  AUC-ROC  : {best_auc:.4f}")
print(f"\nSaved → {MODEL_PATH}")

# ── SHAP feature importance (XGBoost on new features) ────────────────────────
xgb_model   = after_trained["XGBoost"]
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_new_test)

mean_shap = np.abs(shap_values).mean(axis=0)
importance = sorted(zip(NEW_FEATURES, mean_shap), key=lambda x: x[1], reverse=True)

print("\nSHAP Feature Importance (XGBoost, ranked):")
print("-" * 38)
for rank, (feat, val) in enumerate(importance, 1):
    bar = "█" * int(val / max(mean_shap) * 20)
    print(f"  {rank:>2}. {feat:<22}  {val:.4f}  {bar}")
print("-" * 38)

fig, ax = plt.subplots(figsize=(8, 5))
shap.summary_plot(
    shap_values, X_new_test,
    feature_names=NEW_FEATURES,
    plot_type="bar", show=False, color="#e05a2b",
)
plt.title("XGBoost — SHAP Feature Importance (with Confederation)", fontsize=13, pad=12)
plt.tight_layout()
fig.savefig(SHAP_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSHAP plot  → {SHAP_PATH}")

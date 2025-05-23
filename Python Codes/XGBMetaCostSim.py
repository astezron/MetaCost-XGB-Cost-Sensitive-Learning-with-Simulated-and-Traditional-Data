import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
import warnings

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, cohen_kappa_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# ================================
# 1. Load Dataset
# ================================
train_df = pd.read_csv("Sample Dataset/Simu_TrainDemo.csv")
test_df = pd.read_csv("Sample Dataset/Simu_TestDemo.csv")

# ================================
# 2. Predictors & Target
# ================================
predictors = ['As', 'Au', 'Cu', 'Mo', 'bn', 'cp', 'cc', 'cv', 'en', 'py', 'po', 'mo', 'ga', 'sl', 'TS_ppm']
target = 'Alteration'

# ================================
# 3. Encode Target
# ================================
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df[target])
y_test = label_encoder.transform(test_df[target])

X_train = train_df[predictors].astype(np.float32).values
X_test = test_df[predictors].astype(np.float32).values

# ================================
# 4. Class Weights
# ================================
class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = {i: np.log1p(total_samples / class_counts[i]) for i in range(len(class_counts))}

# ================================
# 5. Cost Matrix
# ================================
cost_matrix = np.array([
    [0, 6, 6, 7, 8, 13],
    [5, 0, 5, 6, 10, 11],
    [6, 5, 0, 6, 4, 9],
    [7, 7, 7, 0, 6, 5],
    [8, 11, 7, 6, 0, 3],
    [14, 12, 10, 5, 3, 0]], dtype=float)

cost_matrix[4, :] *= 1.5
cost_matrix[:, 4] *= 1.5
cost_matrix[5, :] *= 1.5
cost_matrix[:, 5] *= 1.5
normalized_cost_matrix = cost_matrix / np.max(cost_matrix)

# ================================
# 6. Bayesian Optimization (Optuna)
# ================================
print("\nStarting Bayesian optimization using Optuna...")

def objective(trial):
    params = {
        'device': 'cuda',
        'tree_method': 'hist',
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'max_depth': trial.suggest_categorical('max_depth', [8, 9, 10]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.05, 0.07]),
        'n_estimators': trial.suggest_categorical('n_estimators', [800, 1500]),
        'subsample': trial.suggest_categorical('subsample', [0.9, 1.0]),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.8, 0.9]),
        'gamma': trial.suggest_categorical('gamma', [0.1, 0.2]),
        'min_child_weight': trial.suggest_categorical('min_child_weight', [2, 3]),
        'reg_alpha': trial.suggest_categorical('reg_alpha', [0.01, 0.1]),
        'reg_lambda': trial.suggest_categorical('reg_lambda', [0.3, 0.5])
    }

    model = xgb.XGBClassifier(**params)
    weights = np.array([class_weights[y] for y in y_train])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        w_tr = weights[train_idx]

        model.fit(X_tr, y_tr, sample_weight=w_tr)
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds, average='weighted'))

    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5, show_progress_bar=True)

best_params_tuned = study.best_params
best_params_tuned.update({
    'device': 'cuda',
    'tree_method': 'hist',
    'eval_metric': 'mlogloss',
    'random_state': 42
})

print("\nBest parameters found:")
for k, v in best_params_tuned.items():
    print(f"{k}: {v}")

# ================================
# 7. MetaCost Class
# ================================
class OptimizedMetaCost(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier=None, confidence_threshold=0.70, min_cost_reduction=0.02, cv_splits=10, random_state=None):
        self.base_classifier = base_classifier if base_classifier is not None else xgb.XGBClassifier(**best_params_tuned)
        self.confidence_threshold = confidence_threshold
        self.min_cost_reduction = min_cost_reduction
        self.cv_splits = cv_splits
        self.random_state = random_state

    def fit(self, X, y):
        np.random.seed(self.random_state)
        skf = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        classifiers = []

        for train_idx, _ in skf.split(X, y):
            sample_weights = [class_weights[label] for label in y[train_idx]]
            clf = clone(self.base_classifier).fit(X[train_idx], y[train_idx], sample_weight=sample_weights)
            classifiers.append(clf)

        prob_matrix = np.mean([clf.predict_proba(X) for clf in classifiers], axis=0)
        expected_costs = np.dot(prob_matrix, normalized_cost_matrix)
        new_labels = np.argmin(expected_costs, axis=1)

        original_costs = expected_costs[np.arange(len(y)), y]
        new_costs = expected_costs[np.arange(len(y)), new_labels]
        cost_reduction = (original_costs - new_costs) / (original_costs + 1e-6)
        confidence = np.max(prob_matrix, axis=1)

        relabel_mask = (cost_reduction > self.min_cost_reduction) & (confidence > self.confidence_threshold)
        y_transformed = np.where(relabel_mask, new_labels, y)

        self.relabel_mask_ = relabel_mask
        self.original_labels_ = y
        self.relabelled_labels_ = y_transformed

        num_relabels = np.sum(relabel_mask)
        print(f"\nMetaCost re-labeled {num_relabels} instances out of {len(y)} ({100 * num_relabels / len(y):.2f}%)")
        print(f"Avg training cost reduced: {np.mean(original_costs):.4f} -> {np.mean(expected_costs[np.arange(len(y)), y_transformed]):.4f}")

        final_weights = [class_weights[label] for label in y_transformed]
        self.final_classifier_ = clone(self.base_classifier).fit(X, y_transformed, sample_weight=final_weights)
        return self

    def predict(self, X):
        return self.final_classifier_.predict(X)
    
    def predict_proba(self, X):
        return self.final_classifier_.predict_proba(X)

# ================================
# 8. Train Baseline Model
# ================================
print("\nTraining Baseline XGBoost...")
baseline_model = xgb.XGBClassifier(**best_params_tuned)
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)

print("\nBaseline XGBoost Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_baseline))
print("Recall:", recall_score(y_test, y_pred_baseline, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_baseline, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_baseline, average='weighted'))
print("Kappa:", cohen_kappa_score(y_test, y_pred_baseline))
print("ROC-AUC:", roc_auc_score(y_test, baseline_model.predict_proba(X_test), multi_class='ovr'))

# ================================
# 9. Train MetaCost 
# ================================
meta_model = OptimizedMetaCost(random_state=42)
meta_model.fit(X_train, y_train)

meta_probs = meta_model.predict_proba(X_test)
meta_pred = meta_model.predict(X_test)
meta_conf = np.max(meta_probs, axis=1)
expected_cost_meta = np.dot(meta_probs, normalized_cost_matrix)[np.arange(len(y_test)), meta_pred]
expected_cost_base = np.dot(meta_probs, normalized_cost_matrix)[np.arange(len(y_test)), y_pred_baseline]

override_mask = (meta_pred != y_pred_baseline) & (meta_conf > 0.75) & ((expected_cost_base - expected_cost_meta) > 0.01)
y_pred_final = np.where(override_mask, meta_pred, y_pred_baseline)

print("\nFirst 5 relabeled samples:")
relabel_indices = np.where(meta_model.relabel_mask_)[0]
for idx in relabel_indices[:5]:
    orig = label_encoder.inverse_transform([meta_model.original_labels_[idx]])[0]
    new = label_encoder.inverse_transform([meta_model.relabelled_labels_[idx]])[0]
    print(f"Index {idx}: {orig} -> {new}")

print("\nMetaCost Final Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_final))
print("Recall:", recall_score(y_test, y_pred_final, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_final, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_final, average='weighted'))
print("Kappa:", cohen_kappa_score(y_test, y_pred_final))
print("ROC-AUC:", roc_auc_score(y_test, meta_probs, multi_class='ovr'))

# ================================
# 10. Visualizations & Save
# ================================
conf_matrix = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - MetaCost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

feature_importance = meta_model.final_classifier_.feature_importances_
plt.figure(figsize=(10, 7))
plt.barh(predictors, feature_importance)
plt.title('Feature Importances - MetaCost XGBoost')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

test_df['pred'] = label_encoder.inverse_transform(y_pred_final)
test_df.to_csv("MetaCostPredictions.csv", index=False)
print("\nMetaCost predictions saved.")


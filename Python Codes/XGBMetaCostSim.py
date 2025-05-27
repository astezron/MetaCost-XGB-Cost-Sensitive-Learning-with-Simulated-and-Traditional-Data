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
from sklearn.calibration import CalibratedClassifierCV

# ================================
# 1. Load Dataset
# ================================
train_df = pd.read_csv("../Sample Dataset/Simu_TrainDemo.csv")
test_df = pd.read_csv("../Sample Dataset/Simu_TestDemo.csv")

# ================================
# 2. Predictors & Target
# ================================
predictors = ['Cu', 'Au', 'Mo', 'As', 'Bn', 'Cp', 'Cc', 'Cv', 'En', 'Py', 'Pyr', 'Mol', 'Ga', 'Sph', 'TS']
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

working_cost_matrix = cost_matrix.copy()

# ================================
# 6. Bayesian Optimization (Optuna)
# ================================
print("\nStarting Bayesian optimization using Optuna...")

def objective(trial):
    params = {
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
    def __init__(self, base_classifier=None, confidence_threshold=0.30, min_cost_reduction=0.001, cv_splits=5, random_state=None):
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
        expected_costs = np.dot(prob_matrix, working_cost_matrix)
        new_labels = np.argmin(expected_costs, axis=1)

        original_costs = expected_costs[np.arange(len(y)), y]
        new_costs = expected_costs[np.arange(len(y)), new_labels]
        cost_reduction = original_costs - new_costs  # Absolute cost reduction
        confidence = np.max(prob_matrix, axis=1)
        
        print(f"Confidence stats: Min={np.min(confidence):.2f}, Mean={np.mean(confidence):.2f}, Max={np.max(confidence):.2f}")
        print(f"Cost reduction stats: Min={np.min(cost_reduction):.4f}, Mean={np.mean(cost_reduction):.4f}, Max={np.max(cost_reduction):.4f}")

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
# 8. Train MetaCost with Calibration
# ================================
print("\nCreating calibrated base classifier...")
calibrated_base = CalibratedClassifierCV(
    xgb.XGBClassifier(**best_params_tuned),
    method='isotonic',
    cv=3,
    n_jobs=-1
)

print("\nTraining MetaCost...")
meta_model = OptimizedMetaCost(
    base_classifier=calibrated_base,
    confidence_threshold=0.30,
    min_cost_reduction=0.5,
    cv_splits=5,
    random_state=42
)
meta_model.fit(X_train, y_train)

meta_probs = meta_model.predict_proba(X_test)
meta_pred = meta_model.predict(X_test)

print("\nMetaCost Performance:")
print("Accuracy:", accuracy_score(y_test, meta_pred))
print("Recall:", recall_score(y_test, meta_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, meta_pred, average='weighted', zero_division=0))
print("Precision:", precision_score(y_test, meta_pred, average='weighted', zero_division=0))
print("Kappa:", cohen_kappa_score(y_test, meta_pred))
print("ROC-AUC:", roc_auc_score(y_test, meta_probs, multi_class='ovr'))

# ================================
# 9. Visualizations & Save
# ================================
conf_matrix = confusion_matrix(y_test, meta_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - MetaCost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

try:
    xgb_model = xgb.XGBClassifier(**best_params_tuned)
    xgb_model.fit(X_train, y_train)
    
    feature_importance = xgb_model.feature_importances_
    
    plt.figure(figsize=(10, 7))
    plt.barh(predictors, feature_importance)
    plt.title('Feature Importances - XGBoost Model')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"\nCould not plot feature importances: {str(e)}")

test_df['pred'] = label_encoder.inverse_transform(meta_pred)
test_df.to_csv("../Sample Dataset/MetaCostPredictions.csv", index=False)
print("\nMetaCost predictions saved.") 

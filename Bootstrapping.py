import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# Define predictors and target
predictors = ['As', 'Au', 'Cu', 'Mo', 'bn_ppm', 'cp_ppm', 'cc_ppm', 'cv_ppm',
              'en_ppm', 'py_ppm', 'po_ppm', 'mb_ppm', 'ga_ppm', 'sp_ppm', 'TS_ppm']
target = 'Alteration'

# Load data
train_df = pd.read_csv(r"D:\Final_Alteration\Data\Master\train_med.csv")
test_df = pd.read_csv(r"D:\Final_Alteration\Data\Master\test_med.csv")

X_train = train_df[predictors]
X_test = test_df[predictors]

# Label encode target variable
le = LabelEncoder()
y_train = le.fit_transform(train_df[target].astype(str))
y_test = le.transform(test_df[target].astype(str))

# Define evaluation function for Bayesian Optimization
def xgb_evaluate(max_depth, learning_rate, n_estimators):
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        n_estimators=int(n_estimators),
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

# Bayesian Optimization
optimizer = BayesianOptimization(
    f=xgb_evaluate,
    pbounds={
        "max_depth": (3, 10),
        "learning_rate": (0.01, 0.3),
        "n_estimators": (500, 1000)
    },
    random_state=42,
    verbose=0
)
optimizer.maximize(init_points=5, n_iter=10)

# Extract best hyperparameters
params = optimizer.max['params']
params['max_depth'] = int(params['max_depth'])
params['n_estimators'] = int(params['n_estimators'])

# Bootstrap Ensemble
n_bootstraps = 100
predictions = []

for i in range(n_bootstraps):
    sample_indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_boot = X_train.iloc[sample_indices]
    y_boot = y_train[sample_indices]

    model = XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0,
        **params
    )
    model.fit(X_boot, y_boot)
    preds = model.predict(X_test)
    predictions.append(preds)

# Transpose predictions: (n_test_samples, n_bootstraps)
predictions = np.array(predictions).T

# Most frequent prediction and confidence score per test sample
final_results = []
for sample_preds in predictions:
    counter = Counter(sample_preds)
    most_common_class, count = counter.most_common(1)[0]
    probability = count / n_bootstraps
    decoded_class = le.inverse_transform([most_common_class])[0]
    final_results.append((decoded_class, probability))

# Decode true labels
true_classes = le.inverse_transform(y_test)

# Create and save results DataFrame
results_df = pd.DataFrame(final_results, columns=["Predicted_Class", "Confidence_Probability"])
results_df["True_Class"] = true_classes

results_df.to_csv("D:/Final_Alteration/Data/Trad_ppm/bootstrap_prediction_uncertaintyNew.csv", index=False)
print(results_df.head())

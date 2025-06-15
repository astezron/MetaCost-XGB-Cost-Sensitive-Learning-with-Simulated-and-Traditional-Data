import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score,
    average_precision_score, cohen_kappa_score,
    precision_score, recall_score, f1_score )
import numpy as np 

# === Load Data ===
file_path = "MetaCostPredictionsMode.csv"
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()

if 'ModePred' not in data.columns or 'Alteration' not in data.columns:
    raise ValueError("The columns 'ModePred' or 'Alteration' are not found in the CSV file.")

valid_rows = data[data['ModePred'].notna() & (data['ModePred'] != "") & 
                  data['Alteration'].notna() & (data['Alteration'] != "")]

valid_rows.loc[:, 'Alteration'] = valid_rows['Alteration'].astype(str)
valid_rows.loc[:, 'ModePred'] = valid_rows['ModePred'].astype(str)

print("Counts of Alteration categories:")
print(valid_rows['Alteration'].value_counts())

print("Counts of ModePred categories:")
print(valid_rows['ModePred'].value_counts())

# === Evaluation ===
y_true = valid_rows['Alteration']
y_pred = valid_rows['ModePred']

fixed_order = ['AAA', 'IAA', 'PHY', 'PRO', 'PTS', 'UAL']
cm = confusion_matrix(y_true, y_pred, labels=fixed_order)

accuracy = accuracy_score(y_true, y_pred) * 100
error_rate = 100 - accuracy

roc_auc = roc_auc_score(pd.get_dummies(y_true), pd.get_dummies(y_pred), multi_class='ovr', average='macro')
prc_auc = average_precision_score(pd.get_dummies(y_true), pd.get_dummies(y_pred), average='macro')

precision = np.diag(cm) / (cm.sum(axis=0) + 1e-10)
recall = np.diag(cm) / (cm.sum(axis=1) + 1e-10)
f_measure = 2 * (precision * recall) / (precision + recall)
f_measure = np.nan_to_num(f_measure)

specificity = [(cm.sum() - cm[:, i].sum() - cm[i, :].sum() + cm[i, i]) / 
               (cm.sum() - cm[i, :].sum() + 1e-10) for i in range(len(fixed_order))]

sensitivity = recall
kappa = cohen_kappa_score(y_true, y_pred)

overall_precision = np.mean(precision)
overall_recall = np.mean(recall)
overall_f_measure = np.mean(f_measure)
overall_specificity = np.mean(specificity)
overall_sensitivity = np.mean(sensitivity)

weighted_precision = precision_score(y_true, y_pred, average='weighted')
weighted_recall = recall_score(y_true, y_pred, average='weighted')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')

# === Print Results ===
print("\n=== Evaluation Metrics ===")
print(f"Correct: {cm.trace()}")
print(f"Total: {cm.sum()}")
print(f"Accuracy: {accuracy / 100:.2f}")
print(f"Error Rate: {error_rate / 100:.2f}")
print(f"ROC Area : {roc_auc:.2f}")
print(f"PRC Area : {prc_auc:.2f}")
print(f"Cohen's Kappa: {kappa:.2f}")
print(f"Precision : {weighted_precision:.2f}")
print(f"Recall : {weighted_recall:.2f}")
print(f"F1 Score : {weighted_f1:.2f}")

print("\n=== Class-wise Metrics ===")
for i, label in enumerate(fixed_order):
    print(f"Class {label}: Precision: {precision[i]:.6f}, Recall: {recall[i]:.6f}, F-Measure: {f_measure[i]:.6f}, Specificity: {specificity[i]:.6f}")

print("\n=== Overall Metrics ===")
print(f"Overall Specificity: {overall_specificity:.2f}")
print(f"Overall Sensitivity: {overall_sensitivity:.2f}")

# === Confusion Matrix Plot ===
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=fixed_order, yticklabels=fixed_order,
            cbar=True, linewidths=0.5, linecolor='white')
plt.title("Confusion Matrix - MetaCost", fontsize=14, fontweight='bold')
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig("ConfusionMatrix_FixedOrder.png")
plt.close()

# === Match Count Plot ===
match_counts = valid_rows['ModePred'] == valid_rows['Alteration']
match_counts = match_counts.value_counts().reset_index()
match_counts.columns = ['Match', 'Count']

plt.figure(figsize=(8, 5))
sns.barplot(data=match_counts, x='Match', y='Count', hue='Match', palette='pastel', legend=False)
plt.title("Count of Matches Between ModePred and Alteration", fontsize=16, fontweight='bold')
plt.xlabel("Match (True/False)", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(ticks=[0, 1], labels=['No Match', 'Match'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("MatchCountsPlot.png")
plt.close()


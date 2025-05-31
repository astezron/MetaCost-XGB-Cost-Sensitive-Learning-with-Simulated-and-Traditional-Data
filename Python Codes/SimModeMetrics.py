import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, average_precision_score, cohen_kappa_score
import numpy as np  # Import numpy for array operations

# Set the path for your CSV file
file_path = "../Sample Dataset/MetaCost_PredictionsMode.csv"# Update with your file path

# Read the CSV file
data = pd.read_csv(file_path)

# Strip any leading or trailing spaces from the column names
data.columns = data.columns.str.strip()

# Check that the necessary columns exist
if 'ModePred' not in data.columns or 'Alteration' not in data.columns:
    raise ValueError("The columns 'ModePred' or 'Alteration' are not found in the CSV file.")

# Filter the DataFrame to only include rows where both ModePredicted and Alteration are filled
valid_rows = data[data['ModePred'].notna() & (data['ModePred'] != "") & 
                  data['Alteration'].notna() & (data['Alteration'] != "")]

# Convert both columns to string to avoid type issues
valid_rows.loc[:, 'Alteration'] = valid_rows['Alteration'].astype(str)
valid_rows.loc[:, 'ModePred'] = valid_rows['ModePred'].astype(str)

# Check counts for diagnostic
print("Counts of Alteration categories:")
print(valid_rows['Alteration'].value_counts())

print("Counts of ModePred categories:")
print(valid_rows['ModePred'].value_counts())

# Create confusion matrix data
y_true = valid_rows['Alteration']
y_pred = valid_rows['ModePred']

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=y_true.unique())

# Accuracy
accuracy = accuracy_score(y_true, y_pred) * 100  # Accuracy in percentage
error_rate = 100 - accuracy  # Error rate

# ROC and PRC Area calculations (multiclass AUC)
roc_auc = roc_auc_score(pd.get_dummies(y_true), pd.get_dummies(y_pred), multi_class='ovr', average='macro')
prc_auc = average_precision_score(pd.get_dummies(y_true), pd.get_dummies(y_pred), average='macro')

# Class-wise Metrics (Precision, Recall, F-Measure, Specificity)
precision = np.diag(cm) / (cm.sum(axis=0) + 1e-10)  # Precision per class
recall = np.diag(cm) / (cm.sum(axis=1) + 1e-10)  # Recall per class
f_measure = 2 * (precision * recall) / (precision + recall)  # F-measure per class

# Handle division by zero in F-measure (replace nan with 0 for classes with no precision or recall)
f_measure = np.nan_to_num(f_measure)

specificity = [(cm.sum() - cm[:, i].sum() - cm[i, :].sum() + cm[i, i]) / (cm.sum() - cm[i, :].sum() + 1e-10) for i in range(len(y_true.unique()))]  # Specificity per class
sensitivity = recall  # Sensitivity is the same as recall

# Cohen's Kappa calculation
kappa = cohen_kappa_score(y_true, y_pred)

# Overall metrics (average of per-class metrics)
overall_precision = np.mean(precision)
overall_recall = np.mean(recall)
overall_f_measure = np.mean(f_measure)
overall_specificity = np.mean(specificity)
overall_sensitivity = np.mean(sensitivity)

# Print results
print("\n=== Evaluation Metrics ===")
print(f"Correct: {cm.trace()}")  # Correct predictions (diagonal sum)
print(f"Total: {cm.sum()}")  # Total predictions
print(f"Accuracy: {accuracy / 100:.2f}")
print(f"Error Rate: {error_rate / 100:.2f}")
print(f"ROC Area: {roc_auc:.2f}")
print(f"PRC Area: {prc_auc:.2f}")
print(f"Cohen's Kappa: {kappa:.2f}")

# Print class-wise metrics
print("\n=== Class-wise Metrics ===")
class_labels = y_true.unique()
for i, label in enumerate(class_labels):
    print(f"Class {label}: Precision: {precision[i]:.6f}, Recall: {recall[i]:.6f}, F-Measure: {f_measure[i]:.6f}, Specificity: {specificity[i]:.6f}")

# Display overall metrics
print("\n=== Overall Metrics ===")
print(f"Overall Precision: {overall_precision:.2f}")
print(f"Overall Recall: {overall_recall:.2f}")
print(f"Overall F-Measure: {overall_f_measure:.2f}")
print(f"Overall Specificity: {overall_specificity:.2f}")
print(f"Overall Sensitivity: {overall_sensitivity:.2f}")

# Plot confusion matrix with improved formatting
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels,
            linewidths=1, linecolor='black', cbar=False)
plt.title("Confusion Matrix", fontsize=16, fontweight='bold')
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.savefig("../Sample Dataset/ConfusionMatrixAll.png")  # Save the confusion matrix plot
plt.close()  # Close the plot to avoid displaying it in some environment

# Prepare data for visualization of matching values
match_counts = valid_rows['ModePred'] == valid_rows['Alteration']
match_counts = match_counts.value_counts().reset_index()
match_counts.columns = ['Match', 'Count']

# Plot: Bar chart for matching values
plt.figure(figsize=(8, 5))
sns.barplot(data=match_counts, x='Match', y='Count', hue='Match', palette='pastel', legend=False)

plt.title("Count of Matches Between ModePred and Alteration", fontsize=16, fontweight='bold')
plt.xlabel("Match (True/False)", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(ticks=[0, 1], labels=['No Match', 'Match'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig("../Sample Dataset/MatchCountsPlotAll.png")  # Save plot as PNG
plt.close()  # Close the plot to avoid displaying it in some environments






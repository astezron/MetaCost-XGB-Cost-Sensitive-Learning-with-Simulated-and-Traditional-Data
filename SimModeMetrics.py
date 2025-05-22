import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, cohen_kappa_score

# Set the path for your CSV file
file_path = r"F:\Final_OUT\Simulations\MetaCost_PredictionsMode.csv" # Update with your file path

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
accuracy = accuracy_score(y_true, y_pred) * 100  # Accuracy in percentage
kappa = cohen_kappa_score(y_true, y_pred)

# Print accuracy and Kappa
print(f"Accuracy of ModePred matching with Alteration: {accuracy:.2f}%")
print(f"Cohen's Kappa: {kappa:.2f}")

# Plot confusion matrix
plt.figure(figsize=(10, 7))
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_true.unique())
cmd.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(r"F:\Final_OUT\Simulations\ConfusionMatrixMeta.png")  # Save the confusion matrix plot
plt.close()  # Close the plot to avoid displaying it in some environment

# Prepare data for visualization of matching values
match_counts = valid_rows['ModePred'] == valid_rows['Alteration']
match_counts = match_counts.value_counts().reset_index()
match_counts.columns = ['Match', 'Count']

# Plot: Bar chart for matching values (updated to resolve FutureWarning)
plt.figure(figsize=(8, 5))
sns.barplot(data=match_counts, x='Match', y='Count', hue='Match', palette='pastel', legend=False)

plt.title("Count of Matches Between ModePred and Alteration", fontsize=16, fontweight='bold')
plt.xlabel("Match (True/False)", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(ticks=[0, 1], labels=['No Match', 'Match'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig(r"F:\Final_OUT\Simulations\MatchCountsPlotMeta.png")  # Save plot as PNG
plt.close()  # Close the plot to avoid displaying it in some environments


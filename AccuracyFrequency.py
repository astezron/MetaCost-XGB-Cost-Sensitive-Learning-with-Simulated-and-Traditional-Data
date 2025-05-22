import pandas as pd
from collections import Counter
import numpy as np

# Load CSV file
df= pd.read_csv("D:\Final_Alteration\MetaCost\Simulations\Ass_Min\Misc\MetaCostPredictionsMode.csv")

# Ensure the dataframe has 'pred' and 'Alteration' columns
assert 'pred' in df.columns and 'Alteration' in df.columns, "CSV must contain 'pred' and 'Alteration' columns"

# Group by test sample index, assuming repeated simulations per test sample
n_simulations = 50
n_test_points = len(df) // n_simulations

# Create containers
results = []

for i in range(n_test_points):
    subset = df.iloc[i * n_simulations: (i + 1) * n_simulations]
    pred_counts = Counter(subset['pred'])
    most_common_class, freq = pred_counts.most_common(1)[0]
    frequency = freq / n_simulations
    true_class = subset['Alteration'].iloc[0]
    is_correct = (most_common_class == true_class)

    # Assign frequency class
    if frequency >= 0.9:
        freq_class = '90-100%'
    elif frequency >= 0.7:
        freq_class = '70-90%'
    elif frequency >= 0.5:
        freq_class = '50-70%'
    else:
        freq_class = '<50%'

    results.append({
        'most_common_class': most_common_class,
        'true_class': true_class,
        'frequency': frequency,
        'correct': is_correct,
        'frequency_class': freq_class
    })

# Convert results to DataFrame
result_df = pd.DataFrame(results)

# Save result_df to CSV
output_path = "D:/Final_Alteration/MetaCost/Simulations/Ass_Min/Misc/MetaCost_Frequency_Analysis.csv"
result_df.to_csv(output_path, index=False)
print(f"Saved output to {output_path}")

# Optional: Compute average frequency and accuracy per frequency class
for freq_class in result_df['frequency_class'].unique():
    bin_df = result_df[result_df['frequency_class'] == freq_class]
    avg_freq = bin_df['frequency'].mean()
    avg_acc = bin_df['correct'].mean()
    print(f"{freq_class}:")
    print(f"  Average Frequency: {avg_freq:.2%}")
    print(f"  Average Accuracy: {avg_acc:.2%}\n")

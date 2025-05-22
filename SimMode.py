import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the path for your CSV file
file_path = r"F:\Final_OUT\Simulations\MetaCost_Predictions.csv"
# Read the CSV file
data = pd.read_csv(file_path)

# Function to calculate the mode, ignoring NaN values
def get_mode(series):
    return series.mode().iloc[0] if not series.mode().empty else np.nan

# Create a new column 'ModePred' initialized as an empty string
data['ModePred'] = ""

# Loop through the data in chunks of 50 rows
for i in range(0, len(data), 50):
    # Get the current chunk of 50 rows
    chunk = data.iloc[i:i + 50]
    
    # Calculate the mode of the 'Pred' column for the current chunk
    mode_value = get_mode(chunk['Pred'])
    
    # Assign this mode value to the first row of the corresponding chunk only
    data.loc[i, 'ModePred'] = mode_value

# Save the updated DataFrame back to a new CSV file
output_file_path = r"F:\Final_OUT\Simulations\MetaCost_PredictionsMode.csv" # Path for the updated file
data.to_csv(output_file_path, index=False)

# Prepare data for plotting
data_summary = data.groupby(data.index // 50)['ModePred'].first().reset_index()
data_summary['group'] = data_summary['index'] + 1  # Create a group column for plotting

# Filter out rows where ModePred is an empty string for the plot
data_summary = data_summary[data_summary['ModePred'] != ""]

# Plot: Line plot of Pred category modes across subsets
plt.figure(figsize=(10, 6))
sns.lineplot(data=data_summary, x='group', y='ModePred', marker='o')

# Set the title and labels
plt.title("Line Plot of Mode of Pred Category for Each 50-row Subset", fontsize=16, fontweight='bold')
plt.xlabel("Subset of 50 Rows (Grouped)", fontsize=14)
plt.ylabel("Pred Category Mode", fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig(r"F:\Final_OUT\Simulations\MetaCost_PredictionsMode.jpg")  # Save plot as PNG
plt.close()  # Close the plot to avoid displaying it in some environments

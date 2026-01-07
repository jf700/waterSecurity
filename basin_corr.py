import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# filepath: c:\Users\ethan\Documents\josh_code\visualize_correlation.py

# Load the correlation matrix CSV
file_path = "basin_correlation_matrix.csv"  # Update the path if needed
correlation_matrix = pd.read_csv(file_path, index_col=0)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,  # Display the correlation values
    fmt=".2f",   # Format the numbers to 2 decimal places
    cmap="coolwarm",  # Color scheme
    vmin=-1, vmax=1,  # Correlation values range from -1 to 1
    linewidths=0.5,  # Add gridlines between cells
    cbar_kws={"label": "Correlation Coefficient"}  # Label for the colorbar
)

# Add title and labels
plt.title("Basin Correlation Matrix", fontsize=16, pad=20)
plt.xticks(rotation=45, ha="right", fontsize=10)  # Rotate x-axis labels
plt.yticks(fontsize=10)  # Adjust y-axis label size
plt.tight_layout()

# Show the plot
plt.show()
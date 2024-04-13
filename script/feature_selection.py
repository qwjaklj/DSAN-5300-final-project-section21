import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = '../data/data_cleaned.csv'
data = pd.read_csv(file_path)

# Fill missing values for numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Calculate correlation matrix for numeric columns only
correlations = data[numeric_columns].corr()

# Plotting and saving the correlation matrix
plt.figure(figsize=(30, 30))
sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig('../image/correlation_matrix.png')
plt.close()
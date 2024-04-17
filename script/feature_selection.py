import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
file_path = '../data/data_cleaned.csv'
data = pd.read_csv(file_path)

numeric_data = data.select_dtypes(include=[np.number])

# Handle missing values for numeric columns
numeric_data = numeric_data.fillna(numeric_data.mean())

# Calculate the correlation matrix for numeric data
correlation_with_target = numeric_data.corr()['GAD_T'].abs().sort_values(ascending=False)

# Select the top 10 features most correlated with 'GAD_T', excluding the target itself and 'GAD' items
top_10_features = correlation_with_target[8:18]

plt.figure(figsize=(10, 6),dpi=300)
ax = plt.subplot()
ax.set_facecolor('#f3efee')
sns.barplot(x=top_10_features.values, y=top_10_features.index)
plt.title('Top 10 Features Most Correlated with GAD_T')
plt.xlabel('Absolute Correlation Coefficient')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('../image/Feature_selection.png',dpi=300)
plt.show()

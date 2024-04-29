import pandas as pd
import statsmodels.api as sm
data = pd.read_csv('../data/data_cleaned.csv',encoding='ISO-8859-1')
data.head()
age = pd.read_csv('../data/age_final.csv')
age

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.set_facecolor('#f3efee')

sns.histplot(data=data, x="GAD_T", bins=20, kde=True, ax=axes[0], color="skyblue")
axes[0].set_title("Distribution of Total GAD Scores")
axes[0].set_facecolor('#f3efee') 

sns.histplot(data=data, x="SWL_T", bins=20, kde=True, ax=axes[1], color="lightgreen")
axes[1].set_title("Distribution of Total SWL Scores")
axes[1].set_facecolor('#f3efee') 

sns.histplot(data=data, x="SPIN_T", bins=20, kde=True, ax=axes[2], color="salmon")
axes[2].set_title("Distribution of Total SPIN Scores")
axes[2].set_facecolor('#f3efee') 

plt.tight_layout()
plt.savefig('../image/distribution_plot.jpg', facecolor=fig.get_facecolor())
plt.show()

data = data.dropna(subset=['Hours'])
X_GAD_T = sm.add_constant(data['GAD_T'])
poisson_model_gad = sm.GLM(data['Hours'], X_GAD_T, family=sm.families.Poisson()).fit()

data['predicted_hours'] = poisson_model_gad.predict(X_GAD_T)
hours_95th_percentile = data['Hours'].quantile(0.95)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='GAD_T', y='Hours', data=data, alpha=0.5, label='Actual Data')
sns.lineplot(x='GAD_T', y='predicted_hours', data=data, color='red', label='Poisson Prediction')
plt.title('Poisson Model Prediction vs Actual Hours Played by GAD_T Scores')
plt.xlabel('GAD_T Scores')
plt.ylabel('Hours Played')
plt.ylim(0, hours_95th_percentile + 10)  # Adding a small buffer above the 95th percentile
plt.legend()
plt.grid(True)
plt.savefig('../image/relationship_plot.jpg', facecolor=fig.get_facecolor())
plt.show()


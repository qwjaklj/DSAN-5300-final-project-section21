import pandas as pd
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




plt.figure(figsize=(10, 6))
ax = plt.subplot()
ax.set_facecolor('#f3efee')
sns.regplot(data=data, x="Hours", y="GAD_T", scatter_kws={'alpha':0.5, 'edgecolor':'none', 'color':'blue'}, line_kws={'color':'red'})
plt.title("Relationship between Time Spent on Gaming and Anxiety with Fitted Line")
plt.xlabel("Hours Spent on Gaming per Week")
plt.ylabel("Total GAD Scores")
plt.xlim(-1, (data['Hours'].quantile(0.95)+1))
plt.ylim((data['GAD_T'].min()-1), (data['GAD_T'].max()+2)) 

plt.savefig('../image/relationship_plot.jpg', facecolor=fig.get_facecolor())
plt.show()


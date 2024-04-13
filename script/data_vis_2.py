import pandas as pd
data = pd.read_csv('../data/data_cleaned.csv',encoding='ISO-8859-1')
data.head()

age = pd.read_csv('../data/age_final.csv')
age

#Data Visualization 2

import matplotlib.pyplot as plt
import seaborn as sns

# EDA
columns_important_for_analysis = ['GADE', 'Hours', 'SPIN_T'] + [f'SPIN{i}' for i in range(1, 18)]
data = data.dropna(subset=columns_important_for_analysis)

descriptive_stats = data.describe()

descriptive_stats

unique_games = data['Game'].unique()

unique_games, len(unique_games)

game_categories = {game: i for i, game in enumerate(unique_games)}

data['Game_Category'] = data['Game'].map(game_categories)

game_categories, data[['Game', 'Game_Category']].head()

scores_by_game = data.groupby('Game_Category')[['GAD_T', 'SWL_T', 'SPIN_T']].mean()

scores_by_game['Game'] = scores_by_game.index.map({v: k for k, v in game_categories.items()})

scores_by_game = scores_by_game[['Game', 'GAD_T', 'SWL_T', 'SPIN_T']]

scores_by_game.reset_index(drop=True)


# Barplot of average Scores of GAD, SWL, and SPIN by Game
scores_visualization = scores_by_game.melt(id_vars="Game", var_name="Metric", value_name="Score")

plt.figure(figsize=(14, 8))
sns.barplot(x="Score", y="Game", hue="Metric", data=scores_visualization)
plt.title("Average Scores of GAD, SWL, and SPIN by Game")
plt.xlabel("Average Score")
plt.ylabel("Game")
plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
#plt.savefig("../image/average_scores_by_game.png")
plt.show()

# barplot for average scores of GAD, SWL, and SPIN by age group
bins = [17, 34, 44, 54]
labels = ['18-34', '35-44', '45-54']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=True)

scores_by_age_group = data.groupby('Age_Group')[['GAD_T', 'SWL_T', 'SPIN_T']].mean().reset_index()

scores_age_group_visualization = scores_by_age_group.melt(id_vars="Age_Group", var_name="Metric", value_name="Score")

plt.figure(figsize=(14, 8))
sns.barplot(x="Age_Group", y="Score", hue="Metric", data=scores_age_group_visualization, palette="viridis")
plt.title("Average Scores of GAD, SWL, and SPIN by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Average Score")
plt.legend(title="Metric")

plt.tight_layout()
#plt.savefig("../image/average_scores_by_age_group.png")
plt.show()


# pie chart for porportion of gamers by age group
plt.figure(figsize=(8, 8))
plt.pie(age['Proportion'], labels=age['Age Group'], autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Gamers by Age Group')
plt.axis('equal')

#plt.savefig("../image/porportions_by_age_group.png")
plt.show()


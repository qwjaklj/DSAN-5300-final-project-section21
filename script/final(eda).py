import pandas as pd
import numpy as np
data = pd.read_csv('../data/data_cleaned.csv',encoding='ISO-8859-1')
data.head()

age = pd.read_csv('../data/age_final.csv')
age

columns_important_for_analysis = ['GADE', 'Hours', 'SPIN_T'] + [f'SPIN{i}' for i in range(1, 18)]
data = data.dropna(subset=columns_important_for_analysis)

descriptive_stats = data.describe()

descriptive_stats
descriptive_stats.to_csv('../result(eda)/descriptive_stats.csv', index=False)

unique_games = data['Game'].unique()

unique_games, len(unique_games)

game_categories = {game: i for i, game in enumerate(unique_games)}

data['Game_Category'] = data['Game'].map(game_categories)

game_categories, data[['Game', 'Game_Category']].head()

scores_by_game = data.groupby('Game_Category')[['GAD_T', 'SWL_T', 'SPIN_T']].mean()

scores_by_game['Game'] = scores_by_game.index.map({v: k for k, v in game_categories.items()})

scores_by_game = scores_by_game[['Game', 'GAD_T', 'SWL_T', 'SPIN_T']]

scores_by_game.reset_index(drop=True)
scores_by_game.to_csv('../result(eda)/scores_by_game.csv', index=False)
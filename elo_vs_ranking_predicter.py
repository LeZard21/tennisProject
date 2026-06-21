import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Loading data...")
df = pd.read_csv("atp_meccsek_elo_pontokkal_sullyozva.csv")
df['tourney_date'] = pd.to_datetime(df['tourney_date'])

df['winner_rank_points'] = df['winner_rank_points'].fillna(0)
df['loser_rank_points'] = df['loser_rank_points'].fillna(0)

np.random.seed(42)
coin_flip = np.random.rand(len(df)) > 0.5

df['target'] = coin_flip.astype(int)

df['p1_rank_points'] = np.where(coin_flip, df['winner_rank_points'], df['loser_rank_points'])
df['p2_rank_points'] = np.where(coin_flip, df['loser_rank_points'], df['winner_rank_points'])

df['p1_elo_overall'] = np.where(coin_flip, df['elo_w_overall'], df['elo_l_overall'])
df['p2_elo_overall'] = np.where(coin_flip, df['elo_l_overall'], df['elo_w_overall'])

df['diff_rank'] = df['p1_rank_points'] - df['p2_rank_points']
df['diff_overall'] = df['p1_elo_overall'] - df['p2_elo_overall']

split_date = pd.to_datetime('2024-01-01')
train_df = df[df['tourney_date'] < split_date]
test_df = df[df['tourney_date'] >= split_date]

model_atp = LogisticRegression()
model_atp.fit(train_df[['diff_rank']], train_df['target'])
pred_atp = model_atp.predict(test_df[['diff_rank']])
acc_atp = accuracy_score(test_df['target'], pred_atp)

model_elo = LogisticRegression()
model_elo.fit(train_df[['diff_overall']], train_df['target'])
pred_elo = model_elo.predict(test_df[['diff_overall']])
acc_elo = accuracy_score(test_df['target'], pred_elo)

print("="*60)
print(f"Official ATP Points Accuracy (Baseline 0): {acc_atp * 100:.2f}%")
print(f"Custom Overall Elo Accuracy:               {acc_elo * 100:.2f}%")
print("="*60)
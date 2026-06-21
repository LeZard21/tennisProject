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

df['diff_rank'] = np.where(coin_flip, df['winner_rank_points'], df['loser_rank_points']) - \
                  np.where(coin_flip, df['loser_rank_points'], df['winner_rank_points'])

df['diff_overall'] = np.where(coin_flip, df['elo_w_overall'], df['elo_l_overall']) - \
                     np.where(coin_flip, df['elo_l_overall'], df['elo_w_overall'])

df['diff_surface'] = np.where(coin_flip, df['elo_w_surface'], df['elo_l_surface']) - \
                     np.where(coin_flip, df['elo_l_surface'], df['elo_w_surface'])

split_date = pd.to_datetime('2024-01-01')
train_df = df[df['tourney_date'] < split_date]
test_df = df[df['tourney_date'] >= split_date]

features_super = ['diff_rank', 'diff_overall', 'diff_surface']

model_super = LogisticRegression()
model_super.fit(train_df[features_super], train_df['target'])
pred_super = model_super.predict(test_df[features_super])
acc_super = accuracy_score(test_df['target'], pred_super)

print("="*60)
print(f"Previous best (Official ATP only):      64.25%")
print(f"NEW HYBRID SUPERMODEL ACCURACY:         {acc_super * 100:.2f}%")
print("="*60)
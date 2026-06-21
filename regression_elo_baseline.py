import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Loading and preparing data...")
df = pd.read_csv("atp_meccsek_elo_pontokkal_sullyozva.csv")
df['tourney_date'] = pd.to_datetime(df['tourney_date'])
df = df.dropna(subset=['surface'])

np.random.seed(42)
coin_flip = np.random.rand(len(df)) > 0.5

df['p1_elo_overall'] = np.where(coin_flip, df['elo_w_overall'], df['elo_l_overall'])
df['p2_elo_overall'] = np.where(coin_flip, df['elo_l_overall'], df['elo_w_overall'])
df['p1_elo_surface'] = np.where(coin_flip, df['elo_w_surface'], df['elo_l_surface'])
df['p2_elo_surface'] = np.where(coin_flip, df['elo_l_surface'], df['elo_w_surface'])

df['target'] = coin_flip.astype(int)
df['diff_overall'] = df['p1_elo_overall'] - df['p2_elo_overall']
df['diff_surface'] = df['p1_elo_surface'] - df['p2_elo_surface']

split_date = pd.to_datetime('2024-01-01')
train_df = df[df['tourney_date'] < split_date]
test_df = df[df['tourney_date'] >= split_date]

print(f"Training matches: {len(train_df)} | Testing matches: {len(test_df)}\n")

model_v0 = LogisticRegression()
model_v0.fit(train_df[['diff_overall']], train_df['target'])
pred_v0 = model_v0.predict(test_df[['diff_overall']])
acc_v0 = accuracy_score(test_df['target'], pred_v0)

model_v1 = LogisticRegression()
model_v1.fit(train_df[['diff_overall', 'diff_surface']], train_df['target'])
pred_v1 = model_v1.predict(test_df[['diff_overall', 'diff_surface']])
acc_v1 = accuracy_score(test_df['target'], pred_v1)

surfaces = ['Hard', 'Clay', 'Grass']
total_correct = 0

for surf in surfaces:
    train_s = train_df[train_df['surface'] == surf]
    test_s = test_df[test_df['surface'] == surf]
    
    if len(train_s) > 0 and len(test_s) > 0:
        model_s = LogisticRegression()
        model_s.fit(train_s[['diff_overall', 'diff_surface']], train_s['target'])
        pred_s = model_s.predict(test_s[['diff_overall', 'diff_surface']])
        total_correct += sum(pred_s == test_s['target'])

acc_v2 = total_correct / len(test_df)

print("="*60)
print(f"V0 (Overall Elo Only):                    {acc_v0 * 100:.2f}%")
print(f"V1 (Universal Model, Dual Elo):           {acc_v1 * 100:.2f}%")
print(f"V2 (Expert Logic: Surface-split Models):  {acc_v2 * 100:.2f}%")
print("="*60)
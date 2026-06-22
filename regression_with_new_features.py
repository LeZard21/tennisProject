import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

print("Loading and preparing data...")
df = pd.read_csv("features_atp_matches.csv")
df['tourney_date'] = pd.to_datetime(df['tourney_date'])
df = df.dropna(subset=['surface'])

# Define feature dictionary mappings
features_dict = {
    'diff_rank_points': ('winner_rank_points', 'loser_rank_points'), 
    'diff_h2h': ('w_h2h', 'l_h2h'),
    'diff_streak': ('w_streak', 'l_streak'),
    'diff_fatigue': ('w_fatigue', 'l_fatigue'),
    'diff_rustiness': ('w_rustiness', 'l_rustiness'),
    'diff_home': ('w_home', 'l_home'),
    'diff_clutch': ('w_clutch', 'l_clutch')
}

# Drop rows with missing values in any required base metrics to prevent model failure
required_cols = []
for w_col, l_col in features_dict.values():
    required_cols.extend([w_col, l_col])
df = df.dropna(subset=required_cols).reset_index(drop=True)

# Generate randomized Player 1 vs Player 2 layout to eliminate constant target bias
np.random.seed(42)
coin_flip = np.random.rand(len(df)) > 0.5
df['target'] = coin_flip.astype(int)

# Calculate randomized difference variables using the dictionary layout
for diff_name, (w_col, l_col) in features_dict.items():
    df[diff_name] = np.where(coin_flip, df[w_col] - df[l_col], df[l_col] - df[w_col])

feature_cols = list(features_dict.keys())

# Chronological split to prevent data leaking
split_date = pd.to_datetime('2024-01-01')
train_df = df[df['tourney_date'] < split_date]
test_df = df[df['tourney_date'] >= split_date]

print(f"Training matches: {len(train_df)} | Testing matches: {len(test_df)}\n")

# Scale features (essential for regularized models combining large points variables with binary variables)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df[feature_cols])
X_test_scaled = scaler.transform(test_df[feature_cols])

# Re-wrap scaled arrays into DataFrames for clean surface slicing
train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=train_df.index)
train_scaled_df['surface'] = train_df['surface']
train_scaled_df['target'] = train_df['target']

test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=test_df.index)
test_scaled_df['surface'] = test_df['surface']
test_scaled_df['target'] = test_df['target']

# --- V0: Baseline Model (Rank Points Only) ---
model_v0 = LogisticRegression(max_iter=1000)
model_v0.fit(train_scaled_df[['diff_rank_points']], train_scaled_df['target'])
pred_v0 = model_v0.predict(test_scaled_df[['diff_rank_points']])
acc_v0 = accuracy_score(test_scaled_df['target'], pred_v0)

# --- V1: Universal Model (All Engineered Difference Features) ---
model_v1 = LogisticRegression(max_iter=1000)
model_v1.fit(train_scaled_df[feature_cols], train_scaled_df['target'])
pred_v1 = model_v1.predict(test_scaled_df[feature_cols])
acc_v1 = accuracy_score(test_scaled_df['target'], pred_v1)

# --- V2: Expert Logic Model (Surface-Split Slices) ---
surfaces = ['Hard', 'Clay', 'Grass']
total_correct = 0

for surf in surfaces:
    train_s = train_scaled_df[train_scaled_df['surface'] == surf]
    test_s = test_scaled_df[test_scaled_df['surface'] == surf]
    
    if len(train_s) > 0 and len(test_s) > 0:
        model_s = LogisticRegression(max_iter=1000)
        model_s.fit(train_s[feature_cols], train_s['target'])
        pred_s = model_s.predict(test_s[feature_cols])
        total_correct += sum(pred_s == test_s['target'])

acc_v2 = total_correct / len(test_scaled_df)

print("="*60)
print(f"V0 (Rank Points Only):                     {acc_v0 * 100:.2f}%")
print(f"V1 (Universal Model, All Features):        {acc_v1 * 100:.2f}%")
print(f"V2 (Expert Logic: Surface-split Models):  {acc_v2 * 100:.2f}%")
print("="*60)
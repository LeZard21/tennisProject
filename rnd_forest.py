import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load ultimate combined burn-in database
print("🌲 Loading ultimate tennis database...")
df = pd.read_csv("atp_matches_combined_1990-2000_burn_in.csv")
df['tourney_date'] = pd.to_datetime(df['tourney_date'])

# Biztonsági lépés: ha egy játékosnak nincs ATP pontja, kapjon 0-t
df['winner_rank_points'] = pd.to_numeric(df['winner_rank_points'], errors='coerce').fillna(0)
df['loser_rank_points'] = pd.to_numeric(df['loser_rank_points'], errors='coerce').fillna(0)

# 2. Anonymization (Coin Flip)
np.random.seed(42)
coin_flip = np.random.rand(len(df)) > 0.5
df['target'] = coin_flip.astype(int)

# 3. Anonymous Feature Calculation (P1 - P2)
features_dict = {
    'diff_elo_overall': ('elo_w_overall', 'elo_l_overall'),
    'diff_elo_surface': ('elo_w_surface', 'elo_l_surface'),
    'diff_rank_points': ('winner_rank_points', 'loser_rank_points'),
    'diff_fatigue': ('w_fatigue', 'l_fatigue'),
    'diff_h2h': ('w_h2h', 'l_h2h'),
    'diff_streak': ('w_streak', 'l_streak'),
    'diff_rustiness': ('w_rustiness', 'l_rustiness'),
    'diff_home': ('w_home', 'l_home'),
    'diff_clutch': ('w_clutch', 'l_clutch')
}

for new_col, (w_col, l_col) in features_dict.items():
    df[new_col] = np.where(coin_flip, df[w_col], df[l_col]) - \
                  np.where(coin_flip, df[l_col], df[w_col])

features = list(features_dict.keys())
df = df.dropna(subset=features)

# 4. Chronological Train/Test Split
split_date = pd.to_datetime('2024-01-01')
train_df = df[df['tourney_date'] < split_date]
test_df = df[df['tourney_date'] >= split_date]

X_train, y_train = train_df[features], train_df['target']
X_test, y_test = test_df[features], test_df['target']

# 5. Training the Random Forest (200 trees)
print("Forest training initialized (200 trees, utilizing all CPU cores)...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 6. Evaluation
train_pred = rf_model.predict(X_train)
test_pred = rf_model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("\n" + "="*60)
print(f"RANDOM FOREST TRAIN ACCURACY: {train_acc * 100:.2f}%")
print(f"RANDOM FOREST TEST ACCURACY:  {test_acc * 100:.2f}%")
print("="*60 + "\n")

# Extra: Feature Importances 
print("Global Feature Importances across the forest:")
for name, importance in sorted(zip(features, rf_model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"{name}: {importance:.4f}")
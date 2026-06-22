import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

print("🚀 Loading data and standardizing formatting...")

# 1. Load Data
df = pd.read_csv('atp_matches_combined_1990-2000_burn_in.csv')
df['tourney_date'] = pd.to_datetime(df['tourney_date'])

df['winner_rank_points'] = pd.to_numeric(df['winner_rank_points'], errors='coerce').fillna(0)
df['loser_rank_points'] = pd.to_numeric(df['loser_rank_points'], errors='coerce').fillna(0)

# 2. Anonymization (Coin flip method)
np.random.seed(42)
coin_flip = np.random.rand(len(df)) > 0.5
df['target'] = coin_flip.astype(int)

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

# Calculate feature differences
for new_col, (w_col, l_col) in features_dict.items():
    df[new_col] = np.where(coin_flip, df[w_col], df[l_col]) - \
                  np.where(coin_flip, df[l_col], df[w_col])

features = list(features_dict.keys())
df = df.dropna(subset=features)
df = df.sort_values('tourney_date') # Required for TimeSeriesSplit

# 3. Strict Chronological Train/Test Split (2024 Cutoff)
split_date = pd.to_datetime('2024-01-01')
train_df = df[df['tourney_date'] < split_date]
test_df = df[df['tourney_date'] >= split_date]

X_train, y_train = train_df[features], train_df['target']
X_test, y_test = test_df[features], test_df['target']

# 4. Grid Search Setup with TimeSeries Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    'max_depth': [3, 5, 7, 9, 11],
    'learning_rate': [0.05, 0.1, 0.2, 0.3],
    'n_estimators': [100, 200, 300, 400, 500],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'reg_lambda': [1.0, 5.0]
}

xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)

print("\n Running XGBoost Grid Search (This might take a while)...")
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=tscv, 
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# 5. Results & Evaluation
print("\n🏆 --- Best XGBoost Parameters ---")
print(grid_search.best_params_)

best_xgb = grid_search.best_estimator_

y_pred_test = best_xgb.predict(X_test)
y_pred_train = best_xgb.predict(X_train)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)

print("\n" + "="*60)
print(f"XGBOOST TRAIN ACCURACY: {acc_train * 100:.2f}%")
print(f"XGBOOST TEST ACCURACY:  {acc_test * 100:.2f}%")
print("="*60 + "\n")

print("Classification Report (2024 Test Set):")
print(classification_report(y_test, y_pred_test))
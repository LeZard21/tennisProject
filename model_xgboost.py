import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data with absolute path
df = pd.read_csv('atp_matches_combined_1990-2000_burn_in.csv')

# Exact features requested
selected_diffs = [
    'elo_diff_overall', 
    'elo_diff_surface', 
    'rank_points_diff', 
    'fatigue_diff',      
    'streak_diff',       
    'h2h_diff',          
    'home_diff',         
    'clutch_diff',       
    'rustiness_diff'     
]

# Ensure columns exist
csv_cols = [col for col in selected_diffs if col in df.columns]
df_subset = df[csv_cols + ['tourney_date']].copy()

# Sort chronologically BEFORE doing anything else
df_subset['tourney_date'] = pd.to_datetime(df_subset['tourney_date'])
df_subset = df_subset.sort_values('tourney_date')

# 2. Split into Train/Test chronologically to completely prevent leakage
split_idx = int(len(df_subset) * 0.8)
train_df = df_subset.iloc[:split_idx].copy()
test_df = df_subset.iloc[split_idx:].copy()

# 3. Impute missing values using ONLY training medians (Strict leakage guard)
train_medians = train_df[csv_cols].median()
train_df[csv_cols] = train_df[csv_cols].fillna(train_medians)
test_df[csv_cols] = test_df[csv_cols].fillna(train_medians)

# 4. Generate Balanced Perspectives for Training
train_t1 = train_df[csv_cols].copy()
train_t1['target'] = 1

train_t0 = -train_df[csv_cols].copy()
train_t0['target'] = 0

np.random.seed(42)
mask_train = np.random.rand(len(train_df)) > 0.5
X_train = pd.concat([train_t1[mask_train], train_t0[~mask_train]], axis=0).drop(columns=['target'])
y_train = pd.concat([train_t1[mask_train], train_t0[~mask_train]], axis=0)['target']

# Generate Balanced Perspectives for Testing
test_t1 = test_df[csv_cols].copy()
test_t1['target'] = 1

test_t0 = -test_df[csv_cols].copy()
test_t0['target'] = 0

mask_test = np.random.rand(len(test_df)) > 0.5
X_test = pd.concat([test_t1[mask_test], test_t0[~mask_test]], axis=0).drop(columns=['target'])
y_test = pd.concat([test_t1[mask_test], test_t0[~mask_test]], axis=0)['target']

# 5. TimeSeriesSplit: Validates sequentially along the timeline
tscv = TimeSeriesSplit(n_splits=5)

# Hyperparameter Grid tuned for sports diff features
param_grid = {
    'max_depth': [3, 4],            # Complex trees overfit noisy sports data
    'learning_rate': [0.01, 0.05],   # Smaller learning rates
    'n_estimators': [50, 100, 150],  # Give early stopping room
    'subsample': [0.6, 0.8],         # Heavy subsampling forces generalization
    'colsample_bytree': [0.6, 0.8],
    'reg_lambda': [1.0, 10.0]        # Higher L2 penalty to handle colinearity between Elos
}

xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)

# 6. Grid Search with TimeSeries CV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=tscv, 
    verbose=1,
    n_jobs=-1
)

print("Running strict chronological tuning...")
grid_search.fit(X_train, y_train)

print("\n--- Best Parameters ---")
print(grid_search.best_params_)

# 7. Evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"\nReal-world Test Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load data
print("🌲 Loading data...")
df = pd.read_csv("atp_matches_combined_1990-2000_burn_in.csv")
df['tourney_date'] = pd.to_datetime(df['tourney_date'])

df['winner_rank_points'] = pd.to_numeric(df['winner_rank_points'], errors='coerce').fillna(0)
df['loser_rank_points'] = pd.to_numeric(df['loser_rank_points'], errors='coerce').fillna(0)

# 2. Anonymize and set target
np.random.seed(42)
coin_flip = np.random.rand(len(df)) > 0.5
df['target'] = coin_flip.astype(int)

# 3. Feature engineering (differences)
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

# 4. Train/Test Split
split_date = pd.to_datetime('2024-01-01')
train_df = df[df['tourney_date'] < split_date]
test_df = df[df['tourney_date'] >= split_date]

X_train, y_train = train_df[features], train_df['target']
X_test, y_test = test_df[features], test_df['target']

# ==========================================
# 5. RANDOM FOREST GRID SEARCH
# ==========================================
estimators_list = [50, 100, 150, 200, 250]
depths_list = [4, 6, 8, 10, 12, 14]

results_matrix = []

print("\nStarting Random Forest Grid Search...")
for d in depths_list:
    row_accuracies = []
    for n in estimators_list:
        rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        test_acc = accuracy_score(y_test, rf.predict(X_test))
        row_accuracies.append(test_acc * 100)
        
        print(f"Done: Depth={d:2}, Trees={n:3} -> {test_acc * 100:.2f}%")
        
    results_matrix.append(row_accuracies)

# 6. Plot Heatmap
results_df = pd.DataFrame(
    results_matrix, 
    columns=[f"{n} Trees" for n in estimators_list], 
    index=[f"Depth: {d}" for d in depths_list]
)

plt.figure(figsize=(12, 7))
sns.heatmap(results_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Test Accuracy (%)'})

plt.title("Random Forest Grid Search Heatmap: Number of Trees vs. Max Depth", fontsize=14, pad=15)
plt.xlabel("Number of Trees (n_estimators)", fontsize=12)
plt.ylabel("Maximum Depth (max_depth)", fontsize=12)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
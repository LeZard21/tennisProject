import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#  Data Loading
print("🌲 Loading data...")
df = pd.read_csv("atp_matches_combined_1990-2000_burn_in.csv")
df['tourney_date'] = pd.to_datetime(df['tourney_date'])

# Biztonsági lépés: ha egy játékosnak nincs ATP pontja (pl. ranglista nélküli szabadkártyás), kapjon 0-t
df['winner_rank_points'] = pd.to_numeric(df['winner_rank_points'], errors='coerce').fillna(0)
df['loser_rank_points'] = pd.to_numeric(df['loser_rank_points'], errors='coerce').fillna(0)

#  Anonymization (Coin Flip)
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

#  Chronological Train/Test Split
split_date = pd.to_datetime('2024-01-01')
train_df = df[df['tourney_date'] < split_date]
test_df = df[df['tourney_date'] >= split_date]

X_train, y_train = train_df[features], train_df['target']
X_test, y_test = test_df[features], test_df['target']


print("Training trees with different depths (1 to 25)...")
depths = range(1, 26)
train_accuracies = []
test_accuracies = []

for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    tree.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, tree.predict(X_train))
    train_accuracies.append(train_acc)
    
    test_acc = accuracy_score(y_test, tree.predict(X_test))
    test_accuracies.append(test_acc)
    
    print(f"Depth: {d:2} | Train: {train_acc*100:.1f}% | Test: {test_acc*100:.2f}%")

best_test_acc = max(test_accuracies)
best_depth = depths[test_accuracies.index(best_test_acc)]

print("\n" + "="*50)
print(f"BEST DEPTH FOUND: {best_depth} (Test Accuracy: {best_test_acc * 100:.2f}%)")
print("="*50 + "\n")

# ==========================================
# AZ OVERFITTING
# ==========================================
print("Rendering the validation curve (Matplotlib window opening)...")
plt.figure(figsize=(12, 6))

plt.plot(depths, train_accuracies, label='Train Accuracy (Memorizing the Past)', color='blue', marker='o')
plt.plot(depths, test_accuracies, label='Test Accuracy (Predicting the Future)', color='red', marker='o')
plt.axvline(x=best_depth, color='green', linestyle='--', label=f'Ideal Depth ({best_depth})')

plt.title("Decision Tree Overfitting (All Features Included)")
plt.xlabel("Max Depth of Tree")
plt.ylabel("Accuracy")
plt.xticks(depths)
plt.legend()
plt.grid(True)
plt.show()
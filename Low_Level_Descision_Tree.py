import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# 1. Data Loading
print("🌲 Loading data...")
df = pd.read_csv("atp_matches_combined_1990-2000_burn_in.csv")
df['tourney_date'] = pd.to_datetime(df['tourney_date'])

# 2. Anonymization (Coin Flip)

np.random.seed(42)
coin_flip = np.random.rand(len(df)) > 0.5
df['target'] = coin_flip.astype(int) 

# 3. Anonymous Feature Calculation (P1 - P2)
print("Generating anonymous differences from new features...")
features_dict = {
    'diff_elo_overall': ('elo_w_overall', 'elo_l_overall'),
    'diff_elo_surface': ('elo_w_surface', 'elo_l_surface'),
    'diff_h2h': ('w_h2h', 'l_h2h'),
    'diff_streak': ('w_streak', 'l_streak'),
    'diff_fatigue': ('w_fatigue', 'l_fatigue'),
    'diff_rustiness': ('w_rustiness', 'l_rustiness'),
    'diff_home': ('w_home', 'l_home'),
    'diff_clutch': ('w_clutch', 'l_clutch')
}

for new_col, (w_col, l_col) in features_dict.items():
    df[new_col] = np.where(coin_flip, df[w_col], df[l_col]) - \
                  np.where(coin_flip, df[l_col], df[w_col])

features = list(features_dict.keys())

# Drop rows with missing values to avoid math errors
df = df.dropna(subset=features)

# 4. Chronological Train/Test Split

split_date = pd.to_datetime('2024-01-01')
train_df = df[df['tourney_date'] < split_date]
test_df = df[df['tourney_date'] >= split_date]

X_train, y_train = train_df[features], train_df['target']
X_test, y_test = test_df[features], test_df['target']

# 5. Training and Prediction
# max_depth=3
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

acc = accuracy_score(y_test, tree_model.predict(X_test))
print("\n" + "="*60)
print(f"DECISION TREE ACCURACY (max_depth=3): {acc * 100:.2f}%")
print("="*60 + "\n")

# 6. Visualization
print("Rendering the decision tree (Matplotlib window opening)...")
plt.figure(figsize=(20, 10))
plot_tree(tree_model, 
          feature_names=features, 
          class_names=['P2 Wins', 'P1 Wins'], 
          filled=True, 
          rounded=True, 
          fontsize=9)
plt.title("Feature Importance: How the Model Decides (Max Depth: 3)")
plt.show()
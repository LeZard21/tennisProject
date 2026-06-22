import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load ultimate combined burn-in database
print("Loading ultimate tennis database...")
df = pd.read_csv("atp_matches_combined_1990-2000_burn_in.csv")
df['tourney_date'] = pd.to_datetime(df['tourney_date'])

# 2. Anonymization (Coin Flip to avoid P1/P2 bias)
np.random.seed(42)
coin_flip = np.random.rand(len(df)) > 0.5
df['target'] = coin_flip.astype(int)

# 3. Define features in order of importance
# As we go down, the model gets more dimensions (and potentially noisier features)
feature_sets = [
    ['elo_diff_overall'], # 1. Only general Elo
    ['elo_diff_overall', 'elo_diff_surface'], # 2. Elo metrics
    ['elo_diff_overall', 'elo_diff_surface', 'diff_rank_points'], # 3. Elo + Rank points
    ['elo_diff_overall', 'elo_diff_surface', 'diff_rank_points', 'diff_fatigue'], # 4. + Fatigue
    ['elo_diff_overall', 'elo_diff_surface', 'diff_rank_points', 'diff_fatigue', 'diff_h2h'], # 5. + H2H
    ['elo_diff_overall', 'elo_diff_surface', 'diff_rank_points', 'diff_fatigue', 'diff_h2h', 'diff_streak'], # 6. + Streak
    ['elo_diff_overall', 'elo_diff_surface', 'diff_rank_points', 'diff_fatigue', 'diff_h2h', 'diff_streak', 'diff_rustiness'] # 7. + Rustiness
]

# Dictionary mapping dictionary keys to difference columns
full_dict = {
    'elo_diff_overall': ('elo_w_overall', 'elo_l_overall'),
    'elo_diff_surface': ('elo_w_surface', 'elo_l_surface'),
    'diff_rank_points': ('winner_rank_points', 'loser_rank_points'), 
    'diff_fatigue': ('w_fatigue', 'l_fatigue'),
    'diff_h2h': ('w_h2h', 'l_h2h'),
    'diff_streak': ('w_streak', 'l_streak'),
    'diff_rustiness': ('w_rustiness', 'l_rustiness')
}

# Create anonymous difference columns (P1 - P2) in the DataFrame
for col, (w_col, l_col) in full_dict.items():
    df[col] = np.where(coin_flip, df[w_col], df[l_col]) - np.where(coin_flip, df[l_col], df[w_col])

# Hyperparameter grid setup: K values (steps of 15) and Feature levels
k_values = range(15, 201, 15) 
results_matrix = []

print("\nGrid Search loop initialized (K Neighbors vs. Feature Levels)...")
for feats in feature_sets:
    row_accuracies = []
    
    # Drop rows with missing values for the currently analyzed features
    temp_df = df.dropna(subset=feats).copy()
    
    # Chronological Train/Test Split
    split_date = pd.to_datetime('2024-01-01')
    train_df = temp_df[temp_df['tourney_date'] < split_date]
    test_df = temp_df[temp_df['tourney_date'] >= split_date]
    
    X_train, y_train = train_df[feats], train_df['target']
    X_test, y_test = test_df[feats], test_df['target']
    
    # Scaling is mandatory for KNN due to geometric distance measurement
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(X_train_scaled, y_train)
        pred = knn.predict(X_test_scaled)
        acc = accuracy_score(y_test, pred)
        row_accuracies.append(acc * 100) # Save as percentage
        
    results_matrix.append(row_accuracies)

# 4. Plotting the Heatmap
results_df = pd.DataFrame(results_matrix, 
                          columns=[f"K={k}" for k in k_values], 
                          index=[f"{i+1} Feature Level" for i in range(len(feature_sets))])

plt.figure(figsize=(14, 8))
sns.heatmap(results_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Test Accuracy (%)'})
plt.title("KNN Grid Search Heatmap: K Neighbors vs. Number of Features")
plt.xlabel("Hyperparameter: K Value")
plt.ylabel("Input Features (Dimensionality)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
import pandas as pd

# Define file paths
elo_file = 'atp_meccsek_elo_pontokkal_1990_2000_burn_in.csv'
features_file = 'features_atp_matches_1990-2000_burn_in.csv'  
output_file = 'atp_matches_combined_1990-2000_burn_in.csv'

print("Loading datasets...")
df_elo = pd.read_csv(elo_file)
df_features = pd.read_csv(features_file)

# Unique keys to align matches precisely across both datasets
join_keys = ['tourney_id', 'match_num', 'winner_id', 'loser_id']

# Columns unique to the features dataset that need to be appended
feature_cols = [
    'w_h2h', 'l_h2h', 'h2h_diff', 
    'w_streak', 'l_streak', 'streak_diff', 
    'w_fatigue', 'l_fatigue', 'fatigue_diff', 
    'w_rustiness', 'l_rustiness', 'rustiness_diff', 
    'w_home', 'l_home', 'home_diff', 
    'w_clutch', 'l_clutch', 'clutch_diff'
]

# Subset the features dataframe to prevent column duplication
df_features_subset = df_features[join_keys + feature_cols]

print("Merging Elo ratings with custom engineered features...")
# Perform an inner join on the unique match identifiers
df_combined = pd.merge(df_elo, df_features_subset, on=join_keys, how='inner')

print(f"Saving combined dataset to {output_file}...")
df_combined.to_csv(output_file, index=False)

print("Process complete.")
print(f"Total Rows: {df_combined.shape[0]} | Total Columns: {df_combined.shape[1]}")
import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt

data_dir = 'tennis_atp-master'

print("Loading data files (2000-present, excluding Doubles and Futures)...")
# 1. Load only relevant files to avoid memory bloat and noisy data
all_files = glob.glob(os.path.join(data_dir, "atp_matches_[2-9]*.csv")) + \
            glob.glob(os.path.join(data_dir, "atp_matches_qual_[2-9]*.csv")) + \
            glob.glob(os.path.join(data_dir, "atp_matches_chall_[2-9]*.csv"))

df_list = [pd.read_csv(f) for f in all_files]
df = pd.concat(df_list, ignore_index=True)

print("Cleaning and sorting data chronologically...")
# 2. Date conversion and strict chronological sorting (Critical for Elo)
df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
df = df.dropna(subset=['tourney_date'])

# Exclude 'Carpet' surface as it is no longer used in modern tennis
df = df[df['surface'] != 'Carpet'].reset_index(drop=True)

# Sort by date and match number
df = df.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)

# 3. Dual Elo Calculation Setup
overall_ratings = {}
surface_ratings = {
    'Hard': {},
    'Clay': {},
    'Grass': {}
}

def get_rating(player, surface=None):
    # Default Elo rating is 1500
    if surface is None:
        return overall_ratings.get(player, 1500)
    else:
        # Fallback to overall rating if surface is missing or invalid
        if pd.isna(surface) or surface not in surface_ratings:
            return overall_ratings.get(player, 1500)
        return surface_ratings[surface].get(player, 1500)

winner_overall = []
loser_overall = []
winner_surface = []
loser_surface = []

K = 20 # Standard K-factor

print("Calculating Dual Elo ratings (this may take a minute)...")
# 4. Iterate through matches
for index, row in df.iterrows():
    w_name, l_name = row['winner_name'], row['loser_name']
    surface = row['surface']
    
    # Fetch pre-match ratings
    w_elo_all = get_rating(w_name)
    l_elo_all = get_rating(l_name)
    w_elo_surf = get_rating(w_name, surface)
    l_elo_surf = get_rating(l_name, surface)
    
    # Store pre-match ratings for ML features
    winner_overall.append(w_elo_all)
    loser_overall.append(l_elo_all)
    winner_surface.append(w_elo_surf)
    loser_surface.append(l_elo_surf)
    
    # Update Overall Elo
    exp_w_all = 1 / (1 + 10 ** ((l_elo_all - w_elo_all) / 400))
    point_change_all = K * (1 - exp_w_all)
    overall_ratings[w_name] = w_elo_all + point_change_all
    overall_ratings[l_name] = l_elo_all - point_change_all
    
    # Update Surface Elo (if surface is valid)
    if pd.notna(surface) and surface in surface_ratings:
        exp_w_surf = 1 / (1 + 10 ** ((l_elo_surf - w_elo_surf) / 400))
        point_change_surf = K * (1 - exp_w_surf)
        surface_ratings[surface][w_name] = w_elo_surf + point_change_surf
        surface_ratings[surface][l_name] = l_elo_surf - point_change_surf

# 5. Append calculated features back to the dataframe
df['elo_w_overall'] = winner_overall
df['elo_l_overall'] = loser_overall
df['elo_w_surface'] = winner_surface
df['elo_l_surface'] = loser_surface

# Calculate the difference for the visualization
df['elo_diff_overall'] = df['elo_w_overall'] - df['elo_l_overall']

print("Generating visualization...")
# 6. Visualization
plt.figure(figsize=(12, 7))
sns.boxplot(x='surface', y='elo_diff_overall', data=df)
plt.axhline(0, color='red', linestyle='--')
plt.title("Distribution of Overall Elo Advantage by Surface")
plt.ylabel("Overall Elo Difference (Winner - Loser)")
plt.show()
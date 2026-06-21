import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt

data_dir = 'tennis_atp-master'

print("Loading data files (2000-present, excluding Doubles and Futures)...")
all_files = glob.glob(os.path.join(data_dir, "atp_matches_[2-9]*.csv")) + \
            glob.glob(os.path.join(data_dir, "atp_matches_qual_[2-9]*.csv")) + \
            glob.glob(os.path.join(data_dir, "atp_matches_chall_[2-9]*.csv"))

df_list = [pd.read_csv(f) for f in all_files]
df = pd.concat(df_list, ignore_index=True)

print("Cleaning and sorting data chronologically...")
df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
df = df.dropna(subset=['tourney_date'])

df = df[df['surface'] != 'Carpet'].reset_index(drop=True)

df = df.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)

overall_ratings = {}
surface_ratings = {
    'Hard': {},
    'Clay': {},
    'Grass': {}
}

def get_rating(player, surface=None):
    # everyone starts at a baseline of 1500
    if surface is None:
        return overall_ratings.get(player, 1500)
    else:
        # if we don't have a valid surface, just fall back to their overall rating
        if pd.isna(surface) or surface not in surface_ratings:
            return overall_ratings.get(player, 1500)
        return surface_ratings[surface].get(player, 1500)

winner_overall = []
loser_overall = []
winner_surface = []
loser_surface = []

print("Calculating Dual Elo ratings (this may take a minute)...")
for index, row in df.iterrows():
    w_name, l_name = row['winner_name'], row['loser_name']
    surface = row['surface']
    tourney_level = row['tourney_level']
    
    w_elo_all = get_rating(w_name)
    l_elo_all = get_rating(l_name)
    w_elo_surf = get_rating(w_name, surface)
    l_elo_surf = get_rating(l_name, surface)
    
    winner_overall.append(w_elo_all)
    loser_overall.append(l_elo_all)
    winner_surface.append(w_elo_surf)
    loser_surface.append(l_elo_surf)
    
    if tourney_level == 'G':
        K_current = 32
    elif tourney_level == 'M':
        K_current = 24
    elif tourney_level == 'A':
        K_current = 16
    else:
        K_current = 10
    
    exp_w_all = 1 / (1 + 10 ** ((l_elo_all - w_elo_all) / 400))
    point_change_all = K_current * (1 - exp_w_all)
    overall_ratings[w_name] = w_elo_all + point_change_all
    overall_ratings[l_name] = l_elo_all - point_change_all
    
    if pd.notna(surface) and surface in surface_ratings:
        exp_w_surf = 1 / (1 + 10 ** ((l_elo_surf - w_elo_surf) / 400))
        point_change_surf = K_current * (1 - exp_w_surf)
        surface_ratings[surface][w_name] = w_elo_surf + point_change_surf
        surface_ratings[surface][l_name] = l_elo_surf - point_change_surf

df['elo_w_overall'] = winner_overall
df['elo_l_overall'] = loser_overall
df['elo_w_surface'] = winner_surface
df['elo_l_surface'] = loser_surface

df['elo_diff_overall'] = df['elo_w_overall'] - df['elo_l_overall']

print("Generating visualization...")
plt.figure(figsize=(12, 7))
sns.boxplot(x='surface', y='elo_diff_overall', data=df)
plt.axhline(0, color='red', linestyle='--')
plt.title("Distribution of Overall Elo Advantage by Surface")
plt.ylabel("Overall Elo Difference (Winner - Loser)")
plt.show()

df.to_csv("atp_meccsek_elo_pontokkal_sullyozva.csv", index=False)
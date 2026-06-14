import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt

data_dir = 'tennis_atp-master'
all_files = glob.glob(os.path.join(data_dir, "atp_matches_*.csv"))

df_list = [pd.read_csv(f) for f in all_files]
df = pd.concat(df_list, ignore_index=True)
df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')

#Correlation Heatmap
numeric_cols = ['minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'l_ace', 'l_df']
corr = df[numeric_cols].corr()


ratings = {}

def get_rating(player):
    return ratings.get(player, 1500)

winner_elos = []
loser_elos = []

for _, row in df.iterrows():
    w_name, l_name = row['winner_name'], row['loser_name']
    w_elo, l_elo = get_rating(w_name), get_rating(l_name)
    
    winner_elos.append(w_elo)
    loser_elos.append(l_elo)
    
    # Update ratings (Standard K=20)
    expected_w = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
    ratings[w_name] = w_elo + 20 * (1 - expected_w)
    ratings[l_elo] = l_elo + 20 * (0 - expected_w)

df['elo_winner'] = winner_elos
df['elo_loser'] = loser_elos
df['elo_diff'] = df['elo_winner'] - df['elo_loser']

plt.figure(figsize=(12, 7))
sns.boxplot(x='surface', y='elo_diff', data=df)
plt.axhline(0, color='red', linestyle='--')
plt.title("Distribution of Elo Advantage by Surface")
plt.ylabel("Elo Difference (Winner - Loser)")
plt.show()
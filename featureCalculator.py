'''
h2h, winning streak, fatigue, rustiness, home-field advantage, clutch factor heuristic
Head-to-Head (w_h2h, l_h2h): 
    Tracks matchups dynamically using a composite dictionary key tuple (player_a, player_b). 
    It maps how many times the winner has beaten the loser historically prior to this meeting, and vice versa.
Winning Streak (w_streak, l_streak): 
    Increments by $1$ for the winner of each match, and completely resets to $0$ for the loser.
Fatigue (w_fatigue, l_fatigue): 
    Counts how many total matches a player has played in the preceding 14 days. 
    This naturally accounts for deep tournament runs (e.g., playing a 4th match within a week) as well as quick turnaround fatigue from winning a tournament the prior weekend.
Rustiness (w_rustiness, l_rustiness): 
    Measures the difference in days between the current tourney_date and the player's last recorded match timestamp. 
    Useful for identifying performance drops after injury or off-season layout blocks.
Home-field Advantage (w_home, l_home): 
    Directly cross-references the tournament nation code (tourney_ioc) with the respective player nation codes (winner_ioc / loser_ioc). 
    It flags as 1 if they match, and 0 otherwise.
Clutch Factor Heuristic (w_clutch, l_clutch): 
    Computes a dynamic running win percentage exclusively for "high-leverage" close matches. 
    A match is flagged as close if the score contains a tiebreak (indicated by parentheses (, e.g., 7-6(5)) or if the number of sets played meets or exceeds the tournament's best_of value (meaning it went to a deciding 3rd or 5th set). Unmapped players default to a neutral baseline of 0.50.
'''
import pandas as pd
import glob
import os

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

# --- Initialize Feature State Trackers ---
player_last_match = {}       
player_streak = {}           
player_recent_matches = {}   
h2h_records = {}             

# New State Trackers for Clutch Factor
player_clutch_wins = {}      # player_name -> count of close matches won
player_clutch_matches = {}   # player_name -> total close matches played

# Lists to store computed features
w_h2h_list, l_h2h_list = [], []
w_streak_list, l_streak_list = [], []
w_fatigue_list, l_fatigue_list = [], []
w_rust_list, l_rust_list = [], []
w_home_list, l_home_list = [], []
w_clutch_list, l_clutch_list = [], []

print("Engineering historical features (including Home Advantage and Clutch Factor)...")
for index, row in df.iterrows():
    w_name = row['winner_name']
    l_name = row['loser_name']
    match_date = row['tourney_date']
    
    # ----------------------------------------------------
    # 1. STATIC / CURRENT MATCH FEATURES
    # ----------------------------------------------------
    
    # Home-Field Advantage (Match country code matches player country code)
    t_ioc = row.get('tourney_ioc')
    w_ioc = row.get('winner_ioc')
    l_ioc = row.get('loser_ioc')
    
    w_home = 1 if (pd.notna(w_ioc) and pd.notna(t_ioc) and w_ioc == t_ioc) else 0
    l_home = 1 if (pd.notna(l_ioc) and pd.notna(t_ioc) and l_ioc == t_ioc) else 0
    
    # ----------------------------------------------------
    # 2. HISTORICAL METRICS (BEFORE UPDATING STATE)
    # ----------------------------------------------------
    
    # Head-to-Head 
    w_h2h = h2h_records.get((w_name, l_name), 0)
    l_h2h = h2h_records.get((l_name, w_name), 0)
    
    # Winning Streak 
    w_strk = player_streak.get(w_name, 0)
    l_strk = player_streak.get(l_name, 0)
    
    # Rustiness 
    w_last = player_last_match.get(w_name)
    l_last = player_last_match.get(l_name)
    w_rust = (match_date - w_last).days if w_last else 0
    l_rust = (match_date - l_last).days if l_last else 0
    
    # Fatigue (Matches played in last 14 days)
    w_history = player_recent_matches.get(w_name, [])
    l_history = player_recent_matches.get(l_name, [])
    w_fatigue = sum(1 for d in w_history if (match_date - d).days <= 14)
    l_fatigue = sum(1 for d in l_history if (match_date - d).days <= 14)
    
    # Clutch Factor (Historical Win % in close matches; baseline 0.50 if no history)
    w_c_wins = player_clutch_wins.get(w_name, 0)
    w_c_total = player_clutch_matches.get(w_name, 0)
    w_clutch = w_c_wins / w_c_total if w_c_total > 0 else 0.50
    
    l_c_wins = player_clutch_wins.get(l_name, 0)
    l_c_total = player_clutch_matches.get(l_name, 0)
    l_clutch = l_c_wins / l_c_total if l_c_total > 0 else 0.50

    # Append to lists
    w_h2h_list.append(w_h2h)
    l_h2h_list.append(l_h2h)
    w_streak_list.append(w_strk)
    l_streak_list.append(l_strk)
    w_fatigue_list.append(w_fatigue)
    l_fatigue_list.append(l_fatigue)
    w_rust_list.append(w_rust)
    l_rust_list.append(l_rust)
    w_home_list.append(w_home)
    l_home_list.append(l_home)
    w_clutch_list.append(w_clutch)
    l_clutch_list.append(l_clutch)
    
    # ----------------------------------------------------
    # 3. UPDATE STATE TRACKERS WITH CURRENT MATCH RESULTS
    # ----------------------------------------------------
    
    # Update H2H & Streaks
    h2h_records[(w_name, l_name)] = w_h2h + 1
    player_streak[w_name] = w_strk + 1
    player_streak[l_name] = 0
    
    # Update Last Match Date & Fatigue Lists
    player_last_match[w_name] = match_date
    player_last_match[l_name] = match_date
    
    if w_name not in player_recent_matches: player_recent_matches[w_name] = []
    if l_name not in player_recent_matches: player_recent_matches[l_name] = []
    player_recent_matches[w_name].append(match_date)
    player_recent_matches[l_name].append(match_date)
    player_recent_matches[w_name] = [d for d in player_recent_matches[w_name] if (match_date - d).days <= 21]
    player_recent_matches[l_name] = [d for d in player_recent_matches[l_name] if (match_date - d).days <= 21]
    
    # Update Clutch Factor State
    # Heuristic for "Close Match": Has a tiebreak OR went to a deciding set
    score_str = str(row.get('score', ''))
    best_of = row.get('best_of', 3)
    try:
        num_sets_played = len(score_str.split())
        is_close_match = ('(' in score_str) or (num_sets_played >= int(best_of) if pd.notna(best_of) else False)
    except:
        is_close_match = False
        
    if is_close_match:
        player_clutch_wins[w_name] = player_clutch_wins.get(w_name, 0) + 1
        player_clutch_matches[w_name] = player_clutch_matches.get(w_name, 0) + 1
        player_clutch_matches[l_name] = player_clutch_matches.get(l_name, 0) + 1

# --- Map Features Back to Main DataFrame ---
df['w_h2h'] = w_h2h_list
df['l_h2h'] = l_h2h_list
df['h2h_diff'] = df['w_h2h'] - df['l_h2h']

df['w_streak'] = w_streak_list
df['l_streak'] = l_streak_list
df['streak_diff'] = df['w_streak'] - df['l_streak']

df['w_fatigue'] = w_fatigue_list
df['l_fatigue'] = l_fatigue_list
df['fatigue_diff'] = df['w_fatigue'] - df['l_fatigue']

df['w_rustiness'] = w_rust_list
df['l_rustiness'] = l_rust_list
df['rustiness_diff'] = df['w_rustiness'] - df['l_rustiness']

df['w_home'] = w_home_list
df['l_home'] = l_home_list
df['home_diff'] = df['w_home'] - df['l_home']

df['w_clutch'] = w_clutch_list
df['l_clutch'] = l_clutch_list
df['clutch_diff'] = df['w_clutch'] - df['l_clutch']

print("Features engineered successfully.")
print(df[['winner_name', 'loser_name', 'w_home', 'l_home', 'w_clutch', 'l_clutch']].tail())

df.to_csv("features_atp_matches.csv", index=False)
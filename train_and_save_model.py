import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

matches_df = pd.read_csv('matches.csv', parse_dates=['date'])
matches_df = matches_df.sort_values('date').reset_index(drop=True)

SEASON_YEAR = 2024

def get_last_n_matches_2024(df, team, before_date, n):
    mask = (
        (df['date'].dt.year == SEASON_YEAR) &
        ((df['team1'] == team) | (df['team2'] == team)) &
        (df['date'] < before_date)
    )
    matches = df[mask].sort_values('date', ascending=False).head(n)
    return matches

def get_team_recent_wins_2024(df, team, before_date, n):
    matches = get_last_n_matches_2024(df, team, before_date, n)
    wins = (matches['winner'] == team).sum()
    return int(wins)

def get_last5_match_results_2024(df, team, before_date):
    matches = get_last_n_matches_2024(df, team, before_date, 5)
    result_list = []
    for _, row in matches.iterrows():
        opponent = row['team2'] if row['team1'] == team else row['team1']
        result = 'W' if row['winner'] == team else 'L'
        venue = row['venue']
        date_str = row['date'].strftime('%Y-%m-%d')
        result_list.append(f"{date_str} at {venue}: vs {opponent} - {result}")
    return result_list

def get_team_ground_stats_all(df, team, venue, before_date):
    mask = ((df['team1'] == team) | (df['team2'] == team)) & (df['venue'] == venue) & (df['date'] < before_date)
    matches = df[mask]
    wins = (matches['winner'] == team).sum()
    losses = len(matches) - wins
    total = len(matches)
    return wins, losses, total

def get_last_n_h2h_matches_all(df, team1, team2, before_date, n):
    mask = (
        (((df['team1'] == team1) & (df['team2'] == team2)) |
         ((df['team1'] == team2) & (df['team2'] == team1))) &
        (df['date'] < before_date)
    )
    matches = df[mask].sort_values('date', ascending=False).head(n)
    return matches

def get_h2h_wins_all(df, team, other_team, before_date, n):
    matches = get_last_n_h2h_matches_all(df, team, other_team, before_date, n)
    wins = (matches['winner'] == team).sum()
    return int(wins)

def get_last5_h2h_results_all(df, team1, team2, before_date):
    matches = get_last_n_h2h_matches_all(df, team1, team2, before_date, 5)
    result_list = []
    for _, row in matches.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        venue = row['venue']
        winner = row['winner']
        result_list.append(f"{date_str} at {venue}: {row['team1']} vs {row['team2']} - Winner: {winner}")
    return result_list

features = {
    'team1_recent_wins_5': [],
    'team2_recent_wins_5': [],
    'team1_ground_wins': [],
    'team1_ground_losses': [],
    'team1_ground_matches': [],
    'team2_ground_wins': [],
    'team2_ground_losses': [],
    'team2_ground_matches': [],
    'team1_h2h_wins': [],
    'team2_h2h_wins': [],
    'team1_last5_results': [],
    'team2_last5_results': [],
    'last5_h2h_results': []
}

for idx, row in matches_df.iterrows():
    date = row['date']
    team1 = row['team1']
    team2 = row['team2']
    venue = row['venue']

    # Recent form: 2024 only
    features['team1_recent_wins_5'].append(get_team_recent_wins_2024(matches_df.iloc[:idx], team1, date, 5))
    features['team2_recent_wins_5'].append(get_team_recent_wins_2024(matches_df.iloc[:idx], team2, date, 5))

    # Recent match results: 2024 only
    features['team1_last5_results'].append(get_last5_match_results_2024(matches_df.iloc[:idx], team1, date))
    features['team2_last5_results'].append(get_last5_match_results_2024(matches_df.iloc[:idx], team2, date))

    # Ground stats: all-time
    gw1, gl1, gm1 = get_team_ground_stats_all(matches_df.iloc[:idx], team1, venue, date)
    features['team1_ground_wins'].append(gw1)
    features['team1_ground_losses'].append(gl1)
    features['team1_ground_matches'].append(gm1)

    gw2, gl2, gm2 = get_team_ground_stats_all(matches_df.iloc[:idx], team2, venue, date)
    features['team2_ground_wins'].append(gw2)
    features['team2_ground_losses'].append(gl2)
    features['team2_ground_matches'].append(gm2)

    # H2H: last 5 ever
    features['team1_h2h_wins'].append(get_h2h_wins_all(matches_df.iloc[:idx], team1, team2, date, 5))
    features['team2_h2h_wins'].append(get_h2h_wins_all(matches_df.iloc[:idx], team2, team1, date, 5))
    features['last5_h2h_results'].append(get_last5_h2h_results_all(matches_df.iloc[:idx], team1, team2, date))

for key in features:
    matches_df[key] = features[key]

feature_cols = [
    'team1', 'team2', 'venue',
    'team1_recent_wins_5', 'team2_recent_wins_5',
    'team1_ground_wins', 'team1_ground_losses', 'team1_ground_matches',
    'team2_ground_wins', 'team2_ground_losses', 'team2_ground_matches',
    'team1_h2h_wins', 'team2_h2h_wins'
]
encoders = {}
for col in ['team1', 'team2', 'venue']:
    le = LabelEncoder()
    matches_df[col] = le.fit_transform(matches_df[col].astype(str))
    encoders[col] = le

target_col = 'winner'
target_encoder = LabelEncoder()
matches_df[target_col] = target_encoder.fit_transform(matches_df[target_col].astype(str))

X = matches_df[feature_cols]
y = matches_df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")

joblib.dump(clf, 'ipl_model.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')
print("Model and encoders saved: ipl_model.pkl, encoders.pkl, target_encoder.pkl")

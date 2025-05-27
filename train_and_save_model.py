import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 1. Load and sort matches data
matches_df = pd.read_csv('matches.csv', parse_dates=['date'])
matches_df = matches_df.sort_values('date').reset_index(drop=True)

# 2. Feature engineering functions

def get_last_n_matches(df, team, before_date, n):
    mask = (((df['team1'] == team) | (df['team2'] == team)) & (df['date'] < before_date))
    matches = df[mask].sort_values('date', ascending=False).head(n)
    return matches

def get_team_recent_wins(df, team, before_date, n):
    matches = get_last_n_matches(df, team, before_date, n)
    wins = (matches['winner'] == team).sum()
    return int(wins)

def get_team_ground_winrate(df, team, venue, before_date):
    mask = (((df['team1'] == team) | (df['team2'] == team)) & (df['venue'] == venue) & (df['date'] < before_date))
    matches = df[mask]
    if len(matches) == 0:
        return 0.0
    wins = (matches['winner'] == team).sum()
    return wins / len(matches)

def get_last_n_h2h_matches(df, team1, team2, before_date, n):
    mask = (
        (((df['team1'] == team1) & (df['team2'] == team2)) |
         ((df['team1'] == team2) & (df['team2'] == team1)))
        & (df['date'] < before_date)
    )
    matches = df[mask].sort_values('date', ascending=False).head(n)
    return matches

def get_h2h_wins(df, team, other_team, before_date, n):
    matches = get_last_n_h2h_matches(df, team, other_team, before_date, n)
    wins = (matches['winner'] == team).sum()
    return int(wins)

# 3. Engineer features for the dataset
features = {
    'team1_recent_wins_5': [],
    'team2_recent_wins_5': [],
    'team1_ground_winrate': [],
    'team2_ground_winrate': [],
    'team1_h2h_wins': [],
    'team2_h2h_wins': []
}

for idx, row in matches_df.iterrows():
    date = row['date']
    team1 = row['team1']
    team2 = row['team2']
    venue = row['venue']

    features['team1_recent_wins_5'].append(get_team_recent_wins(matches_df.iloc[:idx], team1, date, 5))
    features['team2_recent_wins_5'].append(get_team_recent_wins(matches_df.iloc[:idx], team2, date, 5))
    features['team1_ground_winrate'].append(get_team_ground_winrate(matches_df.iloc[:idx], team1, venue, date))
    features['team2_ground_winrate'].append(get_team_ground_winrate(matches_df.iloc[:idx], team2, venue, date))
    features['team1_h2h_wins'].append(get_h2h_wins(matches_df.iloc[:idx], team1, team2, date, 5))
    features['team2_h2h_wins'].append(get_h2h_wins(matches_df.iloc[:idx], team2, team1, date, 5))

for key in features:
    matches_df[key] = features[key]

# 4. Encode categorical features
feature_cols = [
    'team1', 'team2', 'venue',
    'team1_recent_wins_5', 'team2_recent_wins_5',
    'team1_ground_winrate', 'team2_ground_winrate',
    'team1_h2h_wins', 'team2_h2h_wins'
]
encoders = {}
for col in ['team1', 'team2', 'venue']:
    le = LabelEncoder()
    matches_df[col] = le.fit_transform(matches_df[col].astype(str))
    encoders[col] = le

# 5. Encode target
target_col = 'winner'
target_encoder = LabelEncoder()
matches_df[target_col] = target_encoder.fit_transform(matches_df[target_col].astype(str))

# 6. Train/test split
X = matches_df[feature_cols]
y = matches_df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# 7. Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 8. Evaluate (optional)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")

# 9. Save model and encoders
joblib.dump(clf, 'ipl_model.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')
print("Model and encoders saved: ipl_model.pkl, encoders.pkl, target_encoder.pkl")

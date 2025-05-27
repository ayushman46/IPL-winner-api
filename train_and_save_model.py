import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
matches_df = pd.read_csv('matches.csv', parse_dates=['date'])
matches_df = matches_df.sort_values('date').reset_index(drop=True)

SEASON_YEAR = 2024

def get_last_n_matches_2024(df, team, before_date, n):
    """Get last n matches for a team in 2024 season before a given date"""
    mask = (
        (df['date'].dt.year == SEASON_YEAR) &
        ((df['team1'] == team) | (df['team2'] == team)) &
        (df['date'] < before_date)
    )
    matches = df[mask].sort_values('date', ascending=False).head(n)
    return matches

def get_team_recent_wins_2024(df, team, before_date, n):
    """Get number of wins in last n matches for 2024 season"""
    matches = get_last_n_matches_2024(df, team, before_date, n)
    if len(matches) == 0:
        return 0
    wins = (matches['winner'] == team).sum()
    return int(wins)

def get_last5_match_results_2024(df, team, before_date):
    """Get detailed results of last 5 matches in 2024"""
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
    """Get all-time stats for a team at a specific venue"""
    mask = (
        ((df['team1'] == team) | (df['team2'] == team)) & 
        (df['venue'] == venue) & 
        (df['date'] < before_date)
    )
    matches = df[mask]
    if len(matches) == 0:
        return 0, 0, 0
    wins = (matches['winner'] == team).sum()
    total = len(matches)
    losses = total - wins
    return int(wins), int(losses), int(total)

def get_last_n_h2h_matches_all(df, team1, team2, before_date, n):
    """Get last n head-to-head matches between two teams"""
    mask = (
        (((df['team1'] == team1) & (df['team2'] == team2)) |
         ((df['team1'] == team2) & (df['team2'] == team1))) &
        (df['date'] < before_date)
    )
    matches = df[mask].sort_values('date', ascending=False).head(n)
    return matches

def get_h2h_wins_all(df, team, other_team, before_date, n):
    """Get number of wins in last n head-to-head matches"""
    matches = get_last_n_h2h_matches_all(df, team, other_team, before_date, n)
    if len(matches) == 0:
        return 0
    wins = (matches['winner'] == team).sum()
    return int(wins)

def get_last5_h2h_results_all(df, team1, team2, before_date):
    """Get detailed results of last 5 head-to-head matches"""
    matches = get_last_n_h2h_matches_all(df, team1, team2, before_date, 5)
    result_list = []
    for _, row in matches.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        venue = row['venue']
        winner = row['winner']
        result_list.append(f"{date_str} at {venue}: {row['team1']} vs {row['team2']} - Winner: {winner}")
    return result_list

# Initialize features dictionary
features = {
    'team1_recent_wins_5': [],
    'team2_recent_wins_5': [],
    'team1_recent_matches_5': [],  # Total matches played in last 5
    'team2_recent_matches_5': [],
    'team1_ground_wins': [],
    'team1_ground_losses': [],
    'team1_ground_matches': [],
    'team2_ground_wins': [],
    'team2_ground_losses': [],
    'team2_ground_matches': [],
    'team1_h2h_wins': [],
    'team2_h2h_wins': [],
    'total_h2h_matches': [],  # Total H2H matches for proper percentage calculation
    'team1_last5_results': [],
    'team2_last5_results': [],
    'last5_h2h_results': []
}

print("Generating features for each match...")
for idx, row in matches_df.iterrows():
    if idx % 100 == 0:
        print(f"Processing match {idx}/{len(matches_df)}")
    
    date = row['date']
    team1 = row['team1']
    team2 = row['team2']
    venue = row['venue']

    # Recent form: 2024 only
    team1_recent_wins = get_team_recent_wins_2024(matches_df.iloc[:idx], team1, date, 5)
    team2_recent_wins = get_team_recent_wins_2024(matches_df.iloc[:idx], team2, date, 5)
    
    # Count total recent matches played
    team1_recent_matches = len(get_last_n_matches_2024(matches_df.iloc[:idx], team1, date, 5))
    team2_recent_matches = len(get_last_n_matches_2024(matches_df.iloc[:idx], team2, date, 5))
    
    features['team1_recent_wins_5'].append(team1_recent_wins)
    features['team2_recent_wins_5'].append(team2_recent_wins)
    features['team1_recent_matches_5'].append(team1_recent_matches)
    features['team2_recent_matches_5'].append(team2_recent_matches)

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

    # H2H: last 5 matches ever between these teams
    team1_h2h_wins = get_h2h_wins_all(matches_df.iloc[:idx], team1, team2, date, 5)
    team2_h2h_wins = get_h2h_wins_all(matches_df.iloc[:idx], team2, team1, date, 5)
    total_h2h = len(get_last_n_h2h_matches_all(matches_df.iloc[:idx], team1, team2, date, 5))
    
    features['team1_h2h_wins'].append(team1_h2h_wins)
    features['team2_h2h_wins'].append(team2_h2h_wins)
    features['total_h2h_matches'].append(total_h2h)
    features['last5_h2h_results'].append(get_last5_h2h_results_all(matches_df.iloc[:idx], team1, team2, date))

# Add features to dataframe
for key in features:
    matches_df[key] = features[key]

print("Feature generation completed!")

# Define feature columns for the model
feature_cols = [
    'team1', 'team2', 'venue',
    'team1_recent_wins_5', 'team2_recent_wins_5',
    'team1_recent_matches_5', 'team2_recent_matches_5',
    'team1_ground_wins', 'team1_ground_losses', 'team1_ground_matches',
    'team2_ground_wins', 'team2_ground_losses', 'team2_ground_matches',
    'team1_h2h_wins', 'team2_h2h_wins', 'total_h2h_matches'
]

# Encode categorical features
encoders = {}
for col in ['team1', 'team2', 'venue']:
    le = LabelEncoder()
    matches_df[col + '_encoded'] = le.fit_transform(matches_df[col].astype(str))
    encoders[col] = le

# Prepare feature columns with encoded values
model_feature_cols = [
    'team1_encoded', 'team2_encoded', 'venue_encoded',
    'team1_recent_wins_5', 'team2_recent_wins_5',
    'team1_recent_matches_5', 'team2_recent_matches_5',
    'team1_ground_wins', 'team1_ground_losses', 'team1_ground_matches',
    'team2_ground_wins', 'team2_ground_losses', 'team2_ground_matches',
    'team1_h2h_wins', 'team2_h2h_wins', 'total_h2h_matches'
]

# Encode target variable
target_col = 'winner'
target_encoder = LabelEncoder()
matches_df[target_col + '_encoded'] = target_encoder.fit_transform(matches_df[target_col].astype(str))

# Prepare training data
X = matches_df[model_feature_cols].fillna(0)  # Fill NaN values with 0
y = matches_df[target_col + '_encoded']

# Split the data (using temporal split - older matches for training)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Train the model
print("Training the model...")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")

# Print feature importance
feature_importance = pd.DataFrame({
    'feature': model_feature_cols,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Save the model and encoders
joblib.dump(clf, 'ipl_model.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')

# Save additional metadata
metadata = {
    'feature_cols': model_feature_cols,
    'season_year': SEASON_YEAR,
    'model_accuracy': acc,
    'teams': list(encoders['team1'].classes_),
    'venues': list(encoders['venue'].classes_)
}
joblib.dump(metadata, 'model_metadata.pkl')

print("\nModel and encoders saved successfully!")
print("Files saved: ipl_model.pkl, encoders.pkl, target_encoder.pkl, model_metadata.pkl")
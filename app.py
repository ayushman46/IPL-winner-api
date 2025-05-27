from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load model, encoders, and matches data at startup
model = joblib.load('ipl_model.pkl')
encoders = joblib.load('encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')
matches_df = pd.read_csv('matches.csv', parse_dates=['date'])
matches_df = matches_df.sort_values('date').reset_index(drop=True)

FEATURE_COLS = [
    'team1', 'team2', 'venue',
    'team1_recent_wins_5', 'team2_recent_wins_5',
    'team1_ground_winrate', 'team2_ground_winrate',
    'team1_h2h_wins', 'team2_h2h_wins'
]

def get_last5_matches_list(df, team, before_date):
    matches = df[((df['team1'] == team) | (df['team2'] == team)) & (df['date'] < before_date)]
    matches = matches.sort_values('date', ascending=False).head(5)
    result_list = []
    for _, row in matches.iterrows():
        opponent = row['team2'] if row['team1'] == team else row['team1']
        result = 'W' if row['winner'] == team else 'L'
        result_list.append(f"vs {opponent} - {result}")
    return result_list

def get_last5_h2h_list(df, team1, team2, before_date):
    mask = (((df['team1'] == team1) & (df['team2'] == team2)) | ((df['team1'] == team2) & (df['team2'] == team1))) & (df['date'] < before_date)
    matches = df[mask].sort_values('date', ascending=False).head(5)
    result_list = []
    for _, row in matches.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        winner = row['winner']
        result_list.append(f"{date_str}: {row['team1']} vs {row['team2']} - Winner: {winner}")
    return result_list

def get_team_recent_wins(df, team, before_date, n):
    matches = get_last5_matches_list(df, team, before_date)
    wins = sum(1 for res in matches if res.endswith('- W'))
    return wins

def get_team_ground_winrate(df, team, venue, before_date):
    mask = (((df['team1'] == team) | (df['team2'] == team)) & (df['venue'] == venue) & (df['date'] < before_date))
    matches = df[mask]
    if len(matches) == 0:
        return 0.0
    wins = (matches['winner'] == team).sum()
    return wins / len(matches)

def get_h2h_wins(df, team, other_team, before_date, n):
    matches = get_last5_h2h_list(df, team, other_team, before_date)
    wins = sum(1 for m in matches if f"Winner: {team}" in m)
    return wins

def prepare_features(team1, team2, venue, match_date):
    # Compute all features dynamically
    team1_last5_matches = get_last5_matches_list(matches_df, team1, match_date)
    team2_last5_matches = get_last5_matches_list(matches_df, team2, match_date)
    last_5_h2h_matches = get_last5_h2h_list(matches_df, team1, team2, match_date)

    team1_recent_wins_5 = get_team_recent_wins(matches_df, team1, match_date, 5)
    team2_recent_wins_5 = get_team_recent_wins(matches_df, team2, match_date, 5)
    team1_ground_winrate = get_team_ground_winrate(matches_df, team1, venue, match_date)
    team2_ground_winrate = get_team_ground_winrate(matches_df, team2, venue, match_date)
    team1_h2h_wins = get_h2h_wins(matches_df, team1, team2, match_date, 5)
    team2_h2h_wins = get_h2h_wins(matches_df, team2, team1, match_date, 5)

    return {
        'team1': team1,
        'team2': team2,
        'venue': venue,
        'team1_recent_wins_5': team1_recent_wins_5,
        'team2_recent_wins_5': team2_recent_wins_5,
        'team1_ground_winrate': team1_ground_winrate,
        'team2_ground_winrate': team2_ground_winrate,
        'team1_h2h_wins': team1_h2h_wins,
        'team2_h2h_wins': team2_h2h_wins,
        'team1_last5_matches': team1_last5_matches,
        'team2_last5_matches': team2_last5_matches,
        'last_5_h2h_matches': last_5_h2h_matches
    }

def encode_features(row, encoders):
    row_enc = row.copy()
    for col in ['team1', 'team2', 'venue']:
        row_enc[col] = encoders[col].transform([row[col]])[0]
    return row_enc

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Get user input
    team1 = data['team1']
    team2 = data['team2']
    venue = data['venue']
    # Optional: allow user to specify date for "historic" predictions, else use today
    match_date = pd.Timestamp(data.get('date', datetime.today().strftime('%Y-%m-%d')))
    # Compute all features and lists
    features = prepare_features(team1, team2, venue, match_date)
    # Prepare input for model
    model_input = encode_features(features, encoders)
    X_input = np.array([[model_input[col] for col in FEATURE_COLS]], dtype=float)
    # Predict
    pred = model.predict(X_input)
    winner = target_encoder.inverse_transform(pred)[0]
    # Return prediction and lists
    response = {
        "predicted_winner": winner,
        "team1_last5_matches": features['team1_last5_matches'],
        "team2_last5_matches": features['team2_last5_matches'],
        "last_5_h2h_matches": features['last_5_h2h_matches']
    }
    return jsonify(response)

@app.route('/')
def home():
    return "IPL Winner Predictor API is live!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

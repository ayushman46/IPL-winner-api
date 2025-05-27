from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# --- Custom JSON provider for NumPy/pandas types ---
from flask.json.provider import DefaultJSONProvider

class CustomJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            try:
                return obj.item()
            except Exception:
                pass
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        return super().default(obj)

app = Flask(__name__)
app.json_provider_class = CustomJSONProvider
CORS(app)

model = joblib.load('ipl_model.pkl')
encoders = joblib.load('encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')
matches_df = pd.read_csv('matches.csv', parse_dates=['date'])
matches_df = matches_df.sort_values('date').reset_index(drop=True)

SEASON_YEAR = 2024

FEATURE_COLS = [
    'team1', 'team2', 'venue',
    'team1_recent_wins_5', 'team2_recent_wins_5',
    'team1_ground_wins', 'team1_ground_losses', 'team1_ground_matches',
    'team2_ground_wins', 'team2_ground_losses', 'team2_ground_matches',
    'team1_h2h_wins', 'team2_h2h_wins'
]

def get_last5_matches_results_2024(df, team, before_date):
    mask = (df['date'].dt.year == SEASON_YEAR) & ((df['team1'] == team) | (df['team2'] == team)) & (df['date'] < before_date)
    matches = df[mask].sort_values('date', ascending=False).head(5)
    result_list = []
    for _, row in matches.iterrows():
        opponent = row['team2'] if row['team1'] == team else row['team1']
        result = 'W' if row['winner'] == team else 'L'
        venue = row['venue']
        date_str = row['date'].strftime('%Y-%m-%d')
        result_list.append(f"{date_str} at {venue}: vs {opponent} - {result}")
    return result_list

def get_team_recent_wins_2024(df, team, before_date, n):
    mask = (df['date'].dt.year == SEASON_YEAR) & ((df['team1'] == team) | (df['team2'] == team)) & (df['date'] < before_date)
    matches = df[mask].sort_values('date', ascending=False).head(n)
    wins = (matches['winner'] == team).sum()
    return int(wins)

def get_team_ground_stats_all(df, team, venue, before_date):
    mask = ((df['team1'] == team) | (df['team2'] == team)) & (df['venue'] == venue) & (df['date'] < before_date)
    matches = df[mask]
    wins = (matches['winner'] == team).sum()
    losses = len(matches) - wins
    total = len(matches)
    return wins, losses, total

def get_last5_h2h_results_all(df, team1, team2, before_date):
    mask = (
        (((df['team1'] == team1) & (df['team2'] == team2)) |
         ((df['team1'] == team2) & (df['team2'] == team1))) &
        (df['date'] < before_date)
    )
    matches = df[mask].sort_values('date', ascending=False).head(5)
    result_list = []
    for _, row in matches.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        venue = row['venue']
        winner = row['winner']
        result_list.append(f"{date_str} at {venue}: {row['team1']} vs {row['team2']} - Winner: {winner}")
    return result_list

def get_h2h_wins_all(df, team, other_team, before_date, n):
    mask = (
        (((df['team1'] == team) & (df['team2'] == other_team)) |
         ((df['team1'] == other_team) & (df['team2'] == team))) &
        (df['date'] < before_date)
    )
    matches = df[mask].sort_values('date', ascending=False).head(n)
    wins = (matches['winner'] == team).sum()
    return int(wins)

def prepare_features(team1, team2, venue, match_date):
    team1_last5_results = get_last5_matches_results_2024(matches_df, team1, match_date)
    team2_last5_results = get_last5_matches_results_2024(matches_df, team2, match_date)
    last5_h2h_results = get_last5_h2h_results_all(matches_df, team1, team2, match_date)

    team1_recent_wins_5 = get_team_recent_wins_2024(matches_df, team1, match_date, 5)
    team2_recent_wins_5 = get_team_recent_wins_2024(matches_df, team2, match_date, 5)

    team1_ground_wins, team1_ground_losses, team1_ground_matches = get_team_ground_stats_all(matches_df, team1, venue, match_date)
    team2_ground_wins, team2_ground_losses, team2_ground_matches = get_team_ground_stats_all(matches_df, team2, venue, match_date)

    team1_h2h_wins = get_h2h_wins_all(matches_df, team1, team2, match_date, 5)
    team2_h2h_wins = get_h2h_wins_all(matches_df, team2, team1, match_date, 5)

    # Win percentages
    team1_recent_win_pct = (team1_recent_wins_5 / 5) * 100
    team2_recent_win_pct = (team2_recent_wins_5 / 5) * 100

    team1_ground_win_pct = (team1_ground_wins / team1_ground_matches * 100) if team1_ground_matches > 0 else 0
    team2_ground_win_pct = (team2_ground_wins / team2_ground_matches * 100) if team2_ground_matches > 0 else 0

    total_h2h_matches = team1_h2h_wins + team2_h2h_wins
    if total_h2h_matches > 0:
        team1_h2h_win_pct = (team1_h2h_wins / total_h2h_matches) * 100
        team2_h2h_win_pct = (team2_h2h_wins / total_h2h_matches) * 100
    else:
        team1_h2h_win_pct = team2_h2h_win_pct = 0

    # Weighted sum (recent form 50%, ground 30%, H2H 20%)
    team1_win_predictor = (team1_recent_win_pct * 0.5) + (team1_ground_win_pct * 0.3) + (team1_h2h_win_pct * 0.2)
    team2_win_predictor = (team2_recent_win_pct * 0.5) + (team2_ground_win_pct * 0.3) + (team2_h2h_win_pct * 0.2)

    # Normalize to sum to 100
    total = team1_win_predictor + team2_win_predictor
    if total > 0:
        team1_win_predictor = (team1_win_predictor / total) * 100
        team2_win_predictor = (team2_win_predictor / total) * 100
    else:
        team1_win_predictor = team2_win_predictor = 50

    return {
        'team1': team1,
        'team2': team2,
        'venue': venue,
        'team1_recent_wins_5': team1_recent_wins_5,
        'team2_recent_wins_5': team2_recent_wins_5,
        'team1_ground_wins': team1_ground_wins,
        'team1_ground_losses': team1_ground_losses,
        'team1_ground_matches': team1_ground_matches,
        'team2_ground_wins': team2_ground_wins,
        'team2_ground_losses': team2_ground_losses,
        'team2_ground_matches': team2_ground_matches,
        'team1_h2h_wins': team1_h2h_wins,
        'team2_h2h_wins': team2_h2h_wins,
        'team1_last5_results': team1_last5_results,
        'team2_last5_results': team2_last5_results,
        'last5_h2h_results': last5_h2h_results,
        'team1_win_predictor': team1_win_predictor,
        'team2_win_predictor': team2_win_predictor
    }

def encode_features(row, encoders):
    row_enc = row.copy()
    for col in ['team1', 'team2', 'venue']:
        try:
            row_enc[col] = encoders[col].transform([row[col]])[0]
        except Exception as e:
            raise ValueError(f"Unknown value for '{col}': '{row[col]}'. Please use one of: {list(encoders[col].classes_)}")
    return row_enc

def convert_all_types(obj):
    import pandas as pd
    if isinstance(obj, dict):
        return {k: convert_all_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_all_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    elif hasattr(obj, 'item'):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    else:
        return obj

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        for key in ['team1', 'team2', 'venue']:
            if key not in data:
                return jsonify({'error': f"Missing required key: '{key}'"}), 400
        team1 = data['team1']
        team2 = data['team2']
        venue = data['venue']
        match_date = pd.Timestamp(data.get('date', datetime.today().strftime('%Y-%m-%d')))
        features = prepare_features(team1, team2, venue, match_date)
        try:
            model_input = encode_features(features, encoders)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        X_input = np.array([[model_input[col] for col in FEATURE_COLS]], dtype=float)
        pred = model.predict(X_input)
        winner = target_encoder.inverse_transform(pred)[0]
        response = {
            'predicted_winner': str(winner),
            'team1_last5_results': list(features.get('team1_last5_results', [])),
            'team2_last5_results': list(features.get('team2_last5_results', [])),
            'last5_h2h_results': list(features.get('last5_h2h_results', [])),
            'team1_win_predictor': float(round(features['team1_win_predictor'], 2)),
            'team2_win_predictor': float(round(features['team2_win_predictor'], 2)),
            'team1_ground_stats': {
                'matches_played': int(features['team1_ground_matches']),
                'wins': int(features['team1_ground_wins']),
                'losses': int(features['team1_ground_losses'])
            },
            'team2_ground_stats': {
                'matches_played': int(features['team2_ground_matches']),
                'wins': int(features['team2_ground_wins']),
                'losses': int(features['team2_ground_losses'])
            }
        }
        # Ensure all values are serializable
        return jsonify(convert_all_types(response))
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    print(traceback.format_exc())
    return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/')
def home():
    return "IPL Winner Predictor API is live!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

# Custom JSON provider for NumPy/pandas types
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

# Load model and data
try:
    model = joblib.load('ipl_model.pkl')
    encoders = joblib.load('encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    metadata = joblib.load('model_metadata.pkl')
    matches_df = pd.read_csv('matches.csv', parse_dates=['date'])
    matches_df = matches_df.sort_values('date').reset_index(drop=True)
    
    print("Model and data loaded successfully!")
    print(f"Available teams: {metadata['teams']}")
    print(f"Available venues: {metadata['venues']}")
    
except Exception as e:
    print(f"Error loading model/data: {e}")
    raise

SEASON_YEAR = metadata['season_year']
FEATURE_COLS = metadata['feature_cols']

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

def get_last5_matches_results_2024(df, team, before_date):
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

def calculate_win_percentages(team1_features, team2_features):
    """Calculate proper win percentages that sum to 100%"""
    
    # Recent form percentage (2024 season)
    team1_recent_matches = team1_features.get('recent_matches', 0)
    team2_recent_matches = team2_features.get('recent_matches', 0)
    
    team1_recent_pct = (team1_features['recent_wins'] / team1_recent_matches * 100) if team1_recent_matches > 0 else 50
    team2_recent_pct = (team2_features['recent_wins'] / team2_recent_matches * 100) if team2_recent_matches > 0 else 50
    
    # Ground performance percentage
    team1_ground_matches = team1_features['ground_matches']
    team2_ground_matches = team2_features['ground_matches']
    
    team1_ground_pct = (team1_features['ground_wins'] / team1_ground_matches * 100) if team1_ground_matches > 0 else 50
    team2_ground_pct = (team2_features['ground_wins'] / team2_ground_matches * 100) if team2_ground_matches > 0 else 50
    
    # Head-to-head percentage
    total_h2h = team1_features['h2h_wins'] + team2_features['h2h_wins']
    if total_h2h > 0:
        team1_h2h_pct = (team1_features['h2h_wins'] / total_h2h * 100)
        team2_h2h_pct = (team2_features['h2h_wins'] / total_h2h * 100)
    else:
        team1_h2h_pct = team2_h2h_pct = 50
    
    # Weighted combination (Recent: 40%, Ground: 35%, H2H: 25%)
    team1_combined = (team1_recent_pct * 0.4) + (team1_ground_pct * 0.35) + (team1_h2h_pct * 0.25)
    team2_combined = (team2_recent_pct * 0.4) + (team2_ground_pct * 0.35) + (team2_h2h_pct * 0.25)
    
    # Normalize to ensure sum equals 100%
    total = team1_combined + team2_combined
    if total > 0:
        team1_final = (team1_combined / total) * 100
        team2_final = (team2_combined / total) * 100
    else:
        team1_final = team2_final = 50.0
    
    return round(team1_final, 2), round(team2_final, 2)

def prepare_features(team1, team2, venue, match_date):
    """Prepare all features for prediction"""
    
    # Get recent form data (2024 season)
    team1_recent_wins = get_team_recent_wins_2024(matches_df, team1, match_date, 5)
    team2_recent_wins = get_team_recent_wins_2024(matches_df, team2, match_date, 5)
    
    team1_recent_matches = len(get_last_n_matches_2024(matches_df, team1, match_date, 5))
    team2_recent_matches = len(get_last_n_matches_2024(matches_df, team2, match_date, 5))
    
    # Get last 5 match results (2024 season)
    team1_last5_results = get_last5_matches_results_2024(matches_df, team1, match_date)
    team2_last5_results = get_last5_matches_results_2024(matches_df, team2, match_date)
    
    # Get ground stats (all-time)
    team1_ground_wins, team1_ground_losses, team1_ground_matches = get_team_ground_stats_all(matches_df, team1, venue, match_date)
    team2_ground_wins, team2_ground_losses, team2_ground_matches = get_team_ground_stats_all(matches_df, team2, venue, match_date)
    
    # Get head-to-head stats (last 5 matches)
    team1_h2h_wins = get_h2h_wins_all(matches_df, team1, team2, match_date, 5)
    team2_h2h_wins = get_h2h_wins_all(matches_df, team2, team1, match_date, 5)
    total_h2h_matches = len(get_last_n_h2h_matches_all(matches_df, team1, team2, match_date, 5))
    
    # Get H2H results
    last5_h2h_results = get_last5_h2h_results_all(matches_df, team1, team2, match_date)
    
    # Calculate win percentages
    team1_stats = {
        'recent_wins': team1_recent_wins,
        'recent_matches': team1_recent_matches,
        'ground_wins': team1_ground_wins,
        'ground_matches': team1_ground_matches,
        'h2h_wins': team1_h2h_wins
    }
    
    team2_stats = {
        'recent_wins': team2_recent_wins,
        'recent_matches': team2_recent_matches,
        'ground_wins': team2_ground_wins,
        'ground_matches': team2_ground_matches,
        'h2h_wins': team2_h2h_wins
    }
    
    team1_win_pct, team2_win_pct = calculate_win_percentages(team1_stats, team2_stats)
    
    return {
        'team1_encoded': 0,  # Will be encoded later
        'team2_encoded': 0,  # Will be encoded later  
        'venue_encoded': 0,  # Will be encoded later
        'team1_recent_wins_5': team1_recent_wins,
        'team2_recent_wins_5': team2_recent_wins,
        'team1_recent_matches_5': team1_recent_matches,
        'team2_recent_matches_5': team2_recent_matches,
        'team1_ground_wins': team1_ground_wins,
        'team1_ground_losses': team1_ground_losses,
        'team1_ground_matches': team1_ground_matches,
        'team2_ground_wins': team2_ground_wins,
        'team2_ground_losses': team2_ground_losses,
        'team2_ground_matches': team2_ground_matches,
        'team1_h2h_wins': team1_h2h_wins,
        'team2_h2h_wins': team2_h2h_wins,
        'total_h2h_matches': total_h2h_matches,
        # Additional data for response
        'team1_last5_results': team1_last5_results,
        'team2_last5_results': team2_last5_results,
        'last5_h2h_results': last5_h2h_results,
        'team1_win_percentage': team1_win_pct,
        'team2_win_percentage': team2_win_pct,
        'team1': team1,
        'team2': team2,
        'venue': venue
    }

def encode_features(features_dict):
    """Encode categorical features for model input"""
    try:
        features_dict['team1_encoded'] = encoders['team1'].transform([features_dict['team1']])[0]
        features_dict['team2_encoded'] = encoders['team2'].transform([features_dict['team2']])[0]
        features_dict['venue_encoded'] = encoders['venue'].transform([features_dict['venue']])[0]
        return features_dict
    except ValueError as e:
        if 'team1' in str(e):
            raise ValueError(f"Unknown team '{features_dict['team1']}'. Available teams: {list(encoders['team1'].classes_)}")
        elif 'team2' in str(e):
            raise ValueError(f"Unknown team '{features_dict['team2']}'. Available teams: {list(encoders['team2'].classes_)}")
        elif 'venue' in str(e):
            raise ValueError(f"Unknown venue '{features_dict['venue']}'. Available venues: {list(encoders['venue'].classes_)}")
        else:
            raise e

def convert_to_serializable(obj):
    """Convert numpy/pandas types to Python native types"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
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
        
        # Validate required fields
        required_fields = ['team1', 'team2', 'venue']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f"Missing required field: '{field}'"}), 400
        
        team1 = data['team1']
        team2 = data['team2']
        venue = data['venue']
        match_date = pd.Timestamp(data.get('date', datetime.today().strftime('%Y-%m-%d')))
        
        # Prepare features
        features = prepare_features(team1, team2, venue, match_date)
        
        # Encode categorical features
        try:
            features = encode_features(features)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
        # Prepare model input
        model_input = []
        for col in FEATURE_COLS:
            model_input.append(features[col])
        
        X_input = np.array([model_input], dtype=float)
        
        # Make prediction
        prediction = model.predict(X_input)
        prediction_proba = model.predict_proba(X_input)
        
        # Get predicted winner
        predicted_winner = target_encoder.inverse_transform(prediction)[0]
        
        # Get prediction confidence
        max_proba = float(np.max(prediction_proba))
        
        # Prepare response
        response = {
            'predicted_winner': str(predicted_winner),
            'model_confidence': round(max_proba * 100, 2),
            'team1_win_percentage': features['team1_win_percentage'],
            'team2_win_percentage': features['team2_win_percentage'],
            'team1_last5_results': features['team1_last5_results'],
            'team2_last5_results': features['team2_last5_results'],
            'last5_h2h_results': features['last5_h2h_results'],
            'team1_ground_stats': {
                'venue': venue,
                'matches_played': features['team1_ground_matches'],
                'wins': features['team1_ground_wins'],
                'losses': features['team1_ground_losses'],
                'win_percentage': round((features['team1_ground_wins'] / features['team1_ground_matches'] * 100) if features['team1_ground_matches'] > 0 else 0, 2)
            },
            'team2_ground_stats': {
                'venue': venue,
                'matches_played': features['team2_ground_matches'],
                'wins': features['team2_ground_wins'],
                'losses': features['team2_ground_losses'],
                'win_percentage': round((features['team2_ground_wins'] / features['team2_ground_matches'] * 100) if features['team2_ground_matches'] > 0 else 0, 2)
            },
            'head_to_head_stats': {
                'total_matches': features['total_h2h_matches'],
                'team1_wins': features['team1_h2h_wins'],
                'team2_wins': features['team2_h2h_wins'],
                'team1_h2h_percentage': round((features['team1_h2h_wins'] / features['total_h2h_matches'] * 100) if features['total_h2h_matches'] > 0 else 0, 2),
                'team2_h2h_percentage': round((features['team2_h2h_wins'] / features['total_h2h_matches'] * 100) if features['total_h2h_matches'] > 0 else 0, 2)
            },
            'recent_form_2024': {
                'team1_wins_in_last_5': features['team1_recent_wins_5'],
                'team1_matches_in_last_5': features['team1_recent_matches_5'],
                'team1_recent_win_percentage': round((features['team1_recent_wins_5'] / features['team1_recent_matches_5'] * 100) if features['team1_recent_matches_5'] > 0 else 0, 2),
                'team2_wins_in_last_5': features['team2_recent_wins_5'],
                'team2_matches_in_last_5': features['team2_recent_matches_5'],
                'team2_recent_win_percentage': round((features['team2_recent_wins_5'] / features['team2_recent_matches_5'] * 100) if features['team2_recent_matches_5'] > 0 else 0, 2)
            }
        }
        
        # Ensure all values are JSON serializable
        response = convert_to_serializable(response)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/teams', methods=['GET'])
def get_teams():
    """Get list of available teams"""
    try:
        teams = sorted(list(encoders['team1'].classes_))
        return jsonify({'teams': teams})
    except Exception as e:
        return jsonify({'error': f'Error getting teams: {str(e)}'}), 500

@app.route('/venues', methods=['GET'])
def get_venues():
    """Get list of available venues"""
    try:
        venues = sorted(list(encoders['venue'].classes_))
        return jsonify({'venues': venues})
    except Exception as e:
        return jsonify({'error': f'Error getting venues: {str(e)}'}), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get information about the model"""
    try:
        return jsonify({
            'model_accuracy': round(metadata['model_accuracy'], 4),
            'season_year': metadata['season_year'],
            'total_teams': len(metadata['teams']),
            'total_venues': len(metadata['venues']),
            'feature_count': len(metadata['feature_cols'])
        })
    except Exception as e:
        return jsonify({'error': f'Error getting model info: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Unhandled exception: {e}")
    print(traceback.format_exc())
    return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/')
def home():
    return jsonify({
        'message': 'IPL Winner Predictor API is live!',
        'endpoints': {
            '/predict': 'POST - Predict match winner',
            '/teams': 'GET - Get available teams',
            '/venues': 'GET - Get available venues', 
            '/model-info': 'GET - Get model information',
            '/health': 'GET - Health check'
        },
        'example_request': {
            'team1': 'Mumbai Indians',
            'team2': 'Chennai Super Kings',
            'venue': 'Wankhede Stadium',
            'date': '2024-05-01'
        }
    })

if __name__ == '__main__':
    print("Starting IPL Prediction API...")
    print(f"Model accuracy: {metadata['model_accuracy']:.4f}")
    print(f"Available teams: {len(metadata['teams'])}")
    print(f"Available venues: {len(metadata['venues'])}")
    app.run(host='0.0.0.0', port=5000, debug=True)
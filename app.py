from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load('ipl_model.pkl')
encoders = joblib.load('encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')

FEATURE_COLS = [
    'team1', 'team2', 'venue',
    'team1_recent_wins_5', 'team2_recent_wins_5',
    'team1_ground_winrate', 'team2_ground_winrate',
    'team1_h2h_wins', 'team2_h2h_wins'
]

def prepare_input(data):
    row_enc = data.copy()
    for col in ['team1', 'team2', 'venue']:
        try:
            row_enc[col] = encoders[col].transform([data[col]])[0]
        except Exception:
            raise ValueError(f"Unknown value '{data[col]}' for '{col}'. Please ensure it exists in the training data.")
    arr = np.array([[row_enc[col] for col in FEATURE_COLS]], dtype=float)
    return arr

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        X_input = prepare_input(data)
        pred = model.predict(X_input)
        winner = target_encoder.inverse_transform(pred)[0]
        response = {
            "predicted_winner": winner,
            "team1_last5_matches": data.get('team1_last5_matches', []),
            "team2_last5_matches": data.get('team2_last5_matches', []),
            "last_5_h2h_matches": data.get('last_5_h2h_matches', [])
        }
        return jsonify(response)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "IPL Winner API is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
# Save the target encoder
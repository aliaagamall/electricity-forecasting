from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor import ElectricityPredictor
from config import Config

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access if needed

# Initialize predictor
predictor = ElectricityPredictor()

def prepare_data_from_request(data):
    """Helper function to load and prepare data from file path in request."""
    file_path = data.get('file_path', Config.test_data_path)
    predictor.load_and_prepare_data(file_path)

@app.route('/')
def index():
    """Health check route."""
    return jsonify({'message': 'Electricity Predictor API is running'})

@app.route('/api/predict/next', methods=['POST'])
def predict_next():
    """
    Predict the next electricity value.
    Request JSON: { "file_path": "path/to/file.csv" }
    """
    try:
        data = request.get_json()
        prepare_data_from_request(data)
        prediction = predictor.predict_next()
        return jsonify({'prediction': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict/multiple', methods=['POST'])
def predict_multiple():
    """
    Predict multiple future electricity values.
    Request JSON: { "file_path": "path/to/file.csv", "n_steps": 5 }
    """
    try:
        data = request.get_json()
        prepare_data_from_request(data)
        n_steps = data.get('n_steps', 5)
        predictions = predictor.predict_multiple(n_steps)
        return jsonify({'predictions': [float(p) for p in predictions]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/explain', methods=['POST'])
def explain():
    """
    Explain the prediction using SHAP or other method.
    Request JSON: { "file_path": "path/to/file.csv" }
    """
    try:
        data = request.get_json()
        prepare_data_from_request(data)
        explanation = predictor.explain_prediction()
        # Convert explanation values to JSON-friendly format
        return jsonify({
            'explanation': {
                'values': explanation.values.tolist(),
                'base_value': explanation.base_values,
                'features': explanation.feature_names,
                'data': explanation.data.tolist()
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the generated model and scaler files
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    """Renders the landing page"""
    return render_template('home.html')

@app.route('/predict_page')
def predict_page():
    """Renders the professional 9-field input form"""
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    """Renders the Market Insights page"""
    return render_template('adaptivity.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Processes only the 4 features recognized by the model to avoid scaling errors"""
    try:
        # We extract ONLY the 4 features your original scaler expects:
        # 1. Number of Milestones
        # 2. Number of Relationships
        # 3. Number of Funding Rounds
        # 4. Total Funding (in USD)
        features_to_scale = [
            float(request.form['milestones']),
            float(request.form['relationships']),
            float(request.form['funding_rounds']),
            float(request.form['funding_total_usd'])
        ]
        
        # Scale only these 4 features
        scaled_features = scaler.transform([features_to_scale])
        
        # Make prediction (1 for Acquired, 0 for Closed)
        prediction = model.predict(scaled_features)
        
        # Map prediction to text result
        result_text = 'ACQUIRED' if prediction[0] == 1 else 'CLOSED'
        
        # Return result to results.html
        return render_template('results.html', result=result_text)
    
    except Exception as e:
        return f"Error during prediction: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
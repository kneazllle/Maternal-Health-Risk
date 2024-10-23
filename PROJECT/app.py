from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Create Flask app
app = Flask(__name__)

# Load the best model, scaler, and polynomial features
model = joblib.load('best_maternal_health_model.pkl')
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly_features.pkl')  # Load the PolynomialFeatures object

# Define the risk levels
risk_levels = {0: 'low risk', 1: 'mid risk', 2: 'high risk'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.form.to_dict()

        # Convert the input data to a DataFrame
        input_df = pd.DataFrame(data, index=[0])

        # Convert numeric columns to appropriate types
        input_df['Age'] = pd.to_numeric(input_df['Age'], errors='coerce')
        input_df['SystolicBP'] = pd.to_numeric(input_df['SystolicBP'], errors='coerce')
        input_df['DiastolicBP'] = pd.to_numeric(input_df['DiastolicBP'], errors='coerce')
        input_df['BS'] = pd.to_numeric(input_df['BS'], errors='coerce')
        input_df['BodyTemp'] = pd.to_numeric(input_df['BodyTemp'], errors='coerce')
        input_df['HeartRate'] = pd.to_numeric(input_df['HeartRate'], errors='coerce')

        # Display the input data for debugging
        print("Input Data:")
        print(input_df)

        # Check for missing values
        if input_df.isnull().values.any():
            return render_template('index.html', risk_level=None, advice="Please enter valid numeric values for all fields.")

        # Feature scaling
        scaled_input = scaler.transform(input_df)

        # Generate polynomial features
        scaled_input_poly = poly.transform(scaled_input)

        # Make predictions using the model
        predictions = model.predict(scaled_input_poly)

        # Log the predictions
        print("Predictions:", predictions)

        # Decode the predicted risk level
        predicted_risk = risk_levels[predictions[0]]

        print(f"\nPredicted Risk Level: {predicted_risk}")

        # Set advice based on predicted risk
        advice = "Consult your healthcare provider for personalized advice." if predicted_risk != 'low risk' else "Keep up the good health!"

        return render_template('index.html', risk_level=predicted_risk, advice=advice)

    except Exception as e:
        return render_template('index.html', risk_level=None, advice="An error occurred: {}".format(str(e)))

if __name__ == '__main__':
    app.run(debug=True)

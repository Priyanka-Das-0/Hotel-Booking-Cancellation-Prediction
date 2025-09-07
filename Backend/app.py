import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
import sagemaker
import boto3
import json

# Initialize the flask app
app = Flask(__name__, template_folder='../Frontend' )


# --- SageMaker Endpoint Configuration ---
try:
    # Use the same region as your SageMaker training and endpoint
    region = 'us-east-1'
    boto3_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.Session(boto_session=boto3_session)
    
    # Replace with your actual endpoint name
    endpoint_name = "hotel-booking-model-2025-09-07-02-50-32" 
    predictor = sagemaker.predictor.Predictor(
        endpoint_name=endpoint_name, 
        sagemaker_session=sagemaker_session,
        serializer=sagemaker.serializers.JSONSerializer(),
        deserializer=sagemaker.deserializers.JSONDeserializer()
    )
    print("SageMaker predictor configured successfully.")
except Exception as e:
    print(f"Error configuring SageMaker predictor: {e}")
    predictor = None

# --- Define web routes ---

@app.route('/')
def home():
    """Renders the main prediction page."""
    return render_template('index.html')

@app.route('/visualizations')
def visualizations():
    """Renders the EDA visualizations page."""
    return render_template('visualizations.html')

@app.route('/batch', methods=['GET', 'POST'])
def batch_predict():
    """Handles file upload and batch prediction."""
    if predictor is None:
        flash('SageMaker endpoint is not configured. Check server logs.', 'danger')
        return redirect(request.url)

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)
                original_df = df.copy()

                # Send the raw data to the SageMaker endpoint
                # The endpoint's inference.py script handles all preprocessing
                payload = original_df.to_dict('records')
                predictions_response = predictor.predict(payload)

                # The response is a list of predictions and probabilities
                predictions = np.array(predictions_response['predictions'])
                probabilities = np.array(predictions_response['probabilities'])

                original_df['Predicted Cancellation'] = ['Yes' if p == 1 else 'No' for p in predictions]
                original_df['Cancellation Probability'] = [f"{p*100:.2f}%" for p in probabilities]
                total_bookings = len(original_df)
                total_cancellations = np.sum(predictions)
                
                # Use a month map that includes all months to avoid errors
                month_map_to_name = {
                    1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 
                    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
                }
                
                monthly_cancellations = original_df[original_df['Predicted Cancellation'] == 'Yes']['arrival_date_month'].map(month_map_to_name.get).value_counts().to_dict()
                
                recommendation = "Low cancellation risk detected. Standard operational procedures are likely sufficient."
                if total_bookings > 0 and total_cancellations / total_bookings > 0.5:
                    recommendation = "High cancellation risk! Consider implementing a strategic overbooking policy for high-risk months and focus on guest communication to confirm bookings."
                elif total_bookings > 0 and total_cancellations / total_bookings > 0.2:
                    recommendation = "Moderate cancellation risk. It is advisable to send confirmation emails to guests with high cancellation probability and review booking policies for flexibility."

                results_html = original_df[['hotel', 'arrival_date_month', 'lead_time', 'adr', 'Predicted Cancellation', 'Cancellation Probability']].to_html(classes='table table-striped table-hover', index=False)
                
                return render_template('batch.html', 
                                       results_html=results_html, 
                                       total_bookings=total_bookings,
                                       total_cancellations=total_cancellations,
                                       monthly_cancellations=monthly_cancellations,
                                       recommendation=recommendation)
            except Exception as e:
                flash(f'An error occurred: {e}', 'danger')
                return redirect(request.url)

    return render_template('batch.html', results_html=None)


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the single prediction request from the form."""
    if predictor is None:
        return jsonify({'error': 'SageMaker endpoint is not configured. Check server logs.'}), 500
    try:
        data = request.get_json(force=True)
        query_df = pd.DataFrame([data])
        
        # Send raw JSON to the endpoint.
        # The endpoint's inference.py handles all preprocessing.
        predictions_response = predictor.predict(query_df.to_dict('records'))
        
        # The response is a list of predictions and probabilities
        prediction = predictions_response['predictions'][0]
        probabilities = predictions_response['probabilities'][0]
        
        if prediction == 1:
            output = "Booking will be Cancelled"
            probability = f"{probabilities*100:.2f}%"
        else:
            output = "Booking will not be Cancelled"
            probability = f"{probabilities*100:.2f}%"
            
        return jsonify({'prediction_text': output, 'probability': probability})
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

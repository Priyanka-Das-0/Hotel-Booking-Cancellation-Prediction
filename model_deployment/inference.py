import json
import joblib
import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

def model_fn(model_dir):
    """
    Loads the trained model from disk.
    """
    try:
        model = joblib.load(os.path.join(model_dir, "model.joblib"))
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file 'model.joblib' not found in {model_dir}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

def preprocess_data(df):
    """
    Applies the same preprocessing steps from training to the input dataframe.
    """
    df_processed = df.copy()

    # Drop columns that were dropped during training.
    # The 'errors' parameter ensures the script doesn't crash if a column is missing.
    columns_to_drop = [
        'days_in_waiting_list', 'arrival_date_year', 'booking_changes',
        'reservation_status', 'country', 'reservation_status_date',
        'reserved_room_type', 'assigned_room_type'
    ]
    df_processed = df_processed.drop(columns_to_drop, axis=1, errors='ignore')

    # Fill missing values (consistent with training script)
    df_processed['company'] = df_processed.get('company', pd.Series(dtype=object)).fillna(0)
    df_processed['agent'] = df_processed.get('agent', pd.Series(dtype=object)).fillna(0)
    df_processed['children'] = df_processed.get('children', pd.Series(dtype=object)).fillna(0)

    # Use .get() with an empty Series as a default to handle potentially missing columns
    for col, mappings in {
        'hotel': {'Resort Hotel': 0, 'City Hotel': 1},
        'arrival_date_month': {
            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
            'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
        },
        'meal': {'BB': 0, 'FB': 1, 'HB': 2, 'SC': 3, 'Undefined': 4},
        'market_segment': {
            'Direct': 0, 'Corporate': 1, 'Online TA': 2, 'Offline TA/TO': 3, 'Complementary': 4,
            'Groups': 5, 'Undefined': 6, 'Aviation': 7
        },
        'distribution_channel': {
            'Direct': 0, 'Corporate': 1, 'TA/TO': 2, 'Undefined': 3, 'GDS': 4
        },
        'deposit_type': {'No Deposit': 0, 'Refundable': 1, 'Non Refund': 2},
        'customer_type': {
            'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3
        }
    }.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(mappings).fillna(-1) # Use -1 to denote unknown category
    
    # Handle room_match, which is now expected to be sent directly
    if 'room_match' not in df_processed.columns:
        df_processed['room_match'] = 0

    df_processed = df_processed.fillna(0)
    
    log_cols = ['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month', 'agent', 'company', 'adr']
    for col in log_cols:
        if col in df_processed.columns:
            df_processed[col] = np.log1p(df_processed[col].astype(float))
            
    # Reindex to ensure the exact feature order and number
    model_columns = [
        'hotel', 'lead_time', 'arrival_date_month', 'arrival_date_week_number', 
        'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights', 
        'adults', 'children', 'babies', 'meal', 'market_segment', 
        'distribution_channel', 'is_repeated_guest', 'previous_cancellations', 
        'previous_bookings_not_canceled', 'deposit_type', 'agent', 'company', 
        'customer_type', 'adr', 'required_car_parking_spaces', 
        'total_of_special_requests', 'room_match'
    ]
    
    df_processed = df_processed.reindex(columns=model_columns, fill_value=0)

    return df_processed

def input_fn(request_body, request_content_type):
    """
    Deserializes the request body to a pandas DataFrame.
    """
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        df = pd.DataFrame(data)
        return preprocess_data(df)
    else:
        raise ValueError("This model only supports 'application/json' input.")

def predict_fn(input_data, model):
    """
    Makes predictions on the preprocessed input data and returns probabilities.
    """
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]
    
    return {'predictions': predictions.tolist(), 'probabilities': probabilities.tolist()}

def output_fn(prediction, accept):
    """
    Serializes the prediction output to JSON.
    """
    if accept == "application/json":
        return json.dumps(prediction), accept
    else:
        raise ValueError("This model only supports 'application/json' output.")

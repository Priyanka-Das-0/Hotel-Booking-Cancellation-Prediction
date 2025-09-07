# script.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import argparse
from sklearn.metrics import accuracy_score

def get_data(train_dir, test_dir):
    # Load training data
    train_df = pd.read_csv(os.path.join(train_dir, 'train.csv'))
    
    # Load testing data
    test_df = pd.read_csv(os.path.join(test_dir, 'test.csv'))
    
    return train_df, test_df

def preprocess_data(df):
    # (Same preprocessing function as before)
    useless_cols = ['days_in_waiting_list', 'arrival_date_year',
                    'booking_changes', 'reservation_status', 'country', 'reservation_status_date']
    df_ml = df.drop(useless_cols, axis=1)

    # Manual mapping for categorical features
    df_ml['hotel'] = df_ml['hotel'].map({'Resort Hotel': 0, 'City Hotel': 1})
    df_ml['arrival_date_month'] = df_ml['arrival_date_month'].map({'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12})
    df_ml['meal'] = df_ml['meal'].map({'BB': 0, 'FB': 1, 'HB': 2, 'SC': 3, 'Undefined': 4})
    df_ml['market_segment'] = df_ml['market_segment'].map({'Direct': 0, 'Corporate': 1, 'Online TA': 2, 'Offline TA/TO': 3, 'Complementary': 4, 'Groups': 5, 'Undefined': 6, 'Aviation': 7})
    df_ml['distribution_channel'] = df_ml['distribution_channel'].map({'Direct': 0, 'Corporate': 1, 'TA/TO': 2, 'Undefined': 3, 'GDS': 4})
    df_ml['reserved_room_type'] = df_ml['reserved_room_type'].map({'C': 0, 'A': 1, 'D': 2, 'E': 3, 'G': 4, 'F': 5, 'H': 6, 'L': 7, 'B': 8, 'P': 9})
    df_ml['assigned_room_type'] = df_ml['assigned_room_type'].map({'C': 0, 'A': 1, 'D': 2, 'E': 3, 'G': 4, 'F': 5, 'H': 6, 'L': 7, 'B': 8, 'P': 9})
    df_ml['deposit_type'] = df_ml['deposit_type'].map({'No Deposit': 0, 'Refundable': 1, 'Non Refund': 2})
    df_ml['customer_type'] = df_ml['customer_type'].map({'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3})
    df_ml['room_match'] = (df_ml['reserved_room_type'] == df_ml['assigned_room_type']).astype(int)
    
    df_ml = df_ml.drop(['reserved_room_type', 'assigned_room_type'], axis=1)
    df_ml = df_ml.fillna(0)
    
    log_cols = ['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month', 'agent', 'company', 'adr']
    for col in log_cols:
        df_ml[col] = np.log1p(df_ml[col])
        
    return df_ml

def train(args):
    # Get the directory paths for training and testing data
    training_path = args.get('train')
    testing_path = args.get('test')
    
    # Load and preprocess data
    train_df, test_df = get_data(training_path, testing_path)
    
    X_train = preprocess_data(train_df).drop('is_canceled', axis=1)
    y_train = preprocess_data(train_df)['is_canceled']
    
    X_test = preprocess_data(test_df).drop('is_canceled', axis=1)
    y_test = preprocess_data(test_df)['is_canceled']
    
    # Train XGBoost model
    xgb = XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=957,
        max_depth=9,
        learning_rate=0.03862906070110879,
        subsample=0.7900614800261079,
        colsample_bytree=0.6833199952159132,
        min_child_weight=1,
        gamma=0.04046969783495924,
        reg_lambda=0.006033407638970128,
        reg_alpha=0.19586166755735193
    )
    
    xgb.fit(X_train, y_train)
    
    # Evaluate the model on the test data
    y_pred = xgb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on the test set: {accuracy}")
    
    # Save the model
    joblib.dump(xgb, os.path.join(args['model_dir'], 'model.joblib'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    args = parser.parse_args()
    
    train(vars(args))
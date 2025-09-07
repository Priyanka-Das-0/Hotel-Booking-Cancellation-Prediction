# In a Jupyter Notebook cell

# 1. Import Libraries
import pandas as pd
import numpy as np
import warnings
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# 2. Load Data
df = pd.read_csv('./hotel_bookings.csv')
# 3. Initial Cleaning
# Fill missing values
df['company'] = df['company'].fillna(0)
df['agent'] = df['agent'].fillna(0)
df['country'] = df['country'].fillna('Undefined')
df['children'] = df['children'].fillna(0)

# Remove bookings with zero adults, children, and babies
zero_guest_filter = (df['adults'] == 0) & (df['children'] == 0) & (df['babies'] == 0)
df = df[~zero_guest_filter]

print("Initial data cleaning complete.")

# 4. Feature Engineering & Preprocessing
# Drop columns not useful for prediction
useless_cols = ['days_in_waiting_list', 'arrival_date_year',
                'booking_changes', 'reservation_status', 'country', 'reservation_status_date']
df_ml = df.drop(useless_cols, axis=1)

# Separate categorical and numerical columns
cat_cols = [col for col in df_ml.columns if df_ml[col].dtype == 'O']
num_cols = [col for col in df_ml.columns if df_ml[col].dtype != 'O']

# Manual mapping for categorical features (as you specified)
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

# Drop the original reserved and assigned room columns
df_ml = df_ml.drop(['reserved_room_type', 'assigned_room_type'], axis=1)
print("Categorical variables encoded.")

# Handle any remaining NaNs after mapping (e.g., if a category was missed)
df_ml = df_ml.fillna(0) # Simple strategy for any leftovers

# Apply log transformation to reduce skewness
log_cols = ['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month', 'agent', 'company', 'adr']
for col in log_cols:
    df_ml[col] = np.log1p(df_ml[col]) # Use np.log1p which is log(x+1)

print("Numerical variables transformed.")

# 5. Model Training
# Define features (X) and target (y)
X = df_ml.drop('is_canceled', axis=1)
y = df_ml['is_canceled']

# Save the column list
# This is CRITICAL for the Flask app to process inputs correctly


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train XGBoost Classifier
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

# Evaluate model
y_pred = xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
from sklearn.metrics import roc_auc_score,classification_report

# Predict probabilities for the positive class
y_pred_proba = xgb.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC-AUC Score:", roc_auc)
# Metrics
print("\n[Classification Report @ 0.5 threshold]")
print(classification_report(y_test, y_pred))
# 6. Save the Trained Model
joblib.dump(xgb, './model.joblib')
print("Model saved to 'model.joblib'")
joblib.dump(X.columns, './model_columns.pkl')
print("Model columns saved to 'model_columns.pkl'")



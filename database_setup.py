# database_setup.py
import pandas as pd
import sqlite3
import os

print("--- Starting Database Setup ---")

# Define paths
DATABASE_FOLDER = 'database'
DATABASE_PATH = os.path.join(DATABASE_FOLDER, 'trends.db')
DATASET_PATH = 'hotel_bookings.csv' # Make sure this is in your root folder

# Create database folder if it doesn't exist
if not os.path.exists(DATABASE_FOLDER):
    os.makedirs(DATABASE_FOLDER)

# Load the dataset
try:
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: {DATASET_PATH} not found. Please place it in the root directory.")
    exit()

# --- Prepare data for trends ---
# Convert 'arrival_date_month' to a categorical type with a defined order
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
df['arrival_date_month'] = pd.Categorical(df['arrival_date_month'], categories=month_order, ordered=True)

# Create a 'day_of_week' column
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
df['day_of_week'] = df['reservation_status_date'].dt.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=day_order, ordered=True)


# --- Aggregate data ---
# 1. Monthly Cancellations
monthly_cancellations = df[df['is_canceled'] == 1].groupby('arrival_date_month')['is_canceled'].count().reset_index()
monthly_cancellations.rename(columns={'is_canceled': 'cancellation_count'}, inplace=True)
print("Aggregated monthly cancellations.")

# 2. Weekly Cancellations (by day of week)
weekly_cancellations = df[df['is_canceled'] == 1].groupby('day_of_week')['is_canceled'].count().reset_index()
weekly_cancellations.rename(columns={'is_canceled': 'cancellation_count'}, inplace=True)
print("Aggregated weekly cancellations.")


# --- Store in SQLite Database ---
try:
    conn = sqlite3.connect(DATABASE_PATH)
    print(f"Connected to SQLite database at {DATABASE_PATH}")

    # Save the aggregated data to tables
    monthly_cancellations.to_sql('monthly_trends', conn, if_exists='replace', index=False)
    weekly_cancellations.to_sql('weekly_trends', conn, if_exists='replace', index=False)
    
    print("Tables 'monthly_trends' and 'weekly_trends' created and populated successfully.")

    conn.close()
    print("Database connection closed.")
except Exception as e:
    print(f"An error occurred: {e}")

print("--- Database Setup Complete ---")
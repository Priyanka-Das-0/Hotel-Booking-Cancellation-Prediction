import sagemaker
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from time import gmtime, strftime
from dotenv import load_dotenv
import os
import sagemaker

# Load variables from .env file
load_dotenv()

# Create SageMaker session
sagemaker_session = sagemaker.Session()

# Read values
bucket = os.getenv("SAGEMAKER_BUCKET")
role = os.getenv("SAGEMAKER_ROLE")

sagemaker_session = sagemaker.Session()
# 2. DATA PREPARATION (UPLOAD SEPARATE TRAIN/TEST DATA)
# =======================================================
# Load the raw data
data_path = 'hotel_bookings.csv'
df = pd.read_csv(data_path)

# Perform the train-test split locally
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the split data to temporary CSV files
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

# Upload the separate datasets to S3
s3_prefix = 'sagemaker/hotel_booking_cancellation'
s3_train_location = sagemaker_session.upload_data(path='train.csv', bucket=bucket, key_prefix=f'{s3_prefix}/train')
s3_test_location = sagemaker_session.upload_data(path='test.csv', bucket=bucket, key_prefix=f'{s3_prefix}/test')

print(f"Uploaded training data to S3 at: {s3_train_location}")
print(f"Uploaded testing data to S3 at: {s3_test_location}")

# 3. TRAINING
# =======================================
FRAMEWORK_VERSION = "0.23-1"
sklearn_estimator = SKLearn(
    entry_point="model_deployment/script.py", 
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    framework_version=FRAMEWORK_VERSION,
    py_version="py3", # Use a Python version compatible with XGBoost
    dependencies=['model_deployment/requirements.txt'], # Add this line
    base_job_name="Hotel-Booking-XGBoost",
    use_spot_instances=True,
    max_wait=7200,
    max_run=3600,
    sagemaker_session=sagemaker_session
)

# This command runs your script.py on a SageMaker instance
# Pass both S3 locations to the estimator
sklearn_estimator.fit({'train': s3_train_location, 'test': s3_test_location})
# ... (rest of the script)

# =======================================
model_name = "hotel-booking-model-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
endpoint_name = model_name

model = SKLearnModel(
    model_data=sklearn_estimator.model_data,
    role=role,
    entry_point="model_deployment/inference.py",
    framework_version=FRAMEWORK_VERSION,
    py_version="py3", # You can also specify py_version here
    dependencies=['model_deployment/requirements.txt'], # <-- Add this line
    sagemaker_session=sagemaker_session,
    name=model_name
)


predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name=endpoint_name
)

print("Endpoint deployed at:", endpoint_name)


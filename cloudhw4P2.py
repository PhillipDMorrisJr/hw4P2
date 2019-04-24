# [START setup]
import datetime
import os
import subprocess
import sys
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

# Fill in your Cloud Storage bucket name
BUCKET_NAME = 'homework4-237718-mlengine'
# [END setup]

# [START download-data]
kiddcupp_data_filename = 'kddcup.txt'
kiddcupp_target_filename = 'kiddcup_class.txt'
data_dir = 'gs://cloud-samples-data/homework4-237718-mlengine/kiddcup'

# [START load-into-pandas]
kiddcupp_data= pd.read_csv(kiddcupp_data_filename).values
kiddcupp_target = pd.read_csv(kiddcupp_target_filename).values

# Convert one-column 2D array into 1D array for use with XGBoost
kiddcupp_target = kiddcupp_target.reshape((kiddcupp_target.size,))
# [END load-into-pandas]
le.fit(kiddcupp_target)
print("original: ")
print(list(le.classes_))

kiddcupp_target = le.transform(kiddcupp_target)
print("new: ")
print(kiddcupp_target)

# [START train-and-save-model]
# Load data into DMatrix object
dtrain = xgb.DMatrix(kiddcupp_data, label=kiddcupp_target)

# Train XGBoost model
bst = xgb.train({}, dtrain, 20)

# Export the classifier to a file
model_filename = 'model.bst'
bst.save_model(model_filename)
# [END train-and-save-model]

# [START upload-model]
# Upload the saved model file to Cloud Storage
gcs_model_path = os.path.join('gs://', BUCKET_NAME,
    datetime.datetime.now().strftime('kdd_%Y%m%d_%H%M%S'), model_filename)
subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path],
    stderr=sys.stdout)
# [END upload-model]

import csv
import numpy as np
from sklearn.ensemble import IsolationForest

# Load the data from the CSV file
with open('output.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row
    data = list(reader)

# Convert the data to a NumPy array
data = np.array(data)

# Split the data into features and labels
X = data[:, 1].astype(float).reshape(-1, 1)
y = data[:, 0]

# Create an Isolation Forest model
model = IsolationForest()

# Fit the model on the data
model.fit(X)

# Make predictions on the data
predictions = model.predict(X)

# Iterate over the predictions
for i, prediction in enumerate(predictions):
    # Check if the prediction is -1 (outlier)
    if prediction == -1:
        # Get the IP address and load of the outlier
        ip_address = y[i]
        load = X[i][0]
        
        # Print the details of the outlier
        print(f'Abnormal load detected: IP Address = {ip_address}, Load = {load}')

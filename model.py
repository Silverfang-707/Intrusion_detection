import pandas as pd
from sklearn.ensemble import IsolationForest

# Load the data from the CSV file
data = pd.read_csv('output.csv', names=['timestamp', 'ip_address', 'load'])
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Group the data by timestamp and calculate the total load for each timestamp
data = data.groupby('timestamp')['load'].sum().reset_index()

# Extract the load values as a numpy array
X = data['load'].values.reshape(-1, 1)

# Train an Isolation Forest model to detect outliers
model = IsolationForest(contamination=0.01)
model.fit(X)

# Use the model to predict outliers
data['outlier'] = model.predict(X)

# Print the timestamps where abnormal load was detected
outliers = data[data['outlier'] == -1]
print(outliers['timestamp'])
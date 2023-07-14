import csv
import numpy as np
import os
import psutil
import time
from collections import Counter
from sklearn.ensemble import IsolationForest

def generate_csv():
    # Get the list of all network connections
    connections = psutil.net_connections()

    # Create a list to store the data
    data = []

    # Iterate over the connections
    for conn in connections:
        # Get the IP address and load of the connected device
        ip_address = conn.laddr.ip
        load = psutil.cpu_percent()
        
        # Append the data to the list
        data.append([ip_address, load])

    # Check if the output.csv file exists
    if os.path.exists('output.csv'):
        # Open the file in append mode
        f = open('output.csv', 'a', newline='')
        writer = csv.writer(f)
    else:
        # Create a new file and write the header row
        f = open('output.csv', 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['IP Address', 'Load'])

    # Write the data to the CSV file
    writer.writerows(data)
    f.close()

def run_ml_model():
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

    # Check if the abnormal_loads.csv file exists
    if os.path.exists('abnormal_loads.csv'):
        # Open the file in append mode
        f = open('abnormal_loads.csv', 'a', newline='')
        writer = csv.writer(f)
    else:
        # Create a new file and write the header row
        f = open('abnormal_loads.csv', 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['IP Address', 'Load'])

    # Iterate over the predictions
    for i, prediction in enumerate(predictions):
        # Check if the prediction is -1 (outlier)
        if prediction == -1:
            # Get the IP address and load of the outlier
            ip_address = y[i]
            load = X[i][0]
            
            # Write the details of the outlier to the file
            writer.writerow([ip_address, load])

    f.close()

def summarize_abnormal_loads():
    # Load the data from the abnormal_loads.csv file
    with open('abnormal_loads.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        ip_addresses = [row[0] for row in reader]

    # Count the occurrences of each IP address
    counts = Counter(ip_addresses)

    # Open the abnormal_loads_summary.txt file in write mode
    with open('abnormal_loads_summary.txt', 'w') as f:
        # Iterate over the counts
        for ip_address, count in counts.items():
            # Write the IP address and count to the file
            f.write(f'{ip_address}: {count}\n')

while True:
    generate_csv()
    run_ml_model()
    summarize_abnormal_loads()
    time.sleep(1)

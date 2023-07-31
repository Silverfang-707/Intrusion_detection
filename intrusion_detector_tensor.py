import csv
import numpy as np
import os
import psutil
import tensorflow as tf
import time
from collections import Counter

def generate_csv(file_name='output.csv'):
    # Get the list of all network connections
    connections = psutil.net_connections()

    # Get the system's network traffic, memory usage, and disk usage
    net_io_counters = psutil.net_io_counters()
    virtual_memory = psutil.virtual_memory()
    disk_usage = psutil.disk_usage('/')

    # Create a list to store the data
    data = []

    # Iterate over the connections
    for conn in connections:
        # Get the IP address and load of the connected device
        ip_address = conn.laddr.ip
        load = psutil.cpu_percent()
        bytes_sent = net_io_counters.bytes_sent
        bytes_recv = net_io_counters.bytes_recv
        packets_sent = net_io_counters.packets_sent
        packets_recv = net_io_counters.packets_recv
        used_memory = virtual_memory.used
        used_disk = disk_usage.used
        
        # Append the data to the list
        data.append([ip_address, load, bytes_sent, bytes_recv, packets_sent, packets_recv, used_memory, used_disk])

    # Check if the file exists
    if os.path.exists(file_name):
        # Open the file in append mode
        f = open(file_name, 'a', newline='')
        writer = csv.writer(f)
    else:
        # Create a new file and write the header row
        f = open(file_name, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['IP Address', 'Load', 'Bytes Sent', 'Bytes Received', 'Packets Sent', 'Packets Received', 'Used Memory', 'Used Disk'])

    # Write the data to the CSV file
    writer.writerows(data)
    f.close()

def run_ml_model(input_file='output.csv', output_file='abnormal_loads.csv', threshold=0.5):
    # Check if the model already exists
    if os.path.exists('model.h5'):
        # Load the model from disk
        model = tf.keras.models.load_model('model.h5')
    else:
        # Load the data from the CSV file
        with open(input_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            data = list(reader)

        # Convert the data to a NumPy array
        data = np.array(data)

        # Split the data into features and labels
        X = data[:, 1:].astype(float)
        y = data[:, 0]

        # Create a TensorFlow Sequential model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model with binary crossentropy loss and Adam optimizer
        model.compile(loss='binary_crossentropy', optimizer='adam')

        # Fit the model on the data (assuming normal loads are labeled as 0 and abnormal loads are labeled as 1)
        model.fit(X, np.zeros(X.shape[0]))

    # Fit the model on the new data (assuming normal loads are labeled as 0 and abnormal loads are labeled as 1)
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        data = list(reader)

    # Convert the data to a NumPy array
    data = np.array(data)

    # Split the data into features and labels
    X = data[:, 1:].astype(float)
    y = data[:, 0]

    # Update the pre-trained model with new data
    model.fit(X, np.zeros(X.shape[0]))

    # Save the updated model to disk
    model.save('model.h5')

    # Make predictions on the new data (predict returns the probability of each sample being an abnormal load)
    predictions = model.predict(X)[:, 0]

    # Write the predictions to a CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'prediction'])
        for i in range(len(predictions)):
            writer.writerow([y[i], predictions[i]])

def summarize_abnormal_loads(input_file='abnormal_loads.csv', output_file='abnormal_loads_summary.csv'):
    # Load the data from the CSV file
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        data = list(reader)

    # Count the number of times each IP address is labeled abnormal
    counts = Counter(row[0] for row in data if float(row[1]) > 0.5)

    # Write the results to a CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['IP Address', 'Count'])
        for ip_address, count in counts.items():
            writer.writerow([ip_address, count])

while True:
    generate_csv()
    run_ml_model()
    summarize_abnormal_loads()
    print("Iteration Done...")
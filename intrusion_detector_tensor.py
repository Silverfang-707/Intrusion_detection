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

    # Make predictions on the data (predict returns the probability of each sample being an abnormal load)
    predictions = model.predict(X)[:, 0]

    # Check if the output file exists
    if os.path.exists(output_file):
        # Open the file in append mode
        f = open(output_file, 'a', newline='')
        writer = csv.writer(f)
    else:
        # Create a new file and write the header row
        f = open(output_file, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['IP Address', 'Load', 'Bytes Sent', 'Bytes Received', 'Packets Sent', 'Packets Received', 'Used Memory', 'Used Disk'])

    # Iterate over the predictions
    for i, prediction in enumerate(predictions):
        # Check if the prediction is above the threshold (abnormal load)
        if prediction > threshold:
            # Get the details of the abnormal load
            ip_address, load, bytes_sent, bytes_recv, packets_sent, packets_recv, used_memory, used_disk = data[i]
            
            # Write the details of the abnormal load to the file
            writer.writerow([ip_address, load, bytes_sent, bytes_recv, packets_sent, packets_recv, used_memory, used_disk])

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
    print("Iteration Done...")
import csv
import psutil

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

# Write the data to a CSV file
with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['IP Address', 'Load'])
    writer.writerows(data)

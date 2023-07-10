import os
import csv
from datetime import datetime

def get_ip_addresses_and_load():
    stream = os.popen('netstat -ntu')
    output = stream.read()
    lines = output.split('\n')
    ip_addresses = {}
    for line in lines:
        if 'ESTABLISHED' in line:
            columns = line.split()
            ip_address = columns[4].split(':')[0]
            if ip_address not in ip_addresses:
                ip_addresses[ip_address] = 0
            ip_addresses[ip_address] += 1
    return ip_addresses

def append_to_csv(data, filename='output.csv'):
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total_load = sum(data.values())
        writer.writerow([timestamp, total_load])

data = get_ip_addresses_and_load()
append_to_csv(data)

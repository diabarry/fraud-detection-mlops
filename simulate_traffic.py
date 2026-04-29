import requests
import time
import random

url = "http://localhost:8000/predict"

# Sending 100 random requests to simulate traffic
for i in range(100):
    # Generate random PCA features V1 to V28
    data = {f"V{j}": random.uniform(-1, 1) for j in range(1, 29)}
    data["Amount"] = random.uniform(10, 500)

    # Send POST request to the prediction endpoint
    response = requests.post(url, json=data)
    print(f"Request {i}: {response.json()}")
    
    # Short pause to observe the metric curves rising in Grafana/Prometheus
    time.sleep(0.5)
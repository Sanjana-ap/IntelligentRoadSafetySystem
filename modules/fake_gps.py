# fake_gps.py
import requests, random, time

url = "http://127.0.0.1:5000/api/gps"  # match your Flask route

while True:
    lat = 12.97 + random.uniform(-0.01, 0.01)
    lon = 77.59 + random.uniform(-0.01, 0.01)

    payload = {"lat": lat, "lon": lon}  # match keys in your Flask code
    res = requests.post(url, json=payload)

    print("Sent:", payload, "Response:", res.json())
    time.sleep(5)

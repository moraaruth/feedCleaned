import os
import sys
import requests

URL = os.environ.get('PREDICT_URL', 'http://localhost:5000/predict')

payloads = [
    {"cluster": "SIM & SWAP"},
    {"text": "SIM swap and OTP failures reported by many users"}
]

required_keys = {"top_issue", "trend", "severity", "recommendation", "predicted_volume"}

for p in payloads:
    print(f"Testing payload: {p}")
    r = requests.post(URL, json=p, timeout=10)
    try:
        r.raise_for_status()
    except Exception as e:
        print('Request failed:', e)
        print('Status:', r.status_code)
        print('Body:', r.text)
        sys.exit(2)
    data = r.json()
    missing = required_keys - set(data.keys())
    if missing:
        print('Response missing keys:', missing)
        print('Full response:', data)
        sys.exit(3)
    print('OK:', {k: data[k] for k in required_keys})

print('All tests passed')

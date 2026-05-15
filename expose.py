#!/usr/bin/env python
"""Expose Flask app via ngrok tunnel"""
from pyngrok import ngrok

# Connect to ngrok
print("Starting ngrok tunnel on port 5000...")
public_url = ngrok.connect(5000)
print(f"\n✓ Public URL: {public_url}")
print(f"Dashboard: {public_url}/")
print(f"Predict API: {public_url}/predict")
print("\nPress Ctrl+C to stop tunnel")

# Keep it running
try:
    ngrok_process = ngrok.get_ngrok_process()
    ngrok_process.proc.wait()
except KeyboardInterrupt:
    print("\nShutting down...")
    ngrok.kill()

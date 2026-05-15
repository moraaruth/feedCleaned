Power Apps / Power Automate — Predict endpoint

1) Parse JSON schema for Power Automate `Parse JSON` action

Use this schema in the `Parse JSON` action after making the HTTP POST to `/predict` so the flow can reference `top_issue`, `trend`, `severity`, `recommendation`, and `predicted_volume`.

Schema (copy into the Parse JSON action):

{
  "type": "object",
  "properties": {
    "top_issue": { "type": "string" },
    "trend": { "type": "string" },
    "severity": { "type": "string" },
    "recommendation": { "type": "string" },
    "predicted_volume": { "type": "integer" },
    "change_pct": { "type": "number" }
  },
  "required": [ "top_issue", "trend", "severity", "recommendation", "predicted_volume" ]
}

2) Power Automate HTTP action example

- Method: POST
- URL: https://<your-service>/predict
- Headers: Content-Type: application/json
- Body examples:
  - Explicit cluster: { "cluster": "SIM & SWAP" }
  - Free text: { "text": "Many customers reporting SIM swap and OTP failures" }

3) After the HTTP action, add `Parse JSON` with the schema above. Then map the parsed outputs to your `Respond to Power App` action.

4) Quick curl test (run from machine that can reach the service):

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"SIM swap and OTP failures"}'
```

5) Integration test (python)

Run:

```bash
python -m pip install requests
python tests/test_predict.py
```

The test sends a sample request to `http://localhost:5000/predict` and validates the response keys.


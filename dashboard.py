# ============================================
# PA MOJA AI — LOCAL BROWSER DASHBOARD
# Run: python dashboard.py
# Opens: http://localhost:5000
# ============================================
import os
import pandas as pd
from flask import Flask, render_template_string, jsonify, request

ONEDRIVE = r"C:\Users\RMNYANGAU\OneDrive - SAFARICOM PLC"

FILES = {
    "feedback":   os.path.join(ONEDRIVE, "AutoFeedbackAIInsightsOut.xlsx"),
    "decisions":  os.path.join(ONEDRIVE, "AutoFeedbackDecisionLogs.xlsx"),
    "predictions":os.path.join(ONEDRIVE, "AutoFeedbackAIPredictions.xlsx"),
    "alerts":     os.path.join(ONEDRIVE, "AutoFeedbackAIAlerts.xlsx"),
}


class DataLoader:
  """Lightweight cached loader for Excel files.

  Caches DataFrames keyed by FILES entry and reloads when the file's
  modification time changes. This avoids repeated expensive Excel reads
  while ensuring predictions use the latest data.
  """
  def __init__(self, files_map, reload_interval=5):
    self.files = files_map
    self._cache = {}
    self._lock = threading.Lock()
    self.reload_interval = reload_interval

  def _get_mtime(self, path):
    try:
      return os.path.getmtime(path)
    except Exception:
      return None

  def get(self, key):
    path = self.files.get(key)
    if not path:
      return pd.DataFrame()

    now = time.time()
    with self._lock:
      entry = self._cache.get(key)
      mtime = self._get_mtime(path)
      # reload if not cached or file changed
      if (entry is None) or (entry.get("mtime") != mtime):
        try:
          # read only; letting pandas infer dtypes is fine for small Excel files
          df = pd.read_excel(path)
        except Exception:
          df = pd.DataFrame()
        self._cache[key] = {"df": df, "mtime": mtime, "ts": now}
      return self._cache[key]["df"]


import threading, time, re

# create a global loader instance
_loader = DataLoader(FILES)

def load(key):
  return _loader.get(key)

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PA MOJA AI — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: 'Segoe UI', sans-serif; background:#f0f2f5; color:#222; }

  header {
    background:#00A651; color:white;
    padding:16px 32px; display:flex;
    justify-content:space-between; align-items:center;
  }
  header h1 { font-size:20px; font-weight:700; letter-spacing:1px; }
  header span { font-size:13px; opacity:0.85; }

  nav {
    background:white; border-bottom:1px solid #e0e0e0;
    display:flex; gap:0; padding:0 32px;
  }
  nav button {
    background:none; border:none; padding:14px 24px;
    font-size:14px; cursor:pointer; color:#555;
    border-bottom:3px solid transparent;
  }
  nav button.active { color:#00A651; border-bottom-color:#00A651; font-weight:600; }
  nav button:hover  { color:#00A651; }

  .page { display:none; padding:28px 32px; }
  .page.active { display:block; }

  /* KPI CARDS */
  .kpi-row { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:24px; }
  .kpi {
    background:white; border-radius:10px; padding:20px 24px;
    box-shadow:0 1px 4px rgba(0,0,0,0.08);
  }
  .kpi .num { font-size:36px; font-weight:700; line-height:1; }
  .kpi .lbl { font-size:12px; color:#888; margin-top:6px; text-transform:uppercase; letter-spacing:.5px; }
  .kpi.red .num    { color:#d32f2f; }
  .kpi.orange .num { color:#e65100; }
  .kpi.blue .num   { color:#0078d4; }
  .kpi.green .num  { color:#00A651; }

  /* CARDS */
  .card {
    background:white; border-radius:10px; padding:24px;
    box-shadow:0 1px 4px rgba(0,0,0,0.08); margin-bottom:20px;
  }
  .card h2 { font-size:15px; font-weight:600; margin-bottom:18px; color:#333; }

  .two-col { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
  .three-col { display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; }

  /* BAR ROWS */
  .bar-row { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
  .bar-row .label { width:220px; font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .bar-row .bar-wrap { flex:1; background:#f0f0f0; border-radius:4px; height:14px; }
  .bar-row .bar-fill { height:14px; border-radius:4px; transition:width .4s; }
  .bar-row .count { width:30px; text-align:right; font-size:13px; font-weight:600; }
  .bar-row .badge { font-size:11px; padding:2px 8px; border-radius:10px; color:white; min-width:64px; text-align:center; }
  .neg  { background:#d32f2f; }
  .neu  { background:#9e9e9e; }
  .pos  { background:#00A651; }

  /* DECISION TABLE */
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th { text-align:left; padding:10px 12px; background:#f5f5f5; color:#555; font-weight:600; border-bottom:2px solid #e0e0e0; }
  td { padding:10px 12px; border-bottom:1px solid #f0f0f0; vertical-align:top; }
  tr:hover td { background:#fafafa; }

  .sev { display:inline-block; padding:3px 10px; border-radius:10px; font-size:11px; font-weight:700; color:white; }
  .sev.CRITICAL { background:#d32f2f; }
  .sev.HIGH     { background:#e65100; }
  .sev.MEDIUM   { background:#f9a825; color:#333; }
  .sev.LOW      { background:#00A651; }

  .status { display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; }
  .status.PENDING     { background:#fff3e0; color:#e65100; }
  .status.IN_PROGRESS { background:#e3f2fd; color:#0078d4; }
  .status.ACTIONED    { background:#e8f5e9; color:#00A651; }
  .status.CLOSED      { background:#f5f5f5; color:#888; }

  .engine-tag { font-size:11px; color:#0078d4; font-weight:600; }
  .trigger    { color:#555; font-size:12px; max-width:300px; }
  .overdue    { color:#d32f2f; font-size:11px; font-weight:700; }

  canvas { max-height:260px; }

  .refresh-btn {
    float:right; background:#00A651; color:white;
    border:none; padding:8px 18px; border-radius:6px;
    font-size:13px; cursor:pointer; margin-bottom:16px;
  }
  .refresh-btn:hover { background:#007a3d; }

  .no-data { color:#aaa; font-size:13px; padding:20px 0; text-align:center; }
</style>
</head>
<body>

<header>
  <h1>PA MOJA AI &mdash; Intelligence Dashboard</h1>
  <span id="lastRun">Loading...</span>
</header>

<nav>
  <button class="active" onclick="showPage('home',this)">Home</button>
  <button onclick="showPage('issues',this)">Issues</button>
  <button onclick="showPage('decisions',this)">Decisions</button>
  <button onclick="showPage('predictions',this)">Predictions</button>
</nav>

<!-- ── PAGE 1: HOME ── -->
<div class="page active" id="home">
  <div class="kpi-row" id="kpiRow"></div>
  <div class="two-col">
    <div class="card">
      <h2>Sentiment Breakdown</h2>
      <canvas id="sentimentChart"></canvas>
    </div>
    <div class="card">
      <h2>Decisions by Engine</h2>
      <canvas id="engineChart"></canvas>
    </div>
  </div>
</div>

<!-- ── PAGE 2: ISSUES ── -->
<div class="page" id="issues">
  <div class="card">
    <h2>Issues by Volume &amp; Sentiment</h2>
    <div id="issuesBars"></div>
  </div>
</div>

<!-- ── PAGE 3: DECISIONS ── -->
<div class="page" id="decisions">
  <button class="refresh-btn" onclick="loadData()">Refresh</button>
  <div class="card">
    <h2>Decision Queue</h2>
    <table>
      <thead>
        <tr>
          <th>Severity</th><th>Engine</th><th>Trigger</th>
          <th>Decision</th><th>Due</th><th>Status</th>
        </tr>
      </thead>
      <tbody id="decisionTable"></tbody>
    </table>
  </div>
</div>

<!-- ── PAGE 4: PREDICTIONS ── -->
<div class="page" id="predictions">
  <div class="card">
    <h2>What to expect tomorrow — predicted complaint volume per issue group</h2>
    <div id="predBars"></div>
  </div>
  <div class="card">
    <h2>Trend Direction</h2>
    <div id="trendList"></div>
  </div>
</div>

<script>
let sentChart, engChart, predChart;

function showPage(id, btn) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}

async function loadData() {
  const res  = await fetch('/api/data');
  const data = await res.json();

  document.getElementById('lastRun').textContent = 'Last run: ' + (data.last_run || 'Unknown');

  // ── KPI CARDS ──
  const kpi = data.kpi;
  document.getElementById('kpiRow').innerHTML = `
    <div class="kpi blue"><div class="num">${kpi.total}</div><div class="lbl">Total Feedback</div></div>
    <div class="kpi red"><div class="num">${kpi.negative}</div><div class="lbl">Negative Sentiment</div></div>
    <div class="kpi orange"><div class="num">${kpi.critical}</div><div class="lbl">Critical Decisions</div></div>
    <div class="kpi ${kpi.overdue > 0 ? 'red' : 'green'}"><div class="num">${kpi.overdue}</div><div class="lbl">Overdue Actions</div></div>
  `;

  // ── SENTIMENT DOUGHNUT ──
  const sc = document.getElementById('sentimentChart');
  if (sentChart) sentChart.destroy();
  sentChart = new Chart(sc, {
    type: 'doughnut',
    data: {
      labels: Object.keys(data.sentiment),
      datasets: [{ data: Object.values(data.sentiment),
        backgroundColor: ['#d32f2f','#9e9e9e','#00A651'], borderWidth:2 }]
    },
    options: { plugins:{ legend:{ position:'bottom' } }, cutout:'65%' }
  });

  // ── ENGINE BAR ──
  const ec = document.getElementById('engineChart');
  if (engChart) engChart.destroy();
  engChart = new Chart(ec, {
    type: 'bar',
    data: {
      labels: Object.keys(data.engines),
      datasets: [{ label:'Decisions', data: Object.values(data.engines),
        backgroundColor: ['#0078d4','#e65100','#d32f2f','#f9a825'], borderRadius:6 }]
    },
    options: { plugins:{ legend:{ display:false } }, scales:{ y:{ beginAtZero:true, ticks:{ stepSize:1 } } } }
  });

  // ── ISSUES BARS ──
  const maxCount = Math.max(...data.issues.map(i => i.count), 1);
  document.getElementById('issuesBars').innerHTML = data.issues.map(i => `
    <div class="bar-row">
      <div class="label" title="${i.pain_point}">${i.pain_point}</div>
      <div class="bar-wrap">
        <div class="bar-fill ${i.sentiment === 'Negative' ? 'neg' : i.sentiment === 'Positive' ? 'pos' : 'neu'}"
             style="width:${(i.count/maxCount*100).toFixed(1)}%"></div>
      </div>
      <div class="count">${i.count}</div>
      <span class="badge ${i.sentiment === 'Negative' ? 'neg' : i.sentiment === 'Positive' ? 'pos' : 'neu'}">${i.sentiment}</span>
    </div>
  `).join('') || '<div class="no-data">No issues data</div>';

  // ── DECISION TABLE ──
  document.getElementById('decisionTable').innerHTML = data.decisions.map(d => `
    <tr>
      <td><span class="sev ${d.severity}">${d.severity}</span></td>
      <td><span class="engine-tag">${d.engine}</span></td>
      <td><div class="trigger">${d.trigger}</div>${d.overdue ? '<div class="overdue">OVERDUE</div>' : ''}</td>
      <td style="font-size:12px;max-width:220px;">${d.decision}</td>
      <td style="font-size:12px;white-space:nowrap;">${d.due}</td>
      <td><span class="status ${d.status}">${d.status}</span></td>
    </tr>
  `).join('') || '<tr><td colspan="6" class="no-data">No decisions</td></tr>';

  // ── PREDICTIONS ──
  const maxPred = Math.max(...data.predictions.map(p => p.value), 1);
  document.getElementById('predBars').innerHTML = data.predictions.length ? data.predictions.map(p => `
    <div class="bar-row">
      <div class="label" title="${p.cluster}">${p.cluster}</div>
      <div class="bar-wrap">
        <div class="bar-fill ${p.trend === 'UP' ? 'neg' : p.trend === 'DOWN' ? 'pos' : 'neu'}"
             style="width:${(p.value/maxPred*100).toFixed(1)}%"></div>
      </div>
      <div class="count">${p.value}</div>
      <span class="badge ${p.trend === 'UP' ? 'neg' : p.trend === 'DOWN' ? 'pos' : 'neu'}">${p.trend === 'UP' ? 'Rising' : p.trend === 'DOWN' ? 'Falling' : 'Stable'}</span>
    </div>
  `).join('') : '<div class="no-data">No predictions available</div>';

  document.getElementById('trendList').innerHTML = data.predictions.length ? data.predictions.map(p => `
    <div style="display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid #f0f0f0">
      <span style="font-size:22px">${p.trend === 'UP' ? '&#x2197;' : p.trend === 'DOWN' ? '&#x2198;' : '&#x2192;'}</span>
      <div>
        <div style="font-size:13px;font-weight:600">${p.cluster}</div>
        <div style="font-size:12px;color:#888">
          ${p.trend === 'UP' ? 'Complaints likely to increase tomorrow — prepare team' :
            p.trend === 'DOWN' ? 'Complaints likely to decrease — actions working' :
            'Volume stable — monitor'}
        </div>
      </div>
      <span style="margin-left:auto;font-size:20px;font-weight:700;color:${p.trend==='UP'?'#d32f2f':p.trend==='DOWN'?'#00A651':'#888'}">${p.value} cases</span>
    </div>
  `).join('') : '';
}

loadData();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

def safe(val, fallback=""):
    """Convert any value to JSON-safe type."""
    if pd.isna(val) if not isinstance(val, (list, dict)) else False:
        return fallback
    if isinstance(val, float) and val != val:  # NaN check
        return fallback
    return val

@app.route("/api/data")
def api_data():
    fb   = load("feedback")
    dec  = load("decisions")
    pred = load("predictions")

    # KPIs
    total    = len(fb)
    negative = int((fb["Sentiment"] == "Negative").sum()) if "Sentiment" in fb.columns else 0
    critical = int((dec["Severity"] == "CRITICAL").sum()) if "Severity" in dec.columns else 0

    overdue = 0
    if "DueDate" in dec.columns and "Status" in dec.columns:
        dec["DueDate"] = pd.to_datetime(dec["DueDate"], errors="coerce")
        overdue = int(((dec["Status"] == "PENDING") & (dec["DueDate"] < pd.Timestamp.now())).sum())

    # Sentiment breakdown
    sentiment = {k: int(v) for k, v in fb["Sentiment"].value_counts().to_dict().items()} if "Sentiment" in fb.columns else {}

    # Decisions by engine
    engines = {k: int(v) for k, v in dec["Engine"].value_counts().to_dict().items()} if "Engine" in dec.columns else {}

    # Issues bars — group by PainPoint + dominant sentiment
    issues = []
    if "PainPoint" in fb.columns and "Sentiment" in fb.columns:
        grp = fb.groupby("PainPoint")
        for pain, group in grp:
            dominant = group["Sentiment"].value_counts().idxmax()
            issues.append({"pain_point": pain, "count": len(group), "sentiment": dominant})
        issues.sort(key=lambda x: x["count"], reverse=True)

    # Decision rows
    decisions = []
    if not dec.empty:
        for _, row in dec.iterrows():
            due_str = ""
            overdue_flag = False
            if "DueDate" in dec.columns and pd.notna(row.get("DueDate")):
                due_dt = pd.to_datetime(row["DueDate"], errors="coerce")
                due_str = due_dt.strftime("%d %b %Y") if pd.notna(due_dt) else ""
                overdue_flag = due_dt < pd.Timestamp.now() and row.get("Status") == "PENDING"
            decisions.append({
                "severity": safe(row.get("Severity", "")),
                "engine":   safe(row.get("Engine", "")),
                "trigger":  safe(row.get("Trigger", ""))[:80],
                "decision": safe(row.get("Decision", ""))[:100],
                "due":      due_str,
                "status":   safe(row.get("Status", "")),
                "overdue":  overdue_flag,
            })

    # Predictions — latest run only, clean clusters only
    predictions = []
    if not pred.empty and "Cluster" in pred.columns:
        pred["Date"] = pd.to_datetime(pred["Date"], errors="coerce")
        # keep only rows with a valid date and Predicted_Tomorrow
        pred = pred.dropna(subset=["Date", "Predicted_Tomorrow", "Trend"])
        if not pred.empty:
            latest_date = pred["Date"].max()
            pred = pred[pred["Date"] == latest_date]
            for _, row in pred.iterrows():
                val = row.get("Predicted_Tomorrow", 0)
                predictions.append({
                    "cluster": str(row.get("Cluster", "")),
                    "value":   0 if pd.isna(val) else int(val),
                    "trend":   str(row.get("Trend", "FLAT")),
                })
            predictions.sort(key=lambda x: x["value"], reverse=True)

    last_run = ""
    if "RunID" in dec.columns and not dec.empty:
        last_run = str(dec["RunID"].iloc[-1])

    return jsonify({
        "kpi":         {"total": total, "negative": negative, "critical": critical, "overdue": overdue},
        "sentiment":   sentiment,
        "engines":     engines,
        "issues":      issues,
        "decisions":   decisions,
        "predictions": predictions,
        "last_run":    last_run,
    })

@app.route("/predict", methods=["POST"])
def predict():
  payload = request.get_json(silent=True) or {}

  # Accept either 'cluster' (explicit) or 'text' (free text from Power App)
  requested_cluster = None
  text = None
  if isinstance(payload, dict):
    requested_cluster = payload.get("cluster") or payload.get("issue")
    text = payload.get("text") or payload.get("context")

  pred = load("predictions")
  if pred.empty or "Cluster" not in pred.columns:
    return jsonify({"error": "No prediction data available"}), 400

  # normalize date and drop rows missing the predicted value
  pred = pred.copy()
  pred["Date"] = pd.to_datetime(pred["Date"], errors="coerce")
  pred = pred.dropna(subset=["Date", "Predicted_Tomorrow"]) if "Predicted_Tomorrow" in pred.columns else pred
  if pred.empty:
    return jsonify({"error": "No valid prediction rows"}), 400

  latest_date = pred["Date"].max()
  prev_dates = sorted(pred["Date"].unique())
  prev_date = prev_dates[-2] if len(prev_dates) >= 2 else None

  latest = pred[pred["Date"] == latest_date]

  # choose cluster
  chosen = None
  clusters = latest["Cluster"].astype(str).tolist()
  if requested_cluster:
    # exact match preferred
    req = str(requested_cluster).strip()
    if req in clusters:
      chosen = req
    else:
      # case-insensitive match
      low = {c.lower(): c for c in clusters}
      chosen = low.get(req.lower())

  if not chosen and text:
    # simple keyword match against cluster names
    txt = re.sub(r"[^a-z0-9 ]", " ", str(text).lower())
    tokens = set(txt.split())
    best = None
    best_score = 0
    for c in clusters:
      cname = str(c).lower()
      score = sum(1 for t in tokens if t in cname)
      if score > best_score:
        best_score = score
        best = c
    if best_score > 0:
      chosen = best

  # fallback to top predicted cluster
  if not chosen:
    top_idx = latest["Predicted_Tomorrow"].astype(float).idxmax()
    chosen = str(latest.loc[top_idx, "Cluster"])

  # collect latest & previous values for chosen cluster
  row_latest = latest[latest["Cluster"].astype(str) == str(chosen)]
  if row_latest.empty:
    return jsonify({"error": "Requested cluster not found in latest predictions"}), 400
  row_latest = row_latest.iloc[0]
  predicted_volume = float(row_latest.get("Predicted_Tomorrow", 0) or 0)
  declared_trend = str(row_latest.get("Trend", "FLAT")).upper() if "Trend" in row_latest.index else "FLAT"

  prev_volume = None
  if prev_date is not None:
    prev = pred[pred["Date"] == prev_date]
    prev_row = prev[prev["Cluster"].astype(str) == str(chosen)]
    if not prev_row.empty:
      prev_volume = float(prev_row.iloc[0].get("Predicted_Tomorrow", 0) or 0)

  # compute change and trend
  change_pct = None
  if (prev_volume is not None) and (prev_volume > 0):
    change_pct = (predicted_volume - prev_volume) / prev_volume
  # primary source of trend is declared Trend column (if present), otherwise derive from change_pct
  if declared_trend and declared_trend in {"UP", "DOWN", "FLAT"}:
    trend = declared_trend
  else:
    if change_pct is None:
      trend = "FLAT"
    elif change_pct > 0.1:
      trend = "UP"
    elif change_pct < -0.1:
      trend = "DOWN"
    else:
      trend = "FLAT"

  # compute severity (rule-based): volume thresholds and change weight
  severity = "Low"
  vol = predicted_volume
  if vol > 200 or (change_pct is not None and change_pct > 1.0):
    severity = "Critical"
  elif vol > 100 or (change_pct is not None and change_pct > 0.5):
    severity = "High"
  elif vol > 30 or (change_pct is not None and abs(change_pct) > 0.25):
    severity = "Medium"

  # recommendation rules
  if trend == "UP" and severity in {"High", "Critical"}:
    recommendation = "Scale support teams immediately; open incident channels"
  elif trend == "UP":
    recommendation = "Prepare additional shifts; monitor closely"
  elif trend == "DOWN":
    recommendation = "Monitor; consider reassigning resources to other queues"
  else:
    recommendation = "Stable — continue monitoring and validate predictions"

  response = {
    "top_issue": str(chosen),
    "trend": trend,
    "severity": severity,
    "recommendation": recommendation,
    "predicted_volume": int(predicted_volume),
  }
  if change_pct is not None:
    response["change_pct"] = round(float(change_pct), 3)

  return jsonify(response)

if __name__ == "__main__":
    import webbrowser, threading
    threading.Timer(1, lambda: webbrowser.open("http://localhost:5000")).start()
    print("Dashboard running at http://localhost:5000  (Ctrl+C to stop)")
    app.run(debug=False, port=5000)

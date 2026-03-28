# ============================================
# PA MOJA AI — INTELLIGENCE PIPELINE v3
# Feedback > Sentiment > Cluster > Predict > Alert > Decision > Action > Outcome > Learn
# ============================================
import os, sys, uuid, time
import pandas as pd
import numpy as np
from datetime import datetime
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "AutoFeedback.xlsx")
RUN_ID     = datetime.now().strftime("%Y%m%d_%H%M%S")

OUT = {
    "Feedback":     os.path.join(BASE_DIR, "AI_Feedback_Processed.xlsx"),
    "Predictions":  os.path.join(BASE_DIR, "AI_Predictions.xlsx"),
    "Alerts":       os.path.join(BASE_DIR, "AI_Alerts.xlsx"),
    "Insights":     os.path.join(BASE_DIR, "AI_Insights.xlsx"),
    "DecisionLogs": os.path.join(BASE_DIR, "AI_DecisionLogs.xlsx"),
    "Actions":      os.path.join(BASE_DIR, "AI_Actions.xlsx"),
    "Outcomes":     os.path.join(BASE_DIR, "AI_Outcomes.xlsx"),
    "Learning":     os.path.join(BASE_DIR, "AI_Learning.xlsx"),
}

DECISION_LOG_COLUMNS = [
    "DecisionID", "RunID", "Date", "Engine", "Cluster", "Trigger",
    "Severity", "Decision", "Owner", "Status", "ActionTaken",
    "DueDate", "ClosedDate", "OutcomeRating", "Notes",
]

SEVERITY_RULES = {
    "CRITICAL": ("ESCALATE to leadership immediately. Convene emergency review.", 1),
    "HIGH":     ("Assign dedicated team. Resolve within 24 hours.", 1),
    "MEDIUM":   ("Schedule review in next sprint. Monitor daily.", 7),
    "LOW":      ("Log for quarterly review.", 30),
}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def save(path, new_df, key_cols=None):
    if new_df is None or new_df.empty:
        return
    if os.path.exists(path):
        try:
            existing = pd.read_excel(path)
            for col in ["Date", "DueDate", "ClosedDate", "ActionDate", "OutcomeDate"]:
                for frame in [existing, new_df]:
                    if col in frame.columns:
                        frame[col] = pd.to_datetime(frame[col], errors="coerce")
            combined = pd.concat([existing, new_df], ignore_index=True)
            final = combined.drop_duplicates(subset=key_cols) if key_cols else combined.drop_duplicates()
        except Exception:
            final = new_df
    else:
        final = new_df

    if path == OUT["DecisionLogs"]:
        for col in DECISION_LOG_COLUMNS:
            if col not in final.columns:
                final[col] = ""

    for attempt in range(3):
        try:
            with pd.ExcelWriter(path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as w:
                final.to_excel(w, index=False, sheet_name="Data")
                ws = w.sheets["Data"]
                for i, col in enumerate(final.columns):
                    width = max(final[col].astype(str).str.len().max(), len(str(col))) + 2
                    ws.set_column(i, i, min(width, 60))
            break
        except PermissionError:
            if attempt < 2:
                print(f"  File locked: {os.path.basename(path)} — retrying in 2s...")
                time.sleep(2)
            else:
                print(f"  SKIPPED (file open): {os.path.basename(path)}")


def decision_row(engine, cluster, trigger, severity, decision, due_days):
    return {
        "DecisionID":    str(uuid.uuid4()),
        "RunID":         RUN_ID,
        "Date":          pd.Timestamp.now().normalize(),
        "Engine":        engine,
        "Cluster":       cluster,
        "Trigger":       trigger,
        "Severity":      severity,
        "Decision":      decision,
        "Owner":         "UNASSIGNED",
        "Status":        "PENDING",
        "ActionTaken":   "",
        "DueDate":       (pd.Timestamp.now() + pd.Timedelta(days=due_days)).normalize(),
        "ClosedDate":    "",
        "OutcomeRating": "",
        "Notes":         "",
    }


# ── STAGE 1: LOAD ─────────────────────────────────────────────────────────────
if not os.path.exists(INPUT_FILE):
    sys.exit(f"Input file not found: {INPUT_FILE}")

df = pd.read_excel(INPUT_FILE)
df.columns = df.columns.str.strip()
print(f"[1/8] LOAD          — {len(df)} records  (RunID: {RUN_ID})")

# ── STAGE 2: PROCESS ──────────────────────────────────────────────────────────
# Identify text columns — prefer SampleComments, fall back to any text-like column
comment_col = next(
    (c for c in ["SampleComments", "Description", "Feedback", "Comment", "Title"] if c in df.columns),
    None
)
pain_col = next((c for c in ["PainPoint", "Category", "IssueType"] if c in df.columns), None)
date_col  = next((c for c in df.columns if "date" in c.lower() or "created" in c.lower()), None)

if comment_col is None:
    sys.exit("No recognisable text column found in input file.")

# Build combined_text from actual comment + pain point (not duplicated)
parts = [df[comment_col].fillna("")]
if pain_col and pain_col != comment_col:
    parts.append(df[pain_col].fillna(""))
df["combined_text"] = pd.concat(parts, axis=1).astype(str).agg(" ".join, axis=1).str.strip()

df["Date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize() if date_col else pd.Timestamp.now().normalize()
print(f"[2/8] PROCESS       — text from '{comment_col}' + '{pain_col}'")

# ── STAGE 3: INSIGHT — SENTIMENT ──────────────────────────────────────────────
NEGATIVE_KW = [
    "issue", "problem", "fail", "error", "complaint", "wrong", "delay", "reversal",
    "stuck", "block", "suspend", "fraud", "dispute", "loss", "unable", "cannot",
    "not working", "declined", "rejected", "missing", "angry", "frustrated",
    "panicked", "did not", "doesn't", "doesn't work", "no service",
]
POSITIVE_KW = [
    "success", "resolved", "working", "good", "great", "excellent", "thank",
    "appreciate", "happy", "satisfied", "complete", "relieved", "quick", "fast",
]

def classify_sentiment(text):
    t = str(text).lower()
    neg = sum(1 for k in NEGATIVE_KW if k in t)
    pos = sum(1 for k in POSITIVE_KW if k in t)
    if neg > pos:
        return "Negative"
    if pos > neg:
        return "Positive"
    # Fall back to TextBlob polarity
    polarity = TextBlob(t).sentiment.polarity
    if polarity < -0.05:
        return "Negative"
    if polarity > 0.05:
        return "Positive"
    return "Neutral"

df["Polarity"]  = df["combined_text"].apply(lambda x: round(TextBlob(str(x)).sentiment.polarity, 4))
df["Sentiment"] = df["combined_text"].apply(classify_sentiment)

# ── STAGE 3b: CLUSTERING ──────────────────────────────────────────────────────
# Use PainPoint directly as cluster label when available (more meaningful than KMeans on 6 rows)
if pain_col:
    df["ClusterLabel"] = df[pain_col].str.strip()
    df["Cluster"]      = pd.Categorical(df["ClusterLabel"]).codes
else:
    n_clusters = min(4, max(2, len(df) // 2))
    vec = TfidfVectorizer(stop_words="english", max_features=500)
    X   = vec.fit_transform(df["combined_text"])
    km  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)
    df["Cluster"] = km.labels_
    vocab = vec.get_feature_names_out()
    labels = {}
    for i in range(n_clusters):
        top = km.cluster_centers_[i].argsort()[-3:][::-1]
        words = [vocab[j] for j in top if km.cluster_centers_[i][j] > 0]
        labels[i] = " & ".join(words).upper() if words else f"GROUP {i+1}"
    df["ClusterLabel"] = df["Cluster"].map(labels)

sentiment_summary = df["Sentiment"].value_counts().to_dict()
print(f"[3/8] INSIGHT       — Sentiment: {sentiment_summary} | Clusters: {df['ClusterLabel'].nunique()}")

# ── STAGE 4: PREDICTION ───────────────────────────────────────────────────────
daily = df.groupby(["Date", "ClusterLabel"]).size().reset_index(name="Count")

preds = []
for cluster in daily["ClusterLabel"].unique():
    c = daily[daily["ClusterLabel"] == cluster].sort_values("Date").copy()
    c["DayIdx"] = np.arange(len(c))

    if len(c) >= 2:
        reg = LinearRegression().fit(c[["DayIdx"]], c["Count"])
        next_val = max(0, int(round(reg.predict([[len(c)]])[0])))
        coef = reg.coef_[0]
    else:
        # Single data point — use count as flat prediction
        next_val = int(c["Count"].iloc[-1])
        coef = 0.0

    trend = "UP" if coef > 0.1 else ("DOWN" if coef < -0.1 else "FLAT")
    preds.append({
        "Date":               pd.Timestamp.now().normalize(),
        "Cluster":            cluster,
        "TodayCount":         int(c["Count"].iloc[-1]),
        "Predicted_Tomorrow": next_val,
        "Trend":              trend,
        "Confidence":         "LOW" if len(c) < 3 else ("MEDIUM" if len(c) < 7 else "HIGH"),
    })

pred_df = pd.DataFrame(preds)
print(f"[4/8] PREDICTION    — {len(pred_df)} cluster forecasts")

# ── STAGE 5: ALERT ────────────────────────────────────────────────────────────
alerts = []

# Volume spike per cluster
for cluster in daily["ClusterLabel"].unique():
    c = daily[daily["ClusterLabel"] == cluster].sort_values("Date")
    if len(c) >= 2:
        latest, mean = c["Count"].iloc[-1], c["Count"].mean()
        if latest / mean > 1.5:
            alerts.append({
                "Date": c["Date"].iloc[-1], "Cluster": cluster,
                "Alert": f"VOLUME SPIKE: {cluster} — {latest} cases vs avg {mean:.1f}",
                "Severity": "CRITICAL" if latest / mean > 3 else "HIGH",
            })

# High negative sentiment
neg_pct = (df["Sentiment"] == "Negative").mean()
if neg_pct >= 0.4:
    alerts.append({
        "Date": pd.Timestamp.now().normalize(), "Cluster": "ALL",
        "Alert": f"HIGH NEGATIVE SENTIMENT: {neg_pct:.0%} of feedback is negative",
        "Severity": "CRITICAL" if neg_pct >= 0.6 else "HIGH",
    })

# Rising trend with meaningful volume
for _, row in pred_df.iterrows():
    if row["Trend"] == "UP" and row["Predicted_Tomorrow"] >= 2:
        alerts.append({
            "Date": pd.Timestamp.now().normalize(), "Cluster": row["Cluster"],
            "Alert": f"RISING TREND: {row['Cluster']} — {row['Predicted_Tomorrow']} predicted tomorrow",
            "Severity": "MEDIUM",
        })

# Fraud / SIM swap signals
FRAUD_KW = ["fraud", "scam", "stolen", "unauthorized", "sim swap", "hacked", "phishing", "wrong number"]
fraud_hits = df["combined_text"].str.lower().apply(lambda t: any(k in t for k in FRAUD_KW)).sum()
if fraud_hits > 0:
    alerts.append({
        "Date": pd.Timestamp.now().normalize(), "Cluster": "FRAUD_SIGNAL",
        "Alert": f"FRAUD SIGNAL: {fraud_hits} record(s) contain fraud-related keywords",
        "Severity": "CRITICAL" if fraud_hits >= 3 else "HIGH",
    })

alert_df = pd.DataFrame(alerts) if alerts else pd.DataFrame(columns=["Date", "Cluster", "Alert", "Severity"])
print(f"[5/8] ALERT         — {len(alert_df)} alerts raised")

# ── STAGE 6: DECISION ─────────────────────────────────────────────────────────
rows = []

# From alerts
for _, a in alert_df.iterrows():
    sev = a.get("Severity", "LOW")
    action, due = SEVERITY_RULES[sev]
    rows.append(decision_row("FEEDBACK", a["Cluster"], a["Alert"], sev, action, due))

# Proactive from predictions
for _, p in pred_df.iterrows():
    if p["Trend"] == "UP" and p["Predicted_Tomorrow"] >= 2:
        action, due = SEVERITY_RULES["MEDIUM"]
        rows.append(decision_row(
            "FEEDBACK", p["Cluster"],
            f"PREDICTED RISE: {p['Predicted_Tomorrow']} cases tomorrow in '{p['Cluster']}'",
            "MEDIUM", action, due
        ))

# Sub-engines
from engines.churn            import run as run_churn
from engines.fraud            import run as run_fraud
from engines.service_recovery import run as run_service_recovery

rows += run_churn(df, decision_row)
rows += run_fraud(df, daily, decision_row)
rows += run_service_recovery(df, daily, pred_df, decision_row)

decision_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=DECISION_LOG_COLUMNS)
print(f"[6/8] DECISION      — {len(decision_df)} decisions across all engines")

# ── STAGE 7: ACTION ───────────────────────────────────────────────────────────
if not decision_df.empty:
    action_df = decision_df[["DecisionID", "Date", "Engine", "Cluster", "Decision", "Owner", "Status", "DueDate"]].copy()
    action_df.rename(columns={"Date": "ActionDate"}, inplace=True)
else:
    action_df = pd.DataFrame()
print(f"[7/8] ACTION        — {len(action_df)} action items queued")

# ── STAGE 8: OUTCOME + LEARNING ───────────────────────────────────────────────
outcomes, learning = [], []
if os.path.exists(OUT["Feedback"]):
    try:
        prev = pd.read_excel(OUT["Feedback"])
        if "Sentiment" in prev.columns:
            prev_neg = (prev["Sentiment"] == "Negative").mean()
            curr_neg = (df["Sentiment"] == "Negative").mean()
            delta    = curr_neg - prev_neg
            label    = "IMPROVED" if delta < -0.05 else ("WORSENED" if delta > 0.05 else "STABLE")
            outcomes.append({
                "OutcomeDate": pd.Timestamp.now().normalize(), "RunID": RUN_ID,
                "Metric": "Negative Sentiment %",
                "Previous": f"{prev_neg:.0%}", "Current": f"{curr_neg:.0%}",
                "Delta": f"{delta:+.0%}", "Outcome": label,
            })
            learning.append({
                "Date": pd.Timestamp.now().normalize(), "RunID": RUN_ID,
                "Metric": "Negative Sentiment Trend", "Outcome": label, "Delta": round(delta, 4),
                "Lesson": (
                    "Actions are working — continue." if delta < -0.05 else
                    "Review actions — not effective yet." if delta > 0.05 else
                    "Stable — maintain current approach."
                ),
            })
    except Exception:
        pass

outcome_df  = pd.DataFrame(outcomes)
learning_df = pd.DataFrame(learning)
print(f"[8/8] OUTCOME       — {len(outcome_df)} measured | LEARNING — {len(learning_df)} lessons")

# ── EXECUTIVE SUMMARY ─────────────────────────────────────────────────────────
top = pred_df.sort_values("Predicted_Tomorrow", ascending=False).iloc[0]
top_dec = decision_df.iloc[0]["Decision"] if not decision_df.empty else "No action required."
summary_df = pd.DataFrame([{
    "Date":    pd.Timestamp.now().normalize(),
    "RunID":   RUN_ID,
    "Summary": (
        f"Top issue: '{top['Cluster']}' — {top['TodayCount']} cases today, "
        f"~{top['Predicted_Tomorrow']} predicted tomorrow (Trend: {top['Trend']}). "
        f"Sentiment: {sentiment_summary}. "
        f"Action: {top_dec}"
    ),
}])

# ── SAVE ALL ──────────────────────────────────────────────────────────────────
save(OUT["Feedback"],     df)
save(OUT["Predictions"],  pred_df)
save(OUT["Alerts"],       alert_df)
save(OUT["Insights"],     summary_df)
save(OUT["DecisionLogs"], decision_df, key_cols=["DecisionID"])
save(OUT["Actions"],      action_df,   key_cols=["DecisionID"])
save(OUT["Outcomes"],     outcome_df)
save(OUT["Learning"],     learning_df)

print(f"\n{'='*60}")
print(f"  RunID    : {RUN_ID}")
print(f"  Records  : {len(df)}")
print(f"  Sentiment: {sentiment_summary}")
print(f"  Alerts   : {len(alert_df)}")
print(f"  Decisions: {len(decision_df)}")
print(f"  Top Issue: {top['Cluster']} — {top['Predicted_Tomorrow']} predicted tomorrow ({top['Trend']})")
print(f"{'='*60}")

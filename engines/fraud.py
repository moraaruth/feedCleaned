# ── FRAUD ENGINE ──────────────────────────────────────────────────────────────
# Trigger: fraud keywords + volume spikes + reversal patterns
# Decision: freeze account, escalate to security, investigate

FRAUD_KEYWORDS = [
    "fraud", "scam", "stolen", "unauthorized", "wrong number", "reversal",
    "sim swap", "hacked", "phishing", "impersonation", "fake"
]

FRAUD_CLUSTERS = ["fraud", "reversal", "sim swap", "wrong number", "stolen"]

def run(df, daily, make_decision_row):
    decisions = []

    if "combined_text" not in df.columns:
        return decisions

    # Direct fraud keyword hits
    fraud_hits = df["combined_text"].str.lower().apply(
        lambda t: any(k in t for k in FRAUD_KEYWORDS)
    )
    fraud_df   = df[fraud_hits].copy()
    fraud_count = len(fraud_df)

    if fraud_count > 0:
        severity = "CRITICAL" if fraud_count >= 3 else "HIGH"
        decisions.append(make_decision_row(
            "FRAUD", "KEYWORD_MATCH",
            f"FRAUD SIGNAL: {fraud_count} records contain fraud-related keywords",
            severity,
            "FREEZE flagged accounts immediately. Escalate to Security & Fraud team. "
            "Initiate investigation within 1 hour. Notify compliance.",
            due_days=1
        ))

    # Spike in reversal-type clusters (sudden reversal surge = fraud pattern)
    if not daily.empty and "ClusterLabel" in daily.columns:
        for cluster in daily["ClusterLabel"].unique():
            if any(k in str(cluster).lower() for k in FRAUD_CLUSTERS):
                c_data = daily[daily["ClusterLabel"] == cluster].sort_values("Date")
                if len(c_data) >= 2:
                    latest = c_data["Count"].iloc[-1]
                    mean   = c_data["Count"].mean()
                    if latest / mean > 2.0:   # 2x spike = fraud pattern
                        decisions.append(make_decision_row(
                            "FRAUD", cluster,
                            f"FRAUD SPIKE: {cluster} volume {latest:.0f}x vs avg {mean:.1f}",
                            "CRITICAL",
                            "Immediate security review. Suspend new reversals pending investigation. "
                            "Alert fraud operations team.",
                            due_days=1
                        ))

    return decisions

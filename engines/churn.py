# ── CHURN ENGINE ──────────────────────────────────────────────────────────────
# Trigger: customer has repeated negative issues (likely to churn)
# Decision: retention offer, priority callback, account review

CHURN_KEYWORDS = [
    "pin", "reversal", "suspended", "blocked", "fraud", "wrong number",
    "fuliza", "kyc", "registration", "sim swap"
]

def run(df, make_decision_row):
    decisions = []

    # Flag customers with 2+ negative issues (repeat complainants = churn risk)
    if "combined_text" not in df.columns or "Sentiment" not in df.columns:
        return decisions

    neg_df = df[df["Sentiment"] == "Negative"].copy()

    # Count repeat issue types per cluster
    churn_clusters = (
        neg_df.groupby("ClusterLabel")
        .size()
        .reset_index(name="NegCount")
        .query("NegCount >= 2")
    )

    for _, row in churn_clusters.iterrows():
        cluster   = row["ClusterLabel"]
        neg_count = row["NegCount"]
        trigger   = f"CHURN RISK: {neg_count} negative cases in '{cluster}'"
        severity  = "CRITICAL" if neg_count >= 5 else "HIGH" if neg_count >= 3 else "MEDIUM"
        action    = {
            "CRITICAL": "Immediate retention call. Offer waiver or upgrade. Escalate to CX Director.",
            "HIGH":     "Priority callback within 4 hours. Offer loyalty reward or fee waiver.",
            "MEDIUM":   "Schedule proactive outreach. Send satisfaction survey.",
        }[severity]
        decisions.append(make_decision_row("CHURN", cluster, trigger, severity, action,
                                           due_days=1 if severity == "CRITICAL" else 2))

    # High churn risk keywords
    churn_text_hits = neg_df["combined_text"].str.lower().apply(
        lambda t: any(k in t for k in CHURN_KEYWORDS)
    ).sum()

    if churn_text_hits > 0:
        decisions.append(make_decision_row(
            "CHURN", "KEYWORD_MATCH",
            f"CHURN SIGNAL: {churn_text_hits} records match high-churn keywords",
            "MEDIUM",
            "Review keyword-matched cases. Assign retention team for follow-up.",
            due_days=3
        ))

    return decisions

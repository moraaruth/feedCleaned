# ── SERVICE RECOVERY ENGINE ───────────────────────────────────────────────────
# Trigger: volume spike + SLA breach risk + rising trend
# Decision: auto-assign team, escalate, communicate to customers

SLA_THRESHOLDS = {
    "CRITICAL": 1,   # resolve within 1 day
    "HIGH":     2,
    "MEDIUM":   5,
    "LOW":      14,
}

HIGH_IMPACT_KEYWORDS = [
    "network", "data", "mpesa", "pin", "sim", "account", "suspended",
    "blocked", "registration", "kyc", "fuliza"
]

def run(df, daily, pred_df, make_decision_row):
    decisions = []

    if daily.empty or "ClusterLabel" not in daily.columns:
        return decisions

    for cluster in daily["ClusterLabel"].unique():
        c_data = daily[daily["ClusterLabel"] == cluster].sort_values("Date")
        if len(c_data) < 2:
            continue

        latest  = c_data["Count"].iloc[-1]
        mean    = c_data["Count"].mean()
        ratio   = latest / mean if mean > 0 else 0

        # Get prediction for this cluster
        pred_row = pred_df[pred_df["Cluster"] == cluster] if not pred_df.empty else pd.DataFrame()
        trend    = pred_row["Trend"].iloc[0] if not pred_row.empty else "FLAT"
        tomorrow = pred_row["Predicted_Tomorrow"].iloc[0] if not pred_row.empty else 0

        # SLA breach risk: spike + rising trend
        if ratio > 1.5 and trend == "UP":
            severity = "CRITICAL" if ratio > 3 else "HIGH"
            decisions.append(make_decision_row(
                "SERVICE_RECOVERY", cluster,
                f"SLA BREACH RISK: {cluster} at {ratio:.1f}x avg, trending UP ({tomorrow} predicted tomorrow)",
                severity,
                f"Auto-assign to Service Recovery team. "
                f"Target resolution: {SLA_THRESHOLDS[severity]} day(s). "
                f"Send proactive customer communication. Update status page.",
                due_days=SLA_THRESHOLDS[severity]
            ))

        # High-impact service area with any spike
        elif ratio > 1.3 and any(k in str(cluster).lower() for k in HIGH_IMPACT_KEYWORDS):
            decisions.append(make_decision_row(
                "SERVICE_RECOVERY", cluster,
                f"HIGH-IMPACT SPIKE: {cluster} at {ratio:.1f}x average volume",
                "MEDIUM",
                "Assign team lead to monitor. Prepare customer communication template. "
                "Review within 24 hours.",
                due_days=SLA_THRESHOLDS["MEDIUM"]
            ))

    # Overall volume surge across all clusters
    total_today = daily[daily["Date"] == daily["Date"].max()]["Count"].sum()
    total_avg   = daily.groupby("Date")["Count"].sum().mean()
    if total_today / total_avg > 1.5 if total_avg > 0 else False:
        decisions.append(make_decision_row(
            "SERVICE_RECOVERY", "ALL_CLUSTERS",
            f"SYSTEM-WIDE SURGE: Total volume {total_today:.0f} vs avg {total_avg:.1f}",
            "HIGH",
            "Activate incident response protocol. Brief CX leadership. "
            "All hands on deck for resolution.",
            due_days=1
        ))

    return decisions

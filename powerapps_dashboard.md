# PA MOJA AI — POWER APPS DASHBOARD
# 4 screens, no overcrowding, easy to implement
# Data: PainPoint | MonthDate | SampleComments | Sentiment | ClusterLabel
# ─────────────────────────────────────────────────────────────────────────────

## SCREEN LAYOUT OVERVIEW
#
#  Screen 1 — HOME (4 KPI cards + nav buttons)
#  Screen 2 — ISSUES (bar chart + filterable list)
#  Screen 3 — DECISIONS (pending actions queue)
#  Screen 4 — CASE DETAIL (single decision + update form)
#
# ─────────────────────────────────────────────────────────────────────────────


## ════════════════════════════════════════════════════════
## SCREEN 1 — HOME DASHBOARD
## ════════════════════════════════════════════════════════
#
#  ┌─────────────────────────────────────────────────────┐
#  │  PA MOJA AI                          [Today's date] │
#  ├──────────┬──────────┬──────────┬────────────────────┤
#  │ TOTAL    │ NEGATIVE │ CRITICAL │ OVERDUE            │
#  │  31      │   17     │   2      │   3                │
#  │ Feedback │ Sentiment│ Alerts   │ Actions            │
#  ├──────────┴──────────┴──────────┴────────────────────┤
#  │  [View Issues]   [View Decisions]   [View Learning] │
#  └─────────────────────────────────────────────────────┘

## CONTROLS TO ADD:

# 1. Header Label
#    Text:    "PA MOJA AI"
#    Size:    28, Bold
#    Color:   RGBA(0,120,212,1)   ← Safaricom blue or use #00A651 green

# 2. Date Label (top right)
#    Text:    Text(Today(), "dd mmm yyyy")
#    Size:    14, Color: Gray

# 3. KPI Card 1 — Total Feedback
#    Insert > Rectangle (fill: RGBA(0,120,212,0.1), radius: 8)
#    Big number label:
#      Text:  Text(CountRows(AutoFeedbackAIInsightsOut), "0")
#      Size:  36, Bold
#    Small label below:
#      Text:  "Total Feedback"
#      Size:  12, Color: Gray

# 4. KPI Card 2 — Negative Sentiment
#    Big number:
#      Text:  Text(CountIf(AutoFeedbackAIInsightsOut, Sentiment = "Negative"), "0")
#      Color: RGBA(200,0,0,1)
#    Small label: "Negative Sentiment"

# 5. KPI Card 3 — Critical Decisions
#    Big number:
#      Text:  Text(CountIf(AutoFeedbackDecisionLogs, Severity = "CRITICAL"), "0")
#      Color: RGBA(200,60,0,1)
#    Small label: "Critical Decisions"

# 6. KPI Card 4 — Overdue Actions
#    Big number:
#      Text:  Text(CountIf(AutoFeedbackDecisionLogs, Status = "PENDING" && DueDate < Today()), "0")
#      Color: RGBA(180,0,0,1)
#    Small label: "Overdue Actions"

# 7. Navigation Buttons (3 across, equal width)
#    Button 1 — "View Issues"
#      OnSelect: Navigate(Screen2_Issues, ScreenTransition.Fade)
#      Fill: RGBA(0,120,212,1), TextColor: White
#
#    Button 2 — "View Decisions"
#      OnSelect: Navigate(Screen3_Decisions, ScreenTransition.Fade)
#      Fill: RGBA(0,120,212,1), TextColor: White
#
#    Button 3 — "View Learning"
#      OnSelect: Navigate(Screen5_Learning, ScreenTransition.Fade)
#      Fill: RGBA(100,100,100,1), TextColor: White


## ════════════════════════════════════════════════════════
## SCREEN 2 — ISSUES BREAKDOWN
## ════════════════════════════════════════════════════════
#
#  ┌─────────────────────────────────────────────────────┐
#  │  [< Back]   ISSUES BREAKDOWN                        │
#  ├─────────────────────────────────────────────────────┤
#  │  Filter: [ALL SENTIMENT ▼]   [ALL CLUSTERS ▼]       │
#  ├─────────────────────────────────────────────────────┤
#  │  M-PESA Pin Issues        ████████████  8  Negative │
#  │  SIM Swap                 ██████        4  Negative │
#  │  Reversal P2P             ████          3  Neutral  │
#  │  Paybill Reversal         ███           2  Neutral  │
#  ├─────────────────────────────────────────────────────┤
#  │  [tap any row to see sample comments]               │
#  └─────────────────────────────────────────────────────┘

## CONTROLS TO ADD:

# 1. Back Button
#    Text: "< Back"
#    OnSelect: Navigate(Screen1_Home, ScreenTransition.Back)

# 2. Screen Title
#    Text: "Issues Breakdown"
#    Size: 20, Bold

# 3. Sentiment Filter Dropdown
#    Name:  ddSentiment
#    Items: ["ALL", "Negative", "Neutral", "Positive"]

# 4. Cluster Filter Dropdown
#    Name:  ddCluster
#    Items: Distinct(AutoFeedbackAIInsightsOut, ClusterLabel)
#    Add "ALL" option by wrapping:
#      Items: ["ALL"] & Distinct(AutoFeedbackAIInsightsOut, ClusterLabel).Value

# 5. Issues Gallery (the bar chart effect)
#    Insert > Vertical Gallery
#    Name: galIssues
#    Items formula:
#      Sort(
#        AddColumns(
#          GroupBy(
#            Filter(
#              AutoFeedbackAIInsightsOut,
#              ddSentiment.Selected.Value = "ALL" || Sentiment = ddSentiment.Selected.Value,
#              ddCluster.Selected.Value   = "ALL" || ClusterLabel = ddCluster.Selected.Value
#            ),
#            "PainPoint", "Sentiment", "Records"
#          ),
#          "Count", CountRows(Records)
#        ),
#        Count, Descending
#      )
#
#    Inside each gallery row add:
#
#    a) Issue name label
#       Text:  ThisItem.PainPoint
#       Size:  13, Bold
#
#    b) Bar rectangle (the visual bar)
#       Width formula:
#         (ThisItem.Count / Max(galIssues.AllItems, Count)) * 300
#       Height: 12
#       Fill formula:
#         Switch(ThisItem.Sentiment,
#           "Negative", RGBA(200,0,0,1),
#           "Positive", RGBA(0,160,0,1),
#           RGBA(150,150,150,1)
#         )
#
#    c) Count label (right of bar)
#       Text:  Text(ThisItem.Count, "0")
#       Size:  13, Bold
#
#    d) Sentiment badge label
#       Text:  ThisItem.Sentiment
#       Color formula:
#         Switch(ThisItem.Sentiment,
#           "Negative", RGBA(200,0,0,1),
#           "Positive", RGBA(0,160,0,1),
#           RGBA(100,100,100,1)
#         )
#
#    OnSelect (row tap → show comments popup):
#      Set(selectedIssue, ThisItem); UpdateContext({showComments: true})

# 6. Comments Popup (hidden by default)
#    Insert > Rectangle (full screen overlay, semi-transparent)
#      Visible: showComments
#      Fill: RGBA(0,0,0,0.5)
#
#    Insert > Rectangle (white card, centered)
#      Visible: showComments
#
#    Comments Gallery inside the card:
#      Items: Filter(AutoFeedbackAIInsightsOut, PainPoint = selectedIssue.PainPoint)
#      Row text: ThisItem.SampleComments
#      Size: 12, Color: Dark gray
#
#    Close button:
#      Text: "X  Close"
#      OnSelect: UpdateContext({showComments: false})


## ════════════════════════════════════════════════════════
## SCREEN 3 — DECISIONS QUEUE
## ════════════════════════════════════════════════════════
#
#  ┌─────────────────────────────────────────────────────┐
#  │  [< Back]   DECISION QUEUE          [PENDING: 6]    │
#  ├─────────────────────────────────────────────────────┤
#  │  Filter: [ALL ENGINES ▼]  [ALL SEVERITY ▼]          │
#  ├─────────────────────────────────────────────────────┤
#  │  [CRITICAL] FRAUD  Fraud keywords detected          │
#  │             Due: 29 Mar 2026          OVERDUE  >    │
#  ├─────────────────────────────────────────────────────┤
#  │  [HIGH]     CHURN  3 negative in cluster            │
#  │             Due: 30 Mar 2026                    >   │
#  ├─────────────────────────────────────────────────────┤
#  │  [MEDIUM]   FEEDBACK  Predicted rise tomorrow       │
#  │             Due: 04 Apr 2026                    >   │
#  └─────────────────────────────────────────────────────┘

## CONTROLS TO ADD:

# 1. Back Button + Title + Pending Count badge (same row)
#    Pending count:
#      Text:  "PENDING: " & Text(CountIf(AutoFeedbackDecisionLogs, Status="PENDING"), "0")
#      Fill:  RGBA(200,0,0,1), TextColor: White, Radius: 12

# 2. Engine Filter Dropdown
#    Name:  ddEngine
#    Items: ["ALL","FEEDBACK","CHURN","FRAUD","SERVICE_RECOVERY"]

# 3. Severity Filter Dropdown
#    Name:  ddSeverity
#    Items: ["ALL","CRITICAL","HIGH","MEDIUM","LOW"]

# 4. Decisions Gallery
#    Items formula:
#      Sort(
#        Filter(
#          AutoFeedbackDecisionLogs,
#          Status = "PENDING",
#          ddEngine.Selected.Value   = "ALL" || Engine   = ddEngine.Selected.Value,
#          ddSeverity.Selected.Value = "ALL" || Severity = ddSeverity.Selected.Value
#        ),
#        DueDate, Ascending
#      )
#
#    Inside each row:
#
#    a) Severity badge
#       Text: ThisItem.Severity
#       Fill formula:
#         Switch(ThisItem.Severity,
#           "CRITICAL", RGBA(200,0,0,1),
#           "HIGH",     RGBA(230,100,0,1),
#           "MEDIUM",   RGBA(200,160,0,1),
#           "LOW",      RGBA(0,150,0,1),
#           RGBA(150,150,150,1)
#         )
#       TextColor: White, Radius: 4
#
#    b) Engine label
#       Text: ThisItem.Engine
#       Size: 11, Color: RGBA(0,120,212,1), Bold
#
#    c) Trigger label (the issue description)
#       Text: ThisItem.Trigger
#       Size: 12, Color: Dark
#       Overflow: Hidden (1 line only — keeps it clean)
#
#    d) Due date label
#       Text: "Due: " & Text(ThisItem.DueDate, "dd mmm yyyy")
#       Size: 11, Color: Gray
#
#    e) OVERDUE badge
#       Text:    "OVERDUE"
#       Visible: ThisItem.DueDate < Today() && ThisItem.Status = "PENDING"
#       Fill:    RGBA(200,0,0,1), TextColor: White, Radius: 4
#
#    f) Chevron ">"
#       Text: ">"
#       Size: 16, Color: Gray
#
#    OnSelect:
#      Set(selectedDecision, ThisItem);
#      Navigate(Screen4_Detail, ScreenTransition.Cover)


## ════════════════════════════════════════════════════════
## SCREEN 4 — CASE DETAIL + ACTION UPDATE
## ════════════════════════════════════════════════════════
#
#  ┌─────────────────────────────────────────────────────┐
#  │  [< Back]                          [CRITICAL] FRAUD │
#  ├─────────────────────────────────────────────────────┤
#  │  Cluster:   REVERSAL & FRAUD                        │
#  │  Trigger:   Fraud keywords detected in 3 records    │
#  │  Decision:  FREEZE accounts. Escalate to Security.  │
#  │  Due:       29 Mar 2026                             │
#  ├─────────────────────────────────────────────────────┤
#  │  Owner      [Fraud Team          ▼]                 │
#  │  Status     [IN_PROGRESS         ▼]                 │
#  │  Action     [________________________]              │
#  │  Notes      [________________________]              │
#  ├─────────────────────────────────────────────────────┤
#  │              [       SAVE & CLOSE       ]           │
#  └─────────────────────────────────────────────────────┘

## CONTROLS TO ADD:

# 1. Back Button
#    OnSelect: Navigate(Screen3_Decisions, ScreenTransition.Back)

# 2. Severity + Engine badge (top right, same as queue)

# 3. Read-only info labels (Cluster, Trigger, Decision, Due)
#    These are just labels — no editing needed
#    Trigger text wraps to 2 lines max (set Height accordingly)

# 4. Owner Dropdown
#    Name:  ddOwner
#    Items: ["UNASSIGNED","CX Team","Fraud Team","Network Team","Retention Team","Management"]
#    Default: selectedDecision.Owner

# 5. Status Dropdown
#    Name:  ddStatus
#    Items: ["PENDING","IN_PROGRESS","ACTIONED","CLOSED","ESCALATED"]
#    Default: selectedDecision.Status

# 6. Action Taken Text Input
#    Name:    txtAction
#    Default: selectedDecision.ActionTaken
#    HintText: "What action was taken?"
#    Height:  80 (multiline)

# 7. Notes Text Input
#    Name:    txtNotes
#    Default: selectedDecision.Notes
#    HintText: "Any additional notes..."
#    Height:  60

# 8. SAVE Button
#    Text: "Save & Close"
#    Fill: RGBA(0,120,212,1), TextColor: White
#    OnSelect:
#      Patch(
#        AutoFeedbackDecisionLogs,
#        LookUp(AutoFeedbackDecisionLogs, DecisionID = selectedDecision.DecisionID),
#        {
#          Owner:        ddOwner.Selected.Value,
#          Status:       ddStatus.Selected.Value,
#          ActionTaken:  txtAction.Text,
#          Notes:        txtNotes.Text,
#          ClosedDate:   If(ddStatus.Selected.Value = "CLOSED", Today(), Blank())
#        }
#      );
#      Notify("Saved successfully", NotificationType.Success);
#      Navigate(Screen3_Decisions, ScreenTransition.Back)


## ════════════════════════════════════════════════════════
## GLOBAL STYLES (apply to all screens for consistency)
## ════════════════════════════════════════════════════════

# App background:     RGBA(245,245,245,1)   ← light gray
# Card background:    White with shadow (use rectangle + slight border)
# Primary color:      RGBA(0,120,212,1)     ← blue  OR  RGBA(0,166,81,1) ← Safaricom green
# Danger color:       RGBA(200,0,0,1)
# Warning color:      RGBA(230,100,0,1)
# Success color:      RGBA(0,150,0,1)
# Font:               Segoe UI (default in Power Apps)
# Border radius:      8 on all cards, 4 on badges

# App.OnStart formula (set once, used everywhere):
#   Set(appPrimary,  RGBA(0,120,212,1));
#   Set(appDanger,   RGBA(200,0,0,1));
#   Set(appWarning,  RGBA(230,100,0,1));
#   Set(appSuccess,  RGBA(0,150,0,1));
#   Set(appBg,       RGBA(245,245,245,1))


## ════════════════════════════════════════════════════════
## DATA CONNECTIONS NEEDED (in order)
## ════════════════════════════════════════════════════════
# 1. AutoFeedbackDecisionLogs.xlsx  → Sheet: Analysis  (READ + WRITE)
# 2. AutoFeedbackAIInsightsOut.xlsx → Sheet: Analysis  (READ only)
# 3. AutoFeedbackAIPredictions.xlsx → Sheet: Analysis  (READ only)
#
# All must be in SharePoint document library (not personal OneDrive)
# Connect via: Data > Add data > Excel Online (Business) > SharePoint Sites

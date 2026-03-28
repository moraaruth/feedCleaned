# PA MOJA AI — POWER APPS SETUP GUIDE
# Connects your DecisionLogs Excel to a 3-screen Canvas App

## ARCHITECTURE
Python Engine → OneDrive Excel (DecisionLogs) → Power Apps Canvas App
                                               → Power BI Dashboard (optional)

---

## STEP 1: UPLOAD EXCEL TO SHAREPOINT (required for Power Apps)

Power Apps cannot read directly from OneDrive personal folders.
Move your output files to a SharePoint document library:

1. Go to https://safaricom.sharepoint.com
2. Create a Document Library called: PaMojaAI
3. Upload these files into it:
   - AutoFeedbackDecisionLogs.xlsx
   - AutoFeedbackAIActions.xlsx
   - AutoFeedbackAIInsightsOut.xlsx
   - AutoFeedbackAIPredictions.xlsx

4. Update your Python OUTPUT_MAPS paths to point to the SharePoint sync folder:
   SHAREPOINT_SYNC = r"C:\Users\RMNYANGAU\Safaricom PLC\PaMojaAI"

   The SharePoint folder syncs automatically via OneDrive for Business,
   so Python writes locally and SharePoint picks it up within seconds.

---

## STEP 2: CREATE THE CANVAS APP

1. Go to https://make.powerapps.com
2. Click: Create > Canvas App > Blank App > Tablet layout
3. Name it: PA MOJA AI Decision Centre

### Connect to Excel Data:
4. Click: Data (left panel) > Add data > OneDrive for Business
5. Select: AutoFeedbackDecisionLogs.xlsx > Sheet: Analysis
6. Repeat for: AutoFeedbackAIActions.xlsx, AutoFeedbackAIPredictions.xlsx
7. Rename connections in the Data panel:
   - DecisionLogs
   - Actions
   - Predictions

---

## STEP 3: SCREEN 1 — DASHBOARD (KPIs)

### Add these controls:

**Header Label**
  Text: "PA MOJA AI — Decision Centre"
  Font size: 24, Bold, Color: #0078D4

**KPI Card 1 — Total Pending Decisions**
  Insert > Label
  Text formula:
    "Pending: " & CountIf(DecisionLogs, Status = "PENDING")

**KPI Card 2 — Critical Alerts**
  Text formula:
    "Critical: " & CountIf(DecisionLogs, Severity = "CRITICAL")

**KPI Card 3 — Negative Sentiment %**
  Text formula:
    "Negative: " & Text(
        CountIf(AutoFeedbackAIInsightsOut, Sentiment = "Negative") /
        CountRows(AutoFeedbackAIInsightsOut) * 100,
        "0"
    ) & "%"

**KPI Card 4 — Overdue Actions**
  Text formula:
    "Overdue: " & CountIf(Actions, Status = "PENDING" && DueDate < Today())

**Severity Breakdown Gallery**
  Insert > Vertical Gallery
  Items formula:
    GroupBy(
        Filter(DecisionLogs, Status = "PENDING"),
        "Severity",
        "Items"
    )
  Title: ThisItem.Severity
  Subtitle: CountRows(ThisItem.Items) & " decisions"

**Navigate to Queue Button**
  Text: "View Decision Queue"
  OnSelect: Navigate(Screen2_Queue, ScreenTransition.Fade)

---

## STEP 4: SCREEN 2 — DECISION QUEUE

### Add these controls:

**Filter Dropdown — Severity**
  Insert > Dropdown
  Items: ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
  Name: ddSeverity

**Filter Dropdown — Engine**
  Items: ["ALL", "FEEDBACK", "CHURN", "FRAUD", "SERVICE_RECOVERY"]
  Name: ddEngine

**Filter Dropdown — Status**
  Items: ["ALL", "PENDING", "IN_PROGRESS", "ACTIONED", "CLOSED"]
  Name: ddStatus

**Decision Queue Gallery**
  Insert > Vertical Gallery
  Items formula:
    Sort(
        Filter(
            DecisionLogs,
            (ddSeverity.Selected.Value = "ALL" || Severity = ddSeverity.Selected.Value),
            (ddEngine.Selected.Value   = "ALL" || Engine   = ddEngine.Selected.Value),
            (ddStatus.Selected.Value   = "ALL" || Status   = ddStatus.Selected.Value)
        ),
        DueDate,
        Ascending
    )

  In each gallery row add:
    - Title label:    ThisItem.Cluster & " — " & ThisItem.Engine
    - Subtitle label: ThisItem.Trigger
    - Severity badge: ThisItem.Severity
      Fill color formula:
        Switch(ThisItem.Severity,
            "CRITICAL", RGBA(200,0,0,1),
            "HIGH",     RGBA(230,100,0,1),
            "MEDIUM",   RGBA(230,180,0,1),
            "LOW",      RGBA(0,150,0,1),
            RGBA(150,150,150,1)
        )
    - Due date label: "Due: " & Text(ThisItem.DueDate, "dd mmm yyyy")
    - Overdue indicator:
        Visible formula: ThisItem.DueDate < Today() && ThisItem.Status = "PENDING"
        Text: "OVERDUE"
        Color: Red

  OnSelect (row click): Navigate(Screen3_Detail, ScreenTransition.Cover)
                        Set(selectedDecision, ThisItem)

---

## STEP 5: SCREEN 3 — CASE DETAIL + ACTION UPDATE

### Add these controls:

**Decision ID Label**
  Text: "ID: " & selectedDecision.DecisionID

**Engine / Cluster / Severity Labels**
  Text: selectedDecision.Engine & " | " & selectedDecision.Cluster & " | " & selectedDecision.Severity

**Trigger Text**
  Text: selectedDecision.Trigger

**Decision Recommendation**
  Text: selectedDecision.Decision

**Owner Dropdown**
  Items: ["UNASSIGNED","CX Team","Fraud Team","Network Team","Retention Team","Management"]
  Default: selectedDecision.Owner

**Status Dropdown**
  Items: ["PENDING","IN_PROGRESS","ACTIONED","CLOSED","ESCALATED"]
  Default: selectedDecision.Status

**Action Taken Text Input**
  Default: selectedDecision.ActionTaken
  HintText: "Describe what action was taken..."

**Notes Text Input**
  Default: selectedDecision.Notes

**SAVE Button**
  Text: "Save & Update"
  OnSelect:
    Patch(
        DecisionLogs,
        LookUp(DecisionLogs, DecisionID = selectedDecision.DecisionID),
        {
            Status:       ddStatus_detail.Selected.Value,
            Owner:        ddOwner.Selected.Value,
            ActionTaken:  txtActionTaken.Text,
            Notes:        txtNotes.Text,
            ClosedDate:   If(ddStatus_detail.Selected.Value = "CLOSED", Today(), Blank())
        }
    );
    Notify("Decision updated successfully", NotificationType.Success);
    Navigate(Screen2_Queue, ScreenTransition.Back)

**Back Button**
  OnSelect: Navigate(Screen2_Queue, ScreenTransition.Back)

---

## STEP 6: POWER AUTOMATE — AUTO-NOTIFY ON CRITICAL DECISIONS

1. Go to https://make.powerautomate.com
2. Create flow: Automated > When a row is added (Excel Online Business)
   - File: AutoFeedbackDecisionLogs.xlsx
   - Table: Analysis

3. Add condition: Severity is equal to CRITICAL

4. If yes → Send an email (V2):
   To:      your-team@safaricom.com
   Subject: "[PA MOJA AI] CRITICAL: " & Trigger
   Body:    "Engine: " & Engine & "\nCluster: " & Cluster
            & "\nDecision: " & Decision & "\nDue: " & DueDate

5. If yes → Post message in Teams channel (optional):
   Channel: CX Operations
   Message: same as email body

---

## STEP 7: POWER BI (optional — read-only analytics)

1. Open Power BI Desktop
2. Get Data > Excel > AutoFeedbackDecisionLogs.xlsx
3. Also load: AutoFeedbackAIInsightsOut.xlsx, AutoFeedbackAIPredictions.xlsx

### Recommended visuals:
- Card: Count of PENDING decisions
- Donut chart: Decisions by Severity
- Bar chart: Decisions by Engine (FEEDBACK / CHURN / FRAUD / SERVICE_RECOVERY)
- Line chart: Daily negative sentiment trend (from Feedback sheet)
- Table: DecisionLogs filtered to Status = PENDING, sorted by DueDate
- Slicer: Engine, Severity, Status, Date range

4. Publish to Power BI Service
5. Pin to Teams tab for your CX Operations channel

---

## FILE STRUCTURE SUMMARY

pamoja_ai/
  analyze_feedback.py        ← main engine (runs daily via Task Scheduler)
  engines/
    churn.py                 ← churn detection engine
    fraud.py                 ← fraud detection engine
    service_recovery.py      ← service recovery engine
  powerapps_setup.md         ← this file

OneDrive/SharePoint outputs:
  AutoFeedbackDecisionLogs.xlsx   ← MASTER — Power Apps reads/writes this
  AutoFeedbackAIActions.xlsx      ← task queue
  AutoFeedbackAIInsightsOut.xlsx  ← processed feedback + sentiment
  AutoFeedbackAIPredictions.xlsx  ← cluster forecasts
  AutoFeedbackAIAlerts.xlsx       ← raw alerts
  AutoFeedbackAIOutcomes.xlsx     ← outcome tracking
  AutoFeedbackAILearning.xlsx     ← lessons learned log

---

## SCHEDULE THE ENGINE (Windows Task Scheduler)

1. Open Task Scheduler > Create Basic Task
2. Name: PA MOJA AI Daily Run
3. Trigger: Daily at 07:00 AM
4. Action: Start a program
   Program: C:\Users\RMNYANGAU\AppData\Local\Python\pythoncore-3.14-64\python.exe
   Arguments: C:\Users\RMNYANGAU\Desktop\pamoja_ai\analyze_feedback.py
5. Finish

The engine runs every morning, updates all Excel files,
Power Apps reflects the new decisions automatically.

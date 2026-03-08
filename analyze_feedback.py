# ============================================
# PA MOJA AI FEEDBACK ENGINE
# ============================================

import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# ============================================
# LOAD DATA
# ============================================

df = pd.read_excel("AutoFeedback.xlsx")

# Clean column names
df.columns = df.columns.str.strip()

print("\nColumns detected in file:")
print(df.columns)

# ============================================
# DETECT TEXT COLUMNS AUTOMATICALLY
# ============================================

# Adjust these depending on your Excel columns
text_columns = []

if "PainPoint" in df.columns:
    text_columns.append("PainPoint")

if "Description" in df.columns:
    text_columns.append("Description")

if "Feedback" in df.columns:
    text_columns.append("Feedback")

if "Title" in df.columns:
    text_columns.append("Title")

# Combine text columns
df["combined_text"] = df[text_columns].astype(str).agg(" ".join, axis=1)

print("\nData loaded and cleaned:")
print(df.head())

# ============================================
# SENTIMENT ANALYSIS
# ============================================

def get_sentiment(text):

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["combined_text"].apply(get_sentiment)

print("\nSentiment added:")
print(df[["combined_text","Sentiment"]].head())

# ============================================
# CLUSTER SIMILAR ISSUES
# ============================================

vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(df["combined_text"])

kmeans = KMeans(n_clusters=3, random_state=42)

df["Cluster"] = kmeans.fit_predict(X)

print("\nClusters created:")
print(df[["combined_text","Cluster"]].head())

# ============================================
# CLUSTER LABELS
# ============================================

cluster_labels = {}

for cluster in df["Cluster"].unique():

    issues = df[df["Cluster"] == cluster]["combined_text"].head(2)

    label = " | ".join(issues)

    cluster_labels[cluster] = label

df["ClusterLabel"] = df["Cluster"].map(cluster_labels)

print("\nCluster Labels:")
print(cluster_labels)

# ============================================
# DATE PROCESSING
# ============================================

# Detect date column
date_column = None

for col in df.columns:
    if "date" in col.lower() or "created" in col.lower():
        date_column = col
        break

df["Date"] = pd.to_datetime(df[date_column])

daily_counts = df.groupby(["Date","ClusterLabel"]).size().reset_index(name="Count")

# ============================================
# SPIKE DETECTION
# ============================================

alerts = []

for cluster in daily_counts["ClusterLabel"].unique():

    cluster_data = daily_counts[daily_counts["ClusterLabel"] == cluster].sort_values("Date")

    cluster_data["MovingAvg"] = cluster_data["Count"].rolling(7, min_periods=1).mean()

    cluster_data["IncreaseRatio"] = cluster_data["Count"] / cluster_data["MovingAvg"]

    spikes = cluster_data[cluster_data["IncreaseRatio"] > 1.5]

    for _, row in spikes.iterrows():

        alerts.append({
            "Date": row["Date"],
            "Alert": f"Spike detected in {cluster} with {row['Count']} complaints"
        })

print("\nSpike Alerts:")
print(alerts)

alerts_df = pd.DataFrame(alerts)

# ============================================
# PREDICTION ENGINE
# ============================================

predictions = []

for cluster in daily_counts["ClusterLabel"].unique():

    cluster_data = daily_counts[daily_counts["ClusterLabel"] == cluster].sort_values("Date")

    cluster_data["DayIndex"] = np.arange(len(cluster_data))

    X_train = cluster_data[["DayIndex"]]
    y_train = cluster_data["Count"]

    model = LinearRegression()

    model.fit(X_train, y_train)

    next_day = np.array([[len(cluster_data)]])

    pred = max(0, int(model.predict(next_day)[0]))

    predictions.append({
        "Cluster": cluster,
        "PredictedComplaints": pred
    })

predictions_df = pd.DataFrame(predictions)

print("\nPredictions for tomorrow:")
print(predictions_df)

# ============================================
# EXECUTIVE AI INSIGHT
# ============================================

top_issue = predictions_df.sort_values("PredictedComplaints", ascending=False).iloc[0]

insight = f"""
Customers are most likely to experience issues related to {top_issue['Cluster']} tomorrow.
Approximately {top_issue['PredictedComplaints']} complaints are expected based on recent trends.
"""

insight_df = pd.DataFrame([{
    "Date": pd.Timestamp.today(),
    "Insight": insight
}])

print("\nExecutive Insight:")
print(insight)

# ============================================
# SAVE RESULTS
# ============================================

df.to_excel("AI_Feedback_Processed.xlsx", index=False)

predictions_df.to_excel("AI_Predictions.xlsx", index=False)

alerts_df.to_excel("AI_Alerts.xlsx", index=False)

insight_df.to_excel("AI_Insights.xlsx", index=False)

print("\nAI Processing Complete")
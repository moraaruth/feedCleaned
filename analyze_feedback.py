# -------------------------------
# 1. Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
import requests
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# -------------------------------
# 2. Load Feedback Data
# -------------------------------
df = pd.read_excel("feedbackCollected.xlsx")

# FIX 1: Fill empty comments with a blank space so they are strings, not "NaN"
df['SampleComments'] = df['SampleComments'].fillna("").astype(str)

# FIX 2: Create a combined column so we ALWAYS have data to test
# This merges "Sim Card Issues" + " " + "Customer comment"
df['combined_text'] = df['PainPoint'].astype(str) + " " + df['SampleComments']

df['MonthDate'] = pd.to_datetime(df['MonthDate'])

print("Data loaded and cleaned:")
print(df.head())

# -------------------------------
# 3. Sentiment Analysis (NLP)
# -------------------------------
def get_sentiment(text):
    # TextBlob needs a string; our cleaning above ensures it gets one
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Use the combined text so we analyze the mood of the category AND the comment
df['Sentiment'] = df['combined_text'].apply(get_sentiment)

print("\nSentiment added:")
print(df[['PainPoint', 'Sentiment']].head())

# -------------------------------
# 4. Cluster Similar Complaints
# -------------------------------
vectorizer = TfidfVectorizer(stop_words='english')

# Cluster based on the combined text
X = vectorizer.fit_transform(df['combined_text'])

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
df['Cluster'] = kmeans.labels_

print("\nClusters created:")
print(df[['PainPoint', 'Cluster']].head())

# -------------------------------
# 5. Detect Spikes
# -------------------------------
daily_counts = df.groupby([df['MonthDate'], 'Cluster']).size().reset_index(name='Count')

threshold = 1.5
alerts = []

for cluster in daily_counts['Cluster'].unique():
    cluster_data = daily_counts[daily_counts['Cluster']==cluster].sort_values('MonthDate')
    cluster_data['MovingAvg'] = cluster_data['Count'].rolling(7, min_periods=1).mean()
    cluster_data['IncreaseRatio'] = cluster_data['Count'] / cluster_data['MovingAvg']
    
    spikes = cluster_data[cluster_data['IncreaseRatio'] > threshold]
    for index, row in spikes.iterrows():
        alerts.append(
            f"Spike detected: Cluster {cluster} on {row['MonthDate'].date()} with {row['Count']} complaints"
        )

print("\nSpike Alerts:")
print(alerts)

# -------------------------------
# 6. Predict Tomorrow's Complaints
# -------------------------------
predictions = {}
for cluster in daily_counts['Cluster'].unique():
    cluster_data = daily_counts[daily_counts['Cluster']==cluster].sort_values('MonthDate')
    cluster_data['DayIndex'] = np.arange(len(cluster_data))
    
    X_train = cluster_data[['DayIndex']]
    y_train = cluster_data['Count']
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    next_day_index = np.array([[len(cluster_data)]])
    next_count = max(0, int(model.predict(next_day_index)[0]))
    predictions[cluster] = next_count

print("\nPredictions for tomorrow:")
print(predictions)

# See what is in each cluster
print("\nCluster Key (What issues are in which cluster):")
print(df.groupby('Cluster')['PainPoint'].unique())
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

df = pd.read_excel("feedback.xlsx")

df['Date'] = pd.to_datetime(df['Date'])

print("Data loaded:")
print(df.head())


# -------------------------------
# 3. Sentiment Analysis (NLP)
# -------------------------------

def get_sentiment(text):

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.1:
        return "Positive"

    elif polarity < -0.1:
        return "Negative"

    else:
        return "Neutral"


df['Sentiment'] = df['Feedback'].apply(get_sentiment)

print("\nSentiment added:")
print(df.head())


# -------------------------------
# 4. Cluster Similar Complaints
# -------------------------------

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(df['Feedback'])

kmeans = KMeans(n_clusters=3, random_state=42)

kmeans.fit(X)

df['Cluster'] = kmeans.labels_

print("\nClusters created:")
print(df.head())


# -------------------------------
# 5. Detect Spikes
# -------------------------------

daily_counts = df.groupby([df['Date'], 'Cluster']).size().reset_index(name='Count')

threshold = 1.5

alerts = []

for cluster in daily_counts['Cluster'].unique():

    cluster_data = daily_counts[daily_counts['Cluster']==cluster].sort_values('Date')

    cluster_data['MovingAvg'] = cluster_data['Count'].rolling(7, min_periods=1).mean()

    cluster_data['IncreaseRatio'] = cluster_data['Count'] / cluster_data['MovingAvg']

    spikes = cluster_data[cluster_data['IncreaseRatio'] > threshold]

    for index, row in spikes.iterrows():

        alerts.append(
            f"Spike detected: Cluster {cluster} on {row['Date'].date()} with {row['Count']} complaints"
        )

print("\nSpike Alerts:")
print(alerts)


# -------------------------------
# 6. Predict Tomorrow's Complaints
# -------------------------------

predictions = {}

for cluster in daily_counts['Cluster'].unique():

    cluster_data = daily_counts[daily_counts['Cluster']==cluster].sort_values('Date')

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


# -------------------------------
# 7. Send Alert to Teams
# -------------------------------

# teams_webhook = "YOUR_TEAMS_WEBHOOK_URL"

#for cluster, pred in predictions.items():

 #   message = f"Prediction: Cluster {cluster} may receive {pred} complaints tomorrow."

   # requests.post(teams_webhook, json={"text": message})

#print("\nAlerts sent to Teams!")
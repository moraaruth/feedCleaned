import pandas as pd

# Load feedback from Excel
df = pd.read_excel("feedback.xlsx")

# Convert date column to datetime
df['Date'] = pd.to_datetime(df['MonthDate'])

print(df.head())
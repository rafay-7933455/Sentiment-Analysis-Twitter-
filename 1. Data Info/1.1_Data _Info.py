import pandas as pd

df = pd.read_csv('twitter_training.csv')

df['Sentiment'] = df['Positive']
df['Id'] = df['2401']
df['Tool'] = df['Borderlands']
df['Text'] = df['im getting on borderlands and i will murder you all ,']
df = df[['Sentiment','Id','Tool','Text']]
df.to_csv('refined.csv', index=False)
# (Rows, Columns)
print("Rows, Columns:",df.shape)
# Names of all Columns
print("Columns:",df.columns)
# Statistical Summary
print("Description:",df.describe())
# Data types of all columns
df.info()
# No. of null values in each column
print("Nulls:\n",df.isnull().sum(), sep='')
# Leading Lines
print(df.head())
# Ending Lines
print(df.tail())
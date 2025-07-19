import pandas as pd

df = pd.read_csv('refined.csv')

# Removing Null Text Value Records
print("Null Count:",df.isnull().sum(), sep='\n')
df.dropna(subset=['Text'], inplace=True)
print("Null Count:",df.isnull().sum(), sep='\n')

# Removing Duplicated Records
print("No. of Duplicated Records:", df.duplicated().sum())
dup = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"{dup} duplicates removed")

# Saving the cleaned dataset
df.to_csv('cleaned.csv', index=False)
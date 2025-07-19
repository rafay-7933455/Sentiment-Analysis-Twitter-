import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib as jb

df = pd.read_csv('cleaned.csv')

old_col = ['Id','Text']

num = df.select_dtypes(include=['int64','float64']).columns.to_list()
cat = df.select_dtypes(include=['object']).columns.to_list()

df_num = df.drop(columns=cat)
df_cat = df.drop(columns=num).drop(columns=['Text', 'Sentiment'])

# Scaling Id column
scaler = StandardScaler()
std = scaler.fit_transform(df_num)
df[scaler.get_feature_names_out()] = std

# Encoing Sentiments
L_encoder = LabelEncoder()
df['Sentiment'] = L_encoder.fit_transform(df['Sentiment'])
jb.dump(L_encoder, 'models/lencoder.pkl')

# Saving the data
df.to_csv('scaled.csv', index=False)

# Encoding String columns other than 'Text'
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df_cat)
df[encoder.get_feature_names_out()] = encoded
# Saving the model
jb.dump(value=encoder, filename='models/encoder.pkl')


# Vectorizing Text column
vectorizer = TfidfVectorizer()
vectorized = vectorizer.fit_transform(df['Text'])
# df_new = pd.DataFrame.sparse.from_spmatrix(vectorized)
# print(df_new)
# df = pd.concat([df, df_new], axis=1)
# print(df)
# Saving the model
jb.dump(vectorizer,'models/vectorizer.pkl')
# print(vectorizer.vocabulary_)

# Display the Resultant DataFrame
# df_v = pd.DataFrame.sparse.from_spmatrix(vectorized, columns=vectorizer.get_feature_names_out())
# df = pd.concat([df, df_v], axis=1)
# print(df.drop(columns=old_col))
# print(df.columns)
# print()
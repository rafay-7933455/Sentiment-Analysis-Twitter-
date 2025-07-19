import pandas as pd
import joblib as jb
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scipy import sparse

df = pd.read_csv('scaled.csv')
old_cols = df.columns.to_list()
old_cols.remove('Sentiment')
old_cols.remove('Id')

cat = df.select_dtypes(include=['object']).columns.to_list()
cat.remove('Text')

encoder = jb.load('models/encoder.pkl')
vectorizer = jb.load('models/vectorizer.pkl')

df[encoder.get_feature_names_out()] = encoder.transform(df[cat])

vectorized = vectorizer.transform(df['Text'])
# df_vectorized_text = pd.DataFrame.sparse.from_spmatrix(vectorized)

# Create dense matrix for numeric + encoded categorical
dense_features = df.select_dtypes(include=['int64', 'float64']).drop(columns=['Sentiment']).values
dense_sparse = sparse.csr_matrix(dense_features)

# Stack sparse
X = sparse.hstack([dense_sparse, vectorized])
y = df['Sentiment']

df.drop(columns=old_cols, inplace=True)

LRmodel = LogisticRegression(solver='saga', max_iter=100, random_state=42)
XGBmodel = XGBClassifier()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

LRmodel.fit(X_train, y_train)
XGBmodel.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score
y_pred_lr = LRmodel.predict(X_test)
y_pred_xgb = XGBmodel.predict(X_test)

print("LR Accuracy:", classification_report(y_test, y_pred_lr), sep='\n')
print("XGB Accuracy:", classification_report(y_test, y_pred_xgb), sep='\n')

print("LR Accuracy:", accuracy_score(y_test, y_pred_lr), sep='\n')
print("XGB Accuracy:", accuracy_score(y_test, y_pred_xgb), sep='\n')

jb.dump(LRmodel, 'models/LRmodel.pkl')
jb.dump(XGBmodel, 'models/XGBmodel.pkl')
import joblib as jb
import numpy as np
import warnings as war

war.filterwarnings('ignore')

model_lr = jb.load('models/LRmodel.pkl')
model_xgb = jb.load('models/XGBmodel.pkl')

text=input("Your Text (alphanumeric):")
id=input("Your Id (numeric):")
if not id.isnumeric():
     print("Error: Id must be numeric only")
     exit()
tool=input("Your Tool (alphanumeric):")

scaler = jb.load('models/scaler.pkl')
L_encoder = jb.load('models/lencoder.pkl')
encoder = jb.load('models/encoder.pkl')
vectorizer = jb.load('models/vectorizer.pkl')

x = np.hstack([scaler.transform(np.array([[id]])), encoder.transform([[tool]]), vectorizer.transform([text]).toarray()])

y_lr = model_lr.predict(x)
y_xgb = model_xgb.predict(x)

print("Logistic Regressor Response: Your sentiment was {}".format(L_encoder.inverse_transform(y_lr)[0]))
print("XGBoost Classifier Response: Your sentiment was {}".format(L_encoder.inverse_transform(y_lr)[0]))

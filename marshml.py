import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
#resampling library
from imblearn.over_sampling import SMOTE
import pickle
#feature scaling library
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
df = pd.read_csv("car.csv")
print(df.head())
df["AGE"].replace({"16-25": "Young", "26-39": "Middle Age","40-64":"Old","65+":"Very Old"}, inplace=True)
df["DRIVING_EXPERIENCE"].replace({"0-9y": "Newbie", "10-19y": "Amateur","20-29y":"Advanced","30y+":"Expert"}, inplace=True)


print(df.isnull().sum())

missing_col = ['ANNUAL_MILEAGE',"CREDIT_SCORE"]

for i in missing_col:
    df.loc[df.loc[:,i].isnull(),i]=df.loc[:,i].mean()


print(df.isnull().sum())
df.head()

df = df.drop_duplicates()


df.head()


model = LogisticRegression(solver='liblinear', random_state=0)

age = pd.get_dummies(df["AGE"],drop_first =True)
gender = pd.get_dummies(df["GENDER"],drop_first =True)
experience = pd.get_dummies(df["DRIVING_EXPERIENCE"],drop_first =True)
education = pd.get_dummies(df["EDUCATION"],drop_first =True)
income = pd.get_dummies(df["INCOME"],drop_first =True)

df = df.drop(["ID","AGE","DRIVING_EXPERIENCE","EDUCATION","INCOME","GENDER","RACE","VEHICLE_YEAR","POSTAL_CODE","VEHICLE_TYPE"],axis = 1)

df = pd.concat([age,gender,experience,education,income,df]
    ,axis=1)
print(df)
X, y = df.iloc[:,:-1],df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=0.33, random_state=42)

model.fit(X_train,y_train)


pickle.dump(model, open('model.pkl', 'wb'))

y_pred = model.predict(X_test)
score_logreg = accuracy_score(y_test, y_pred)
print(score_logreg)
data = df.iloc[:,:-1]
print(data)
print(data.iloc[:1])
print(data.iloc[:1].columns.tolist())
model.predict(data.iloc[:1])

k = model.predict_proba(data.iloc[:1]) #1 is claimed, 0 is not proba = [no claim,claim]
print(k[0][0])

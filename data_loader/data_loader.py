from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import pickle
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('C:\Datasets\DVC + MLflow project\synthetic_coffee_health_10000.csv')
X = df.drop('Sleep_Quality', axis = 1)
X = pd.get_dummies(X)
y = df.Sleep_Quality
enc = LabelEncoder()
y = enc.fit_transform(y.values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

pd.concat([X_train, pd.DataFrame(y_train)], axis=1).to_csv('data_loader/train.csv', index=False)
pd.concat([X_test, pd.DataFrame(y_test)], axis=1).to_csv('data_loader/test.csv', index=False)
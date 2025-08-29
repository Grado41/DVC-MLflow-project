from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
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

mlflow.set_experiment('Sleep_Quality_log_reg')

with mlflow.start_run():
    penalty='elasticnet'                 
    solver='saga' 
    max_iter=1000
    C_param = 0.5
    l1_ratio = 0.1
    clf=LogisticRegression(penalty=penalty,
                        solver=solver, 
                        max_iter=max_iter,
                        C = C_param,
                        l1_ratio = l1_ratio,
                        random_state=101)
    clf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    mlflow.log_param("model_type", 'LogisticRegression')
    mlflow.log_param("solver", solver)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("C", C_param)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(clf, "model")
    mlflow.log_param('train_size', len(X_train))
    mlflow.log_param('test_size', len(X_test))

    with open('models/log_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print(f"Accuracy: {accuracy:.4f}")


mlflow.set_experiment('Sleep_Quality_log_reg')

with mlflow.start_run():
    alpha = 0.5 
    n_estimators = 20
    max_depth = 10
    clf=XGBClassifier(alpha=alpha,
                        n_estimators=n_estimators, 
                        max_depth=max_depth,
                        random_state=101)
    clf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    mlflow.log_param("model_type", 'XGBClassifier')
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(clf, "model")
    mlflow.log_param('train_size', len(X_train))
    mlflow.log_param('test_size', len(X_test))

    with open('models/xgbc_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print(f"Accuracy: {accuracy:.4f}")
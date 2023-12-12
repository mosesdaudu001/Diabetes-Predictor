import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score

df = pd.read_csv('train.csv')

q_cutoff = df['Insulin'].quantile(0.95)
mask = df['Insulin'] < q_cutoff
trimmed_df = df[mask]

X = trimmed_df.drop(columns = 'Outcome', axis = 1)
Y = trimmed_df['Outcome']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

model_SVC = svm.SVC(kernel = 'poly')

model_SVC.fit(X_train, Y_train)

SVC_train_pred = model_SVC.predict(X_train)
SVC_test_pred = model_SVC.predict(X_test)

print('SVC Model Accuracy Train:',accuracy_score(Y_train, SVC_train_pred))

import pickle

filename = 'diabetes_svm_model.pkl'
pickle.dump(model_SVC, open(filename, 'wb'))

pickle.dump(scaler, open('scaler_min_max.pkl', 'wb'))
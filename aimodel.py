#!/usr/bin/env python
# coding: utf-8

import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression


housing = datasets.fetch_california_housing()


housing


df = pd.DataFrame(housing.data, columns=housing.feature_names)


df.head()


print(housing.DESCR)


dfTarget = pd.DataFrame(housing.target, columns=housing.target_names)


dfTarget.head()


X = df
y = dfTarget
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


scaler = StandardScaler()
X_train_Scaled = scaler.fit_transform(X_train)
X_test_Scaled = scaler.fit_transform(X_test)


reg = LinearRegression().fit(X_train_Scaled, y_train)
results = reg.predict(X_test_Scaled)


reg.score(X_test_Scaled, y_test)


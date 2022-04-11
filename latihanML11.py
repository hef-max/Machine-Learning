import pandas as pd
import sklearn.model_selection as ms
import sklearn.ensemble as se
from sklearn.tree import export_graphviz
import os
import pydotplus.graphviz as pg


rf = se.RandomForestClassifier(n_estimators=100)


df = pd.read_csv('dataset/decisiontree_ch6.csv')
encoding = {
            "mesin": {"bensin": 0, "diesel": 1},
            "penggerak": {"depan": 0, "belakang": 1}
            }

df.replace(encoding, inplace=True)
X = df.drop(['ID', 'label'], axis=1)
y = df['label']
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2)
f = rf.fit(X_train, y_train)
print(f)
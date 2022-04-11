import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('kawalcovid_30sept21.csv')
df.drop(columns=(['Provinsi', 'WN', 'Gender', 'Unnamed: 8', 'MD', 'DP', '?', 'Status']), inplace=True)

df1 = df.fillna(np.mean(df))

model = LinearRegression()
X = df['Umur']
y = df[['Sumber Kontak']]

X_train, y_train, X_test, y_test = train_test_split(X, y, train_size=0.5)

train = model.fit(X_train, y_train)
prediksi = train.predict(X_test)
print(prediksi)


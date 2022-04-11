import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

df1 = pd.read_csv('dataset/bensin2.csv')

X = df1[['Liter', 'Penumpang', 'Suhu', 'Kecepatan']]
Y = df1[['Kilometer']]

#train dataset
X_train, X_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.2, random_state=0) #test size hanya 20%

#pemodelan
model = lm.LinearRegression()
model.fit(X_train, y_train)
print("intercept=", model.intercept_)
print("slope=", model.coef_)

#uji coba
Bensin = int(input("Bensin : "))
Penumpang = int(input("Penumpang : "))
Suhu = int(input("Suhu : "))
Kecepatan = int(input("Kecepatan : "))

data = np.array([[Bensin, Penumpang, Suhu, Kecepatan]]) #np.array([[30, 2, 10, 50]])
print(f"{data[0,0]} Liter, {data[0,1]} Penumpang, {data[0,2]} drajat, {data[0,3]} KM/h")
hasil = model.predict(data)
print(f"jarak yang ditempuh : {hasil} KM")

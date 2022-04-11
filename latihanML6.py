import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import pyspark as spark
from sklearn.ensemble import RandomForestRegressor

df1 = pd.read_csv('dataset/bensin.csv')

liter = df1['Liter']
kilometer = df1['Kilometer']
X_train, X_test, y_train,y_test = ms.train_test_split(liter, kilometer, test_size=0.2, random_state=0)
plt.scatter(X_train, y_train, edgecolors='r')
plt.xlabel('liter')
plt.ylabel('kilometer')
plt.title('Konsumsi Bahan Bakar')
x1 = np.linspace(0,45)
y1 = 3.94 + 6.67 * x1
plt.plot(x1, y1)
plt.show()


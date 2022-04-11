import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np

df= pd.read_csv('dataset/[Dataset]_Module8_Train_(Employee).csv')

X = df['Attrition_rate']
y = df['Age']

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=0)
plt.scatter(X_train, y_train, edgecolors='r')
plt.xlabel('Attrition_rate')
plt.ylabel('Age')
plt.title('employee')
x1 = np.linspace(0,45)
y1 = 3.94 + 6.67 * x1
plt.plot(x1, y1)
plt.show()

import pandas as pd
import sklearn.neural_network as ann
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.metrics as met

df = pd.read_csv('dataset/datatraining.txt', header=0, names=['id', 'date', 'Temperature', 'Humidity', 'Light', "CO2", 'Humidityradio', "Occupancy"])
X = df.drop(['id', 'date'], axis=1)
y = df['Occupancy']

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2)

scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
scl.fit(X_train)

X_train = scl.transform(X_train)
X_test = scl.transform(X_test)
print(X_train.min())
print(X_train.max())

mlp = ann.MLPClassifier(hidden_layer_sizes=(3), max_iter=(5), activation='logistic')
mlp.fit(X_train, y_train)

y_prediksi = mlp.predict(X_test)

met.classification_report(y_test, y_prediksi)

print(mlp.coefs_)

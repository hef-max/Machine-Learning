#memprediksi menggunaakn chur prediction


import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.feature_selection as fs

import numpy as np

datasets = pd.read_excel('dataset/satgas-covid-19-dp_cvd_pcr_agregasi_provinsi_data.xlsx')

df = pd.DataFrame( data=datasets)


X = df.drop(['id', 'hari', 'tanggal', 'hari', 'invalid', 'negatif', 'jumlah_sampling'], axis=1)
encoding = {"hari" : {'Senin': 0,
                      'Selasa': 1,
                      'Rabu' : 2,
                      'Kamis': 3,
                      "Jum'at": 4,
                      'Jumat':4,
                      "jum'at":4,
                      'Sabtu': 5,
                      'Minggu': 6
                      }
            }
df.replace(encoding, inplace=False)
y = df['hari']

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=0)

plt.scatter(X_train, y_train, edgecolors='r')
plt.ylabel('hari')
plt.xlabel('poitif covid')
plt.show()








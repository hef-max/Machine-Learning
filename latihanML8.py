import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as met
import matplotlib.pyplot as plt

df1 = pd.read_csv('dataset/calonpembeli_ch5.csv')
df1 = df1[df1['Usia'] < 100]
X = df1[['Usia', 'Status', 'Kelamin', 'Memiliki_Mobil', 'Penghasilan']]
y = df1['Beli_Mobil']
#train
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=0) #test prediksi 20%
#permodelan
model = lm.LogisticRegression(solver='lbfgs')
model = model.fit(X_train, y_train)

y_prediksi = model.predict(X_test)
print(X_test.head())
print("\n")
print(y_test.head(1))

print("\n")
confusionmatrix = met.confusion_matrix(y_test, y_prediksi)
print(confusionmatrix)

print("\n")
score =model.score(X_test, y_test)
print(score)


y_pred_proba = model.predict_proba(X_test) [::,1]
fp, tp, _ = met.roc_curve(y_test, y_pred_proba)
auc = met.roc_auc_score(y_test, y_pred_proba)
plt.plot(fp, tp, label="data 1 , auc="+str(auc))
plt.legend(loc=4)
plt.show()
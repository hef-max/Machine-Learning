# ALGORITMA NAIVE BAYES


import pandas as pd
import sklearn.model_selection as ms
import sklearn.naive_bayes as nb
import sklearn.metrics as met
import matplotlib.pyplot as plt

df_train = pd.read_csv('dataset/train.csv')
df_train.drop(['Cabin'], axis=1, inplace=True)
df_train['Embarked'].value_counts()
df_train['Embarked'].fillna('S', inplace=True)
embarked = {"Embarked": {"S": 0, "C": 1, "Q": 2}}
df_train.replace(embarked, inplace=True)
df_train.dropna(inplace=True, how='any')
df_train['Fare'] = df_train['Fare'].astype(int)
df_train['Age'] = df_train['Age'].astype(int)
df_train = df_train.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
sex = {"Sex": {"male":0, "female":1}}
df_train.replace(sex, inplace=True)

features = df_train[['Pclass', 'Embarked', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']]
label = df_train['Survived']
X_train, X_test, y_train, y_test = ms.train_test_split(features, label, test_size=0.25, random_state=0)

gnb = nb.GaussianNB()
gnb.fit(X_train, y_train)

y_prediksi = gnb.predict(X_test)
accuracy = met.accuracy_score(y_test, y_prediksi)
precisions = met.precision_score(y_test, y_prediksi)
print("accuracy = ",accuracy, "precisions=", precisions)

y_pred_proba = gnb.predict_proba(X_test)[::,1]
fp, tp, _ = met.roc_curve(y_test, y_pred_proba)
auc = met.roc_auc_score(y_test, y_pred_proba)
plt.plot(fp, tp, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
df = pd.read_csv('dataset/churnprediction_ch9.csv', sep=',', index_col=['customer_id'])

diaktifkan = df.groupby('churn').count()

# plt.pie(diaktifkan['product'], labels=['Aktif', 'Churn'], autopct='%1.0f%%')
# plt.title('Persentase Pelanggan Aktif vs Churn')
# plt.axis('equal')

df['product'].value_counts()

df2 = pd.concat([df, pd.get_dummies(df['product'])], axis=1, sort=False)
df2.drop(['product'], axis=1, inplace=True)

dfkorelasi = df2.corr()
sns.heatmap(dfkorelasi, xticklabels=df2.columns.values, yticklabels=dfkorelasi.columns, annot=True, annot_kws={'size': 12})
heat_map=plt.gcf()
heat_map.set_size_inches(10, 10)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

X = df2.drop(['reload_2', 'socmed_2', 'games', 'churn'], axis=1, inplace=False)
y = df2['churn']

X_test, X_train, y_test, y_train = ms.train_test_split(X, y, test_size=0.8, random_state=0)

scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
scl.fit(X_train)
X_train = scl.transform(X_train)
X_test = scl.transform(X_test)

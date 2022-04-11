import pandas as pd
import sklearn.model_selection as ms
import sklearn.tree as tree
import sklearn.metrics as met
from sklearn.tree import export_graphviz
import os
from IPython.display import Image
import pydotplus
import graphviz

df1 = pd.read_csv('dataset/decisiontree_ch6.csv')
print(df1.head())
encoding = {
            "mesin": {"bensin": 0, "diesel": 1},
            "penggerak": {"depan": 0, "belakang": 1}
            }

df1.replace(encoding, inplace=True)
X = df1.drop(['ID', 'label'], axis=1)
y = df1['label']
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2)

model1 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
model1.fit(X_train, y_train)

y_prediksi = model1.predict(X_test)
print(y_prediksi)
print(met.accuracy_score(y_test, y_prediksi))

labels = ['mesin', 'bangku', 'penggerak']
dot_data = tree.export_graphviz(model1, out_file=None,
                                feature_names=labels, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_png('decisiontree.png')



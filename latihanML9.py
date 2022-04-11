import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pyspark as spark

data = pd.read_csv('dataset/temps2.csv', sep=';')
creation = data.head(5)
print("The shape of our feature is:", creation.shape)
creation.describe()
creation = pd.get_dummies(creation)
creation.iloc[:,5:].head(5)
labels = np.array(creation['Temperature'])
creation = creation.drop('Temperature',axis=1)
creation_list = list(creation.columns)
creation = np.array(creation)

train_creation, test_creation, train_labels, test_labels= train_test_split(creation,labels, test_size=0.30,random_state=4)

print('Training creation shape:', train_creation.shape)
print('Training labels shape:', train_labels.shape)
print('Testing creation shape:', test_creation.shape)
print('Testing label shape:', test_labels.shape)
rf=RandomForestRegressor(n_estimators=1000, random_state=4)
rf.fit(train_creation, train_labels)
predictions=rf.predict(test_creation)
print(predictions)
errors=abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape=100* (errors/test_labels)
accuracy=100-np.mean(mape/3)
print('Accuracy of the model:', round(accuracy,2),'%')


# Every record contains a label and feature vector
df = spark.sql.SparkSession.createDataFrame(data, [labels,creation])

# Split the data into train/test datasets
train_df, test_df = df.randomSplit([.80, .20], seed=42)

# Set hyperparameters for the algorithm
rf = RandomForestRegressor(numTrees=100)

# Fit the model to the training data
model = rf.fit(train_creation, train_labels)

# Generate predictions on the test dataset.
# model.tran(test_creation, test_labels).show()

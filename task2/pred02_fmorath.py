import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFwe, f_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# read in data from csv files
df_train = pd.read_csv('train.csv', float_precision = 'round_trip')
df_test = pd.read_csv('test.csv', float_precision = 'round_trip')

# prepare feature matrix X and response vector y
feature_cols = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15']
y_train = df_train['y']
X_train = df_train[feature_cols]
X_test = df_test[feature_cols]

# build the classifier
exported_pipeline = make_pipeline(
    SelectKBest(f_classif, k = 11),
    GradientBoostingClassifier(max_depth=9, max_features=0.55,
    min_samples_leaf=9, min_samples_split=4, subsample=0.9000000000000001, random_state = 2)
)


# train the classifier
exported_pipeline.fit(X_train, y_train)

# print out cross validation mean score for the chosen model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(exported_pipeline, X_train, y_train, cv = 10, scoring = 'accuracy')
print('cross val mean score = ', scores.mean())

# print accuracy
from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(y_train, exported_pipeline.predict(X_train))
print('Accuracy on training data = ', accuracy_train)

# predict values on test data
y_test = exported_pipeline.predict(X_test)

# write predictions to csv file
id = []
for i in range(1000,4000):
    id.append(i)

df_pred = pd.DataFrame(columns=['Id', 'y'])
df_pred['y'] = y_test
df_pred['Id'] = id
df_pred.to_csv('pred02_fmorath.csv', index = False)


import pandas as pd
import numpy as np
import csv

# read in data from csv files
df_train = pd.read_hdf('train.h5')
df_test = pd.read_hdf('test.h5')

# prepare feature matrix X and response vector y
X_train = df_train.ix[:,1:101].as_matrix()
y_train = df_train.ix[:, 0:1].as_matrix()


# build the NN
from sknn.mlp import Classifier, Layer

nn = Classifier(
	layers=[Layer('Tanh', units=100), Layer('Sigmoid', units = 25), Layer('Softmax', units=5)],
	learning_rate=.03,
	n_iter=73,
	batch_size = 10
)

nn.fit(X_train, y_train)

 # print accuracy
from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(y_train, nn.predict(X_train))
print('Accuracy on training data = ', accuracy_train)

# predict values on test data
prediction = nn.predict(df_test.as_matrix())

# write predictions to csv file
outfile = open('pred01_fmorath.csv','w') # change the file name
writer = csv.writer(outfile)
writer.writerow(['Id', 'y'])
ids = df_test.ix[:, 0:1]
for i in range(prediction.shape[0]):
 	writer.writerow([i+45324, prediction[i][0]])
outfile.close()

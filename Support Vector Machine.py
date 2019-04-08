import pandas as pd
import numpy as np

from pandas import DataFrame
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#Read data from Kaggle_data directory
train_samples = pd.read_csv('Kaggle_data\\train_samples.csv', dtype = 'double', header = None) 
train_labels = pd.read_csv('Kaggle_data\\train_labels.csv', dtype = 'double', header = None) 
test_samples = pd.read_csv('Kaggle_data\\test_samples.csv', dtype = 'double', header = None) 

#Define regularization parameter, gamma and kernel
classifier = svm.SVC(C = 1.0, gamma = 'scale', kernel = 'rbf')
#Training data usin train_samples and train_labels
classifier.fit(train_samples, train_labels.values.ravel())

#predict labels for test_samples
prediction = classifier.predict(scaled_test_samples)

#export predicted labels to a csv file
col = {'Prediction':prediction} #create column prediction for csv file
exp = pd.DataFrame(col)
exp.index = exp.index + 1
exp.to_csv('prediction.csv')
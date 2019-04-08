import pandas as pd
import numpy as np

from pandas import DataFrame
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from normalizer import *

#Read data
train_samples = pd.read_csv('Kaggle_data\\train_samples.csv', dtype = 'double', header = None) 
train_labels = pd.read_csv('Kaggle_data\\train_labels.csv', dtype = 'double', header = None) 
test_samples = pd.read_csv('Kaggle_data\\test_samples.csv', dtype = 'double', header = None) 


#Create 3 folds by splitting train data in 3 equal arrays
labels1 = train_labels.iloc[0:5000,]
samples1 = train_samples.iloc[0:5000,:]

labels2 = train_labels.iloc[5000:10000,]
samples2 = train_samples.iloc[5000:10000,:]

labels3 = train_labels.iloc[10000:15000,]
samples3 = train_samples.iloc[10000:15000,:]


#Concatenate each 2 folds so we could train them
train12_samples = np.concatenate((samples1, samples2))
train12_labels = np.concatenate((labels1, labels2))

train13_samples = np.concatenate((samples1, samples3))
train13_labels = np.concatenate((labels1, labels3))

train23_samples = np.concatenate((samples2, samples3))
train23_labels = np.concatenate((labels2, labels3))

#Define 3 classifiers and train them using 2 folds for each of them
classifier1 = svm.SVC(C = 1.0, gamma = 'scale', kernel = 'rbf')
classifier1.fit(train23_samples, train23_labels.ravel())

classifier2 = svm.SVC(C = 1.0, gamma = 'scale', kernel = 'rbf')
classifier2.fit(train13_samples, train13_labels.ravel())

classifier3 = svm.SVC(C = 1.0, gamma = 'scale', kernel = 'rbf')
classifier3.fit(train12_samples, train12_labels.ravel())


#Predict data for the one fold remained
pred3 = classifier3.predict(samples3)
pred2 = classifier2.predict(samples2)
pred1 = classifier1.predict(samples1)
#Print accuracy scores
print("Accuracy score(12 => 3) : ", accuracy_score(labels3, pred3))
print("Accuracy score(13 => 2) : ", accuracy_score(labels2, pred2))
print("Accuracy score(23 => 1) : ", accuracy_score(labels1, pred1))

#Calculate confusion matrix
conf_matrix1 = confusion_matrix(labels1, pred1)
conf_matrix2 = confusion_matrix(labels2, pred2)
conf_matrix3 = confusion_matrix(labels3, pred3)
#Print confusion matrix
print("Matricea de confuzie 1: ", conf_matrix1)
print("Matricea de confuzie 2: ", conf_matrix2)
print("Matricea de confuzie 3: ", conf_matrix3)



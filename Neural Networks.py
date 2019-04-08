import pandas as pd
import numpy as np

from pandas import DataFrame
from sklearn.neural_network import MLPClassifier

#Define perceptron neural network
mlp_classifier_model = MLPClassifier(hidden_layer_sizes=(600, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', 
learning_rate = 'constant', learning_rate_init = 0.001, power_t = 0.5, max_iter = 150, shuffle = True, random_state = None, tol = 0.0001, momentum = 0.9, 
early_stopping=False, validation_fraction=0.1,n_iter_no_change=10)

#Read data from Kaggle_data directory
train_samples = pd.read_csv('Kaggle_data\\train_samples.csv', dtype = 'double', header = None) 
train_labels = pd.read_csv('Kaggle_data\\train_labels.csv', dtype = 'double', header = None) 
test_samples = pd.read_csv('Kaggle_data\\test_samples.csv', dtype = 'double', header = None) 

#Train data using train_samples and train_labels
mlp_classifier_model.fit(train_samples, train_labels)

#Predict labels for test_samples
prediction = mlp_classifier_model.predict(test_samples)

#Export predicted labels to csv file
col = {'Prediction':prediction} #Create prediction column
export = pd.DataFrame(col)
export.index = export.index + 1
export.to_csv('prediction.csv')
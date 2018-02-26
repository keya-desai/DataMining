#!/usr/bin/env python3
# -*- coding: utf-8 -*-





import csv
import numpy as np
import matplotlib.pyplot as plt
from math import exp, expm1,log
result = []
data = []
fields = []
with open ('iris.csv','rt') as csvfile:
	reader = csv.reader(csvfile)
	fields = next(reader)
	for row in reader:
		data.append(row)
X = []
Y = []
for j in range(len(data)):
    X.append([])
    Y.append([])
    X[j].append(1)
    Y[j].append(1)
    for k in range(4):
        X[j].append(float(data[j][k]))
    Y[j][0]=data[j][4]

#Normalization 


curr_min=[]

curr_max=[]

data_tran=np.transpose(X)
for i in range (1,5):
    #curr_mean=np.mean(data_tran[i])
    cur_min=min(data_tran[i])
    cur_max=max(data_tran[i])
    curr_min.append(cur_min)
    curr_max.append(cur_max)
    for val in range (len(data)):
        data_tran[i][val]= ( data_tran[i][val] - cur_min ) / (cur_max-cur_min)
X=np.transpose(data_tran)
#Initialising weights or thetas or b in equation 
w=[]
w=np.random.rand(1,9)*0
curr_min=np.array(curr_min)
curr_max=np.array(curr_max)
epoch=30000
alpha=0.1
#INITIALIZATION 
cost_history = [0] * epoch 
y=np.array(Y)
m = len(data)
#Spilting the dataset into training and testing 
from random import shuffle
n = m #number of rows in your dataset
indices = list(range(m))
shuffle(indices)
y=np.array(Y)
print(indices[:int(m/2)])

print(int(m/2) + 1) 
w=np.array(w)
#end of spliting 
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility

from numpy import array
from numpy import argmax
from keras.utils import to_categorical
# define example
import numpy
import pandas
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
# create model
xtrain= X[indices[0:int(m/2)]]
ytrain= dummy_y[indices[0:int(m/2)]]
xtest=X[indices[int(m/2)+1:m]]
ytest=dummy_y[indices[int(m/2)+1:m]]
model = Sequential()
model.add(Dense(10, input_dim=5, activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(3, activation='sigmoid'))

#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
#model.compile(loss='binary_crossentropy',
#              optimizer=sgd,
#              metrics=['accuracy'])
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# Fit the model

model.fit(xtrain, ytrain, epochs=100, batch_size=10)

scores = model.evaluate(xtest, ytest)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

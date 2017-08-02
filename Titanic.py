#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 21:19:49 2017
https://www.kaggle.com/c/titanic/data
@author: johnnyhsieh
"""
import pandas as pd
import numpy as np

dataset = pd.read_csv('train.csv',engine = 'python')
test_data = pd.read_csv('test.csv',engine = 'python')
totall_data = pd.concat([dataset, test_data])
#dataset = dataset.dropna(subset = ['Embarked'])
totall_data['Embarked'].value_counts()
#dataset['Family'] = dataset['SibSp']+dataset['Parch']
y_train = dataset.values[:,1]
x_train = dataset.drop(labels = ['PassengerId','Survived','Name','Ticket','Cabin','SibSp','Parch'],axis = 1)
fill_value = {'Age':int(totall_data['Age'].median()),
              "Fare" : int(totall_data['Fare'].median())
              , "Embarked" : "S"}
#.fillna() replace nan with 
x_train = x_train.fillna(fill_value)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
lab_encoder = LabelEncoder()

x_train = x_train.values
x_train[:,1] = lab_encoder.fit_transform(x_train[:,1])
x_train[:,4] = lab_encoder.fit_transform(x_train[:,4])
#x_train[:,4] = lab_encoder.fit_transform(x_train[:,4])
oh_encoder = OneHotEncoder(categorical_features = [4])
x_train = oh_encoder.fit_transform(x_train).toarray()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train
                                                    ,y_train,test_size = 0.1
                                                    , random_state = 42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import EarlyStopping,TensorBoard
from sklearn.utils import class_weight

Classifier = Sequential()
Classifier.add(Dense(units = 256, activation = 'relu'
                     ,kernel_initializer="uniform", input_dim = 7))
Classifier.add(Dropout(0.6))
Classifier.add(Dense(units = 128, activation = 'relu'
                     ,kernel_initializer="uniform"))
Classifier.add(Dropout(0.4))
Classifier.add(Dense(units = 64, activation = 'relu'
                     ,kernel_initializer="uniform"))
Classifier.add(Dropout(0.2))
Classifier.add(Dense(units = 128, activation = 'relu'
                     ,kernel_initializer="uniform"))
Classifier.add(Dense(units = 1, activation = 'sigmoid'
                     ,kernel_initializer="uniform"))
Classifier.compile(optimizer='adam',loss = 'binary_crossentropy'
                   ,metrics = ['accuracy'])
#early_stop = EarlyStopping(monitor='val_loss', patience=10
                           #, verbose=0, mode='auto')
class_weight = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
Classifier.fit(x_train,y_train,batch_size = 42,epochs = 200
               ,validation_data= (x_test,y_test)
               ,class_weight = class_weight
               ,shuffle=True)

predict = Classifier.predict(x_train)
predict = (predict>0.5)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
cm = confusion_matrix(np.float32(y_train), predict)
plt.figure(figsize = (2,2))
sn.heatmap(cm, annot=True)

plt.show()



#submit to kaggle
reslut = pd.read_csv('test.csv',engine='python')
reslut = reslut.fillna(fill_value)
#reslut['Family'] = reslut['SibSp']+ reslut['Parch']
reslut = reslut.drop(labels = ['PassengerId','Name','Ticket','Cabin','SibSp','Parch'],axis = 1)
reslut = reslut.values
reslut[:,1] = lab_encoder.fit_transform(reslut[:,1])
reslut[:,4] = lab_encoder.fit_transform(reslut[:,4])
#reslut[:,4] = lab_encoder.fit_transform(reslut[:,4])
oh_encoder = OneHotEncoder(categorical_features = [4])
reslut = oh_encoder.fit_transform(reslut).toarray()
reslut = sc.fit_transform(reslut)
result = Classifier.predict(reslut)
result = (result>0.5)
result = lab_encoder.fit_transform(result)
result = pd.DataFrame(result)
test_data = pd.read_csv('test.csv',engine='python') 
results = pd.DataFrame({
    'PassengerId' : test_data['PassengerId'],
    'Survived' : result.values[:,0]
})

results.to_csv("reslut.csv",index = False)
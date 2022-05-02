# 1.0 Import Libraries
import pandas as pd
import os
import matplotlib.pyplot as py
from sklearn.model_selection import train_test_split

#1.1 Read data
carddataset=pd.read_csv("E:\\Self_Study\\DataScience\\Projects\\DataFiles\\creditcard.zip")#1.2 Get an insight into the dataset.carddataset.info()
carddataset.head()
carddataset.describe()
carddataset.columns
carddataset['Class'].value_counts()
X = carddataset.drop('Class',axis=1)
y = carddataset['Class']
X.columns
y.value_counts()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42,
                                                    shuffle= True,stratify=y)
X_train.shape
y_train.shape
X_test.shape
y_test.shape
y_train.value_counts()
y_test.value_counts()
X_train.nunique(axis=0)

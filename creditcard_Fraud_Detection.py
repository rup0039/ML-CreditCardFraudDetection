# 1.0 Import Libraries

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1.1 Read data

carddataset=pd.read_csv("E:\\Self_Study\\DataScience\\Projects\\DataFiles\\creditcard.zip")

#1.2 Get an insight into the dataset

carddataset.info()
carddataset.head()
carddataset.describe()
carddataset.columns
carddataset['Class'].value_counts()

# 1.3 Seperate training and testing dataset

X = carddataset
y = carddataset['Class']
y.value_counts()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42,
                                                    shuffle= True,stratify=y)
X_train.shape
y_train.shape
X_test.shape
y_test.shape
X_train['Class'].value_counts()
X_train.columns

# 1.4 Data Visualisation

%matplotlib qt5

pd.plotting.radviz(X_train,'Class',colormap=plt.cm.winter)

# Radviz plt indicate parameters seperate fraud transactions from normal transactions

plt.imshow(X_train.corr(),cmap = 'gray')
plt.colorbar()


plt.hist(X_train[X_train['Class']==1]['Amount'])
plt.hist(X_train[X_train['Class']==0]['Amount'])

# Fraud transactions amount are smaller in comparison to normal transactions, it may be due to 
# amount greater than a threshold involves more layer of security
fig, ax = plt.subplots(2,1,figsize=(6,6))
ax[0].scatter(x='Time', y='Amount', data = X_train[X_train['Class']==1],label = 'Fraud')
ax[1].scatter(x='Time', y='Amount', data = X_train[X_train['Class']==0],label = 'Normal')
plt.legend()

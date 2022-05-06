# 1.0 Import Libraries

import pandas as pd
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

plt.scatter('V2','V3', c= y_train, data = X_train)

columns = X_train.columns
fig, axs = plt.subplots(10, 3,figsize=(30,40))
for i, ax in enumerate(axs.flat):
    ax.boxplot([X_train[X_train['Class']==1][columns[i]],X_train[X_train['Class']==0][columns[i]]],labels=[1,0])

plt.imshow(X_train.corr(),cmap = 'gray')
plt.colorbar()
# Plots to check pattern in data just for academic purposes




pd.plotting.radviz(X_train,'Class',colormap=plt.cm.winter)
pd.plotting.parallel_coordinates(X_train,'Class',colormap=plt.cm.winter)
pd.plotting.andrews_curves(X_train,'Class',colormap=plt.cm.winter)

#Plotting##############################################################################################
plt.hist(X_train[X_train['Class']==1]['Amount'])
plt.hist(X_train[X_train['Class']==0]['Amount'])
plt.hist([X_train[X_train['Class']==1]['Amount'],X_train[X_train['Class']==0]['Amount']])
################################################################################################

fig,ax1 = plt.subplots()
ax1.set_ylabel("Fraud Amount")
ax1.set_xlabel("Time")
#ax1.legend("Fraud")
ax1.scatter('Time','Amount',c="red",label="Fraud",data=X_train[X_train['Class']==1])
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.set_ylabel("Non Fraud Amount")
ax2.scatter('Time','Amount',c="blue",label="Non-Fraud",data=X_train[X_train['Class']==0])
ax2.legend(loc="upper right")
#plt.legend()

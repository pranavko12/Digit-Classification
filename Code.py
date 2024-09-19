#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import pickle 
import gzip
import random


# In[6]:


digits = pd.read_csv("D:\\NMIMS\\Internship - Verzeo\\Major Project\\digit_svm.csv")
digits.info()


# In[7]:


digits.head()


# In[10]:


four = digits.iloc[3,1:]
four.shape


# In[11]:


four = four.values.reshape(28, 28)
plt.imshow(four, cmap='gray')


# In[12]:


#Visualize the array
print(four[5:-5, 5:-5])


# In[13]:


#Summarise the counts of 'label' to see how many labels of each digit are present
digits.label.value_counts()


# In[14]:


# Summarise count in terms of percentage 
100*(round(digits.label.astype('category').value_counts()/len(digits.index), 4))


# In[15]:


#Check for missing value
digits.isnull().sum()


# In[16]:


#average values/distributions of features
description = digits.describe()
description


# In[74]:


# Creating training and test sets
# Splitting the data into train and test
X = digits.iloc[:, 1:]
Y = digits.iloc[:, 0]

# Rescaling the features
from sklearn.preprocessing import scale
X = scale(X)

# train test split with train_size=10% and test size=90%
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.1, random_state=101)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[75]:


#Linear SVM
from sklearn import svm
from sklearn import metrics

# an initial SVM model with linear kernel   
svm_linear = svm.SVC(kernel='linear')

# fit
svm_linear.fit(x_train, y_train)


# In[78]:


# predict
predictions = svm_linear.predict(x_test)
predictions


# In[79]:


confusion = metrics.confusion_matrix(y_true = y_test, y_pred = predictions)
confusion


# In[81]:


# measure accuracy
metrics.accuracy_score(y_true=y_test, y_pred=predictions)


# In[82]:


# class-wise accuracy
class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions)
print(class_wise)


# In[83]:


gc.collect()


# In[84]:


# rbf kernel with other hyperparameters kept to default 
svm_rbf = svm.SVC(kernel='rbf')
svm_rbf.fit(x_train, y_train)


# In[85]:


# predict
predictions = svm_rbf.predict(x_test)

# accuracy 
print(metrics.accuracy_score(y_true=y_test, y_pred=predictions))


# In[91]:


# conduct (grid search) cross-validation to find the optimal values 
# of cost C and the choice of kernel

from sklearn.model_selection import GridSearchCV

parameters = {'C':[1, 10, 100], 
             'gamma': [1e-2, 1e-3, 1e-4]}

# instantiate a model 
svc_grid_search = svm.SVC(kernel="rbf")

# create a classifier to perform grid search
clf = GridSearchCV(svc_grid_search, param_grid=parameters, scoring='accuracy')

# fit
clf.fit(x_train, y_train)


# In[92]:


# results
cv_results = pd.DataFrame(clf.cv_results_)
cv_results


# In[93]:


# optimal hyperparameters
best_C = 1
best_gamma = 0.001

# model
svm_final = svm.SVC(kernel='rbf', C=best_C, gamma=best_gamma)

# fit
svm_final.fit(x_train, y_train)


# In[94]:


# predict
predictions = svm_final.predict(x_test)


# In[95]:


# evaluation: CM 
confusion = metrics.confusion_matrix(y_true = y_test, y_pred = predictions)

# measure accuracy
test_accuracy = metrics.accuracy_score(y_true=y_test, y_pred=predictions)

print(test_accuracy, "\n")
print(confusion)


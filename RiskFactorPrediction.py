
# coding: utf-8

# In[ ]:


# Install all the dependencies through requirements.txt 


# In[ ]:


# !pip install -r requirements.txt


# In[ ]:


# References
# http://scikit-learn.org/stable/modules/classes.html
# http://scikit-learn.org/stable/modules/linear_model.html#bayesian-ridge-regression


# In[ ]:


# All Imports


# In[192]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib as mpl
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from sklearn import model_selection
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression



# In[193]:


### Read from SuperFan CSV Example dataset


# In[194]:


columns = ['RiskFactor','Loss','Make','FT','Asprtn','Door','BS','Drive','EngineLoc',
           'WB','Len','Wdth','Hght','Curbwt','EngineType','Cyl','EngineSize',
           'FS','Bore','Stroke','CmprRatio','HP','PeakRpm','Citympg','Hwympg','Price']

# dataset = pd.read_csv('Auto_data.csv', names = columns)
dataset = pd.read_csv('Auto_data.csv')


# summarize the number of rows and columns in the dataset
print('Before removing missing values:', dataset.shape)

dataset = dataset.replace('?', np.NaN)

# drop rows with missing values
dataset.dropna(inplace=True)

print(dataset)

# summarize the number of rows and columns in the dataset
print('After removing missing values:', dataset.shape)

print(dataset.describe())


# In[195]:


# Create a label (category) encoder object

dataset = dataset.apply(preprocessing.LabelEncoder().fit_transform)


# In[196]:


### Visualization of the dataset


# In[172]:


dataset.plot(kind='line', subplots=True, layout=(1, 26), sharex=False, sharey=False, figsize=(18, 8))
plt.show()


# In[197]:


columns = ['Loss','Make','FT','Asprtn','Door','BS','Drive','EngineLoc',
           'WB','Len','Wdth','Hght','Curbwt','EngineType','Cyl','EngineSize',
           'FS','Bore','Stroke','CmprRatio','HP','PeakRpm','Citympg','Hwympg','Price']

# If you also want to see RiskFactor column in matrix uncomment this line
pd.plotting.scatter_matrix(dataset[columns], c=dataset["RiskFactor"], figsize=(10, 10), marker='.', cmap='brg',
                                   hist_kwds={'bins': 20}, s=60, alpha=.8)

labels=["Normal Fan", "Missing Data","RiskFactor"]
plt.legend(handles, labels, loc=(1.03,0))

plt.show()


# In[198]:


# Dataset

numpy_dataset = dataset.values


# In[199]:


X = numpy_dataset[:,1:26]
Y = numpy_dataset[:,0]

print(X)
print(Y)


# In[206]:


# Data Split
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size= 0.30)


# In[207]:


## Prediction logic
def predict(X_validation, Y_validation, algorithm, model):
    # Prediction Report
    predictions = model.predict(X_validation)
    # print('y_validation', Y_validation)
    # print('predictions', predictions)
    #print('Accuracy score:', algorithm, accuracy_score(Y_validation, predictions))
    # print('Confusion Matrix', confusion_matrix(Y_validation, predictions))

    # Explained variance score: 1.0 is perfect prediction
    print('Variance score: %.2f' %r2_score(Y_validation, predictions))
    print('Explained Variance score: %.2f' %explained_variance_score(Y_validation, predictions))
    print('Mean Absolute Error: %.2f' %mean_absolute_error(Y_validation, predictions))
    print('Mean Squared Error: %.2f' %mean_squared_error(Y_validation, predictions))
    print('Mean Squared_log Error: %.2f' %mean_squared_log_error(Y_validation, predictions))
    print('Median Absolute Error: %.2f' %median_absolute_error(Y_validation, predictions))
    
    r2_score


# In[208]:


# Cross Validation Result
def build_cross_validation_result(X_train, Y_train, algo, model):
    kfold = model_selection.KFold(n_splits = 10, random_state = 7)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring = 'accuracy')
    # print('cv_results:', cv_results)
    result = "Cross Verification Result -> %s:  %0.4f (+/- %0.4f)" % (algo, cv_results.mean(), cv_results.std()*2)
    print(result)


# In[209]:


## Decision Tree: CART
decisionTreeModel = DecisionTreeRegressor()

build_cross_validation_result(X_train, Y_train, 'Decision Tree', decisionTreeModel)

# Model built
decisionTreeModel.fit(X_train, Y_train)

# Predict
predict(X_validation, Y_validation, 'Decision Tree', decisionTreeModel)
    


# In[210]:


## LinearRegression
linearRegressionModel = LinearRegression()

# Model built
linearRegressionModel.fit(X_train, Y_train)

# Predict
predict(X_validation, Y_validation, 'Linear Regression Model', linearRegressionModel)


# In[211]:


## Bayesian Ridge Regression
bayesianRidgeModel = linear_model.BayesianRidge()

# Model built
bayesianRidgeModel.fit(X_train, Y_train)

# Predict
predict(X_validation, Y_validation, 'Bayesian Ridge Regression', bayesianRidgeModel)


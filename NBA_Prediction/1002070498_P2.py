#!/usr/bin/env python
# coding: utf-8

# In[123]:


# Dedeepya Nallamothu - 1002070498

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

# loading from the dataset
data = pd.read_csv("nba2021.csv")

# selecting specific features for analysis based on Domain Knowledge
selected_features = ['GS','MP','FG%','FG','3PA','2P','2P%', 'FT%','eFG%', 'ORB','TRB','DRB','AST', 'STL', 'BLK', 'PTS','PF']

# Extracting features and target variable from the dataset
X = data[selected_features]
y = data['Pos']

# Splitting the dataset into training and testing sets with a 75-25 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Setting up the parameter grid for hyperparameter tuning
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

# Performing hyperparameter tuning using GridSearchCV with 10-fold cross-validation
grid_search = GridSearchCV(LinearSVC(dual=False, random_state=0), param_grid, cv=10)
grid_search.fit(X_train, y_train)

# Training a new LinearSVC model with the best parameters obtained from the hyperparameter tuning
best_linearsvm = LinearSVC(dual=False, C=grid_search.best_params_['C'], random_state=0).fit(X_train, y_train)

# Printing the test and train set scores with the best parameters
print("Train set score with best parameters: {:.3f}".format(best_linearsvm.score(X_train, y_train)))
print("Test set score with best parameters: {:.3f}".format(best_linearsvm.score(X_test, y_test)))

# Making predictions on the test set and computing the confusion matrix
y_pred = best_linearsvm.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
confusion_matrix_df = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(confusion_matrix_df)

# Performing 10-fold cross-validation and calculating the average cross-validated score
cv_scores = cross_val_score(best_linearsvm, X_train, y_train, cv=10)
print("Cross-validated scores: ", cv_scores)
avg_score = np.mean(cv_scores)
print("Average cross-validated score: {:.3f}".format(avg_score))


# In[ ]:





# In[ ]:





# In[ ]:





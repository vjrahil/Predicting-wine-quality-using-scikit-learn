# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#preprocessing phase
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#exctracting dataset

dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:,0:13].values
y = dataset.iloc[:,13].values

#splitting dataset

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25)

#feature scaling

from sklearn.preprocessing import StandardScaler
standard_x = StandardScaler()
X_train = standard_x.fit_transform(X_train)
X_test = standard_x.transform(X_test)

#dimensionality reduction

from sklearn.decomposition import PCA
pca = PCA(n_components =2 )
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
#explained_variance = pca.explained_variance_ratio_ 

#classifier

from sklearn.svm import SVC
classifier = SVC(kernel ='linear',random_state =0)
classifier.fit(X_train,y_train)

#prediction

y_pred= classifier.predict(X_test)

#confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#k-foldcross validation

from sklearn.model_selection import cross_val_score
accuray = cross_val_score(estimator = classifier,X = X_train,y = y_train,cv = 10)
mean =accuray.mean()
dev = accuray.std()

#grid search

from sklearn.model_selection import GridSearchCV
para = [{'C' : [1,2,3,4,5,6,7,8],'kernel':['linear']},
         {'C' : [1,10,100,1000],'kernel':['rbf'],'gamma':[.1,.01,.001,.0001]}]

grid_search = GridSearchCV(estimator = classifier,param_grid = para,scoring= 'accuracy',cv =10,n_jobs =-1)
grid_search = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#visualising the train set results

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
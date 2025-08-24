from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score,RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from scipy.stats import randint

#=====Load Data=========
iris=load_iris()
X,y=iris.data , iris.target
#======================

#====Model Training======
dt=DecisionTreeClassifier(random_state=42)
dt_score=cross_val_score(dt,X,y,cv=5)
#========================

#====Prediction===========
print("Decision Tree CV Accuracy:", dt_score)
print("Mean Accuracy:", np.mean(dt_score))
#=========================

#====HyperParameters=====
dt_params={
    "max_depth":[None,5,10,20],
    "min_samples_split":randint(2,11),
    "min_samples_leaf":randint(1,5),
    "criterion":['gini','entropy']
}

dt_grid=RandomizedSearchCV(dt,dt_params,cv=5,scoring='accuracy',n_iter=10,random_state=42)
dt_grid.fit(X,y)
print("Best Hyperparameters:", dt_grid.best_params_)
print("Best Cross-Validation Score:", dt_grid.best_score_)
#=========================
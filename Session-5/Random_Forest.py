from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from scipy.stats import randint
#=====Load Data=========
iris=load_iris()
X,y=iris.data , iris.target
#======================

#====Model Training======
rfc=RandomForestClassifier(random_state=42)
rfc_score=cross_val_score(rfc,X,y,cv=5)
#========================

#====Prediction===========
print("Random Forest Classifier CV Accuracy:", rfc_score)
print("Mean Accuracy:", np.mean(rfc_score))
#=========================

#====HyperParameters=====
rfc_params={
    "n_estimators":randint(50,200),
    "max_depth":[None,5,10,20],
    "min_samples_split":randint(2,11),
    "min_samples_leaf":randint(1,5),
    "criterion":['gini','entropy']
}

rfc_grid=RandomizedSearchCV(rfc,rfc_params,cv=5,scoring='accuracy',n_iter=10,random_state=42)
rfc_grid.fit(X,y)
print("Best Hyperparameters:", rfc_grid.best_params_)
print("Best Cross-Validation Score:", rfc_grid.best_score_)
#=========================
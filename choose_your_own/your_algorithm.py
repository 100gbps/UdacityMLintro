#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
##plt.xlim(0.0, 1.0)
##plt.ylim(0.0, 1.0)
##plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
##plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
##plt.legend()
##plt.xlabel("bumpiness")
##plt.ylabel("grade")
##plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

import numpy as np
from time import time
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


best_accuracy = 0
Args = {}


############################### Gaussian Naive Bayes ###########################
                        
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)

if (accuracy > best_accuracy):
    best_accuracy = accuracy
    best_clf = clf
    algorithm = "Gaussian Naive Bayes"
    Args.clear()

############################## Support Vector Machine ##########################
for kern in ("linear", "poly", "rbf", "sigmoid"):
    for c in range(1,10000,500):
        for shrink in (True, False):
            clf = svm.SVC(kernel=kern, C=c, shrinking=shrink)
            clf.fit(features_train, labels_train) 
            pred = clf.predict(features_test)
            accuracy = accuracy_score(labels_test, pred)

            if (accuracy > best_accuracy):
                best_accuracy = accuracy
                best_clf = clf
                algorithm = "SVM"
                Args.clear()
                Args = {"kernel":kern, "C":c, "shrinking":shrink}

############################## Support Vector Machine ##########################

for n in range(1, 20):
    for crit in ("entropy", "gini"):
        for feat in ("sqrt", "log2", None):
            clf = RandomForestClassifier(n_estimators=n, max_features=feat, criterion=crit)
            clf = clf.fit(features_train, labels_train)
            pred = clf.predict(features_test)
            accuracy = accuracy_score(labels_test, pred)
            
            if (accuracy > best_accuracy):
                best_accuracy = accuracy
                best_clf = clf
                algorithm = "Random forest"
                Args.clear()
                Args = {"n_estimators":n,"max_features":feat, "criterion":crit}

############################### K Nearest Neighbor #############################

for alg in ("auto", "ball_tree", "kd_tree", "brute"):
    for n in range(1, 15):
        for w in ("uniform", "distance"):
            for p in range(1, 2):
                for size in range(1, 50):
                    clf = KNeighborsClassifier(leaf_size=size, algorithm=alg, \
                                               p=p, weights=w, n_neighbors=n)
                    clf.fit(features_train, labels_train)
                    pred = clf.predict(features_test)
                    accuracy = accuracy_score(labels_test, pred)
                    
                    if (accuracy > best_accuracy):
                        best_accuracy = accuracy
                        best_clf = clf
                        algorithm = "K Nearest Neighbor"
                        Args = {"best_alg":alg, "best_n":n, "best_w":w, \
                                "best_p":p, "best_size":size}

print "Accuracy:", best_accuracy
print "Algorithm:", algorithm
for k, v in Args.iteritems():
    print k, v


try:
    prettyPicture(best_clf, features_test, labels_test)
except NameError:
    pass

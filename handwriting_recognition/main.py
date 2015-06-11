#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  main.py
#  
#  Copyright 2015 Eusebio Aguilera <eusebio.aguilera@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#

__author__ = "Eusebio J. Aguilera Aguilera"
__copyright__ = "Eusebio J. Aguilera Aguilera"
__credits__ = "Eusebio J. Aguilera Aguilera"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Eusebio J. Aguilera Aguilera"


import os
from os.path import join
import time
from skimage import io
from skimage.feature import local_binary_pattern # Local binary pattern
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm, metrics, cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from prettytable import PrettyTable

# This delta values are used to extract the images inside the big image

TRAIN_DELTA = 28
TEST_DELTA = 28
LBP_BINS = 59   # LBPU

radius = 3
n_points = 8 * radius



"""
    This function is used to extract the little examples in the big image
    The param is a numpy array
"""


def extract_images(img):
    imgs = []
    x, y = img.shape

    for i in range(0, x, TRAIN_DELTA):
        for j in range(0, y, TRAIN_DELTA):
            if np.sum(img[i:i+27, j:j+27]) > 0:
                imgs.append(img[i:i+27, j:j+27])
                #print i, j

    #print x, y

    return imgs

"""
    This method extract the features used for classification
"""


def extract_features(img):
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    lbp_hist, bin_edges = np.histogram(lbp, LBP_BINS)

    # Histogram normalization    
    lbp_hist_norm = sum(abs(lbp_hist))
    lbp_hist_l1sqrtnorm = np.sqrt(lbp_hist/float(lbp_hist_norm))

    return lbp_hist_l1sqrtnorm

"""
    This function is used to train a model using a classifier
"""


def test_classifiers(train_path, test_path):

    myfeat_train = list()
    mylabel_train = list()

    myfeat_test = list()
    mylabel_test = list()

    for i in range(10):
        fname = join(train_path, '%s.jpg' % str(i))
        img = io.imread(fname)
        # Extract images
        imgs = extract_images(img)

        # Extract features
        for x in imgs:
            lbp = extract_features(x)
            myfeat_train.append(lbp)
            mylabel_train.append(i)
        
        print "Features obtained for train class", i
    
    for i in range(10):
        fname = join(test_path, '%s.jpg' % str(i))
        img = io.imread(fname)
        # Extract images
        imgs = extract_images(img)

        # Extract features
        for x in imgs:
            lbp = extract_features(x)
            myfeat_test.append(lbp)
            mylabel_test.append(i)
        
        print "Features obtained for test class", i

    # Train
    # Create a classifier: a support vector classifier
    svml = svm.LinearSVC()
    rf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
    gnb = GaussianNB()
    tr = tree.DecisionTreeClassifier()
    knn = KNeighborsClassifier()
    dummy = DummyClassifier()

    #print "Obtaining the cross validation groups ..."
    # Cross validation
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(myfeat, mylabel, test_size=0.3, random_state=0)

    print "Training ..."
    # Train
    # Compute traing time
    ttime = []
    tt = time.time()
    gnb.fit(myfeat_train, mylabel_train)
    ttime.append(time.time()-tt)

    tt = time.time()
    svml.fit(myfeat_train, mylabel_train)
    ttime.append(time.time()-tt)

    tt = time.time()
    rf.fit(myfeat_train, mylabel_train)
    ttime.append(time.time()-tt)

    tt = time.time()
    tr.fit(myfeat_train, mylabel_train)
    ttime.append(time.time()-tt)

    tt = time.time()
    knn.fit(myfeat_train, mylabel_train)
    ttime.append(time.time()-tt)

    tt = time.time()
    dummy.fit(myfeat_train, mylabel_train)
    ttime.append(time.time()-tt)

    print "Classifying ..."
    #expected = y_test
    #predicted = classifier.predict(X_test)

    #print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted)))
    #print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    names = ["Gaussian Naive Bayes", "Random Forest", "SVM", "Decision Tree", "K Nearest Neighbors", "Dummy (Baseline)"]
    colors = ["r", "b", "g", "m", "y", "k"]

    scores = []
    ctime = []
    tt = time.time()
    scores.append(gnb.score(myfeat_test, mylabel_test))
    ctime.append(time.time()-tt)

    tt = time.time()
    scores.append(rf.score(myfeat_test, mylabel_test))
    ctime.append(time.time()-tt)

    tt = time.time()
    scores.append(svml.score(myfeat_test, mylabel_test))
    ctime.append(time.time()-tt)

    tt = time.time()
    scores.append(tr.score(myfeat_test, mylabel_test))
    ctime.append(time.time()-tt)

    tt = time.time()
    scores.append(knn.score(myfeat_test, mylabel_test))
    ctime.append(time.time()-tt)

    tt = time.time()
    scores.append(dummy.score(myfeat_test, mylabel_test))
    ctime.append(time.time()-tt)

    print "Score for Gaussian Naive Bayes classifier", scores[0]
    print "Score for Random Forest classifier", scores[1]
    print "Score for Support Vector Machine (Linear kernel) classifier", scores[2]
    print "Score for Decision Tree classifier", scores[3]
    print "Score for K Nearest Neighbors classifier", scores[4]
    print "Score for Dummy (Baseline) classifier", scores[5]
    
    pt = PrettyTable(["Classifier", "Score", "Training time (s)", "Classifying time (s)", "Total time (s)"])
    
    for i in range(len(names)):
        pt.add_row([names[i], scores[i], ttime[i], ctime[i], ttime[i]+ctime[i]])

    print pt

    for i in range(len(names)):
        plt.bar(i, scores[i], label=names[i], color=colors[i])

    plt.legend()
    plt.show()


def main():
    
    current = os.path.dirname(os.path.abspath(__file__))

    train_path = join(current, 'dataset/train/')
    test_path = join(current, 'dataset/test/')

    test_classifiers(train_path, test_path)

    return 0

if __name__ == '__main__':
    main()

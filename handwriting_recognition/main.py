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
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.dummy import DummyClassifier
from prettytable import PrettyTable

from tools.features import LBPUFeatures, LBPUMultiblockFeatures, HOGFeatures, LBPUMultiBlockAndHOGFeatures

# This delta values are used to extract the images inside the big image

TRAIN_DELTA = 28
TEST_DELTA = 28



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
            #tmp = LBPUFeatures(x)
            #tmp = LBPUMultiblockFeatures(x)
            #tmp = HOGFeatures(x)
            tmp = LBPUMultiBlockAndHOGFeatures(x)
            feat = tmp.getFeatures()
            myfeat_train.append(feat)
            mylabel_train.append(i)
        
        print "Features obtained for train class", i
    
    for i in range(10):
        fname = join(test_path, '%s.jpg' % str(i))
        img = io.imread(fname)
        # Extract images
        imgs = extract_images(img)

        # Extract features
        for x in imgs:
            #lbp = extract_features(x)
            #tmp = LBPUMultiblockFeatures(x)
            #tmp = HOGFeatures(x)
            tmp = LBPUMultiBlockAndHOGFeatures(x)
            feat = tmp.getFeatures()
            myfeat_test.append(feat)
            mylabel_test.append(i)
        
        print "Features obtained for test class", i

    # Train
    # Create a classifier: a support vector classifier
    svml = LinearSVC()
    rf = RandomForestClassifier()
    gnb = GaussianNB()
    tr = tree.DecisionTreeClassifier()
    dummy = DummyClassifier()

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
    dummy.fit(myfeat_train, mylabel_train)
    ttime.append(time.time()-tt)

    print "Classifying ..."

    names = ["Gaussian Naive Bayes", "Random Forest", "SVM", "Decision Tree", "Dummy (Baseline)"]
    colors = ["r", "b", "g", "m", "k"]

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
    scores.append(dummy.score(myfeat_test, mylabel_test))
    ctime.append(time.time()-tt)
    
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

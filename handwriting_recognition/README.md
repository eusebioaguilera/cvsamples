# cvsamples
Handwriting recognition
========================================================

This example contains a small example of the usage of different visual features
for object recognition. We have also used several different classifiers in order
to check what features/classifier perform better.

We have test the following classifiers

* Gaussian Naive Bayes
* Random Forest
* SVM (Linear kernel)
* Decision Tree
* Dummy: Baseline classifiers used as the baseline for comparing with others

We use the database known as "MNIST Handwritten Digits" which is one the common
used databased for handwriting recognition. This database could be downloaded 
from http://cs.nyu.edu/~roweis/data.html.

As the feature vector used for classification the Local Binary Pattern has been 
used (Uniform patterns). The results are good but they could be improved by 
applying the Multi-block LBP. The results also get better if the feature HOG is
used instead. Finally the best results could be obtained by applying a mix of
features LBPUMultiblock+HOG.

Requirements
========================================================

* Numpy >= 1.8.2 http://www.numpy.org/
* Scikit-image >= 0.11.0 http://scikit-image.org/
* Scikit-learn >= 0.16.1 http://scikit-learn.org/
* Matplotlib >= 1.3.1 http://matplotlib.org/
* PrettyTable >= 0.7.2 https://code.google.com/p/prettytable/


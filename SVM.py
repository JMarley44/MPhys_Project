# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 13:50:38 2021

@author: James
"""

'Support vector machine program'
from sklearn import svm
from joblib import dump, load

# SVM Classifiers offer good accuracy and perform faster prediction compared to Naïve Bayes algorithm. 
# They also use less memory because they use a subset of training points in the decision phase. 
# SVM works well with a clear margin of separation and with high dimensional space.

def SVM(X_train, y_train, X_test, C, gamma, tol, tag, ForceModel):
    
    # File for model save/load
    svm_file = 'SVM_models/SVM_'+tag+'.joblib'

    # Define a blank model
    clf = None
    
    # If force model is enabled train the SVM
    if ForceModel:
        clf = svm.SVC(C = C, gamma=gamma, kernel='rbf', probability=True, cache_size=800, 
                      tol = tol, break_ties=True, decision_function_shape = 'ovr') # Cache_size is memory usage
        
        clf.fit(X_train, y_train)
        dump(clf, svm_file)
        print('Trained SVM ' + tag)
        prob_train = clf.predict_proba(X_train)
        prob_test = clf.predict_proba(X_test)
    else:
        try:
            clf = load(svm_file)
            print('Loaded SVM ' + tag)
            prob_train = clf.predict_proba(X_train)
            prob_test = clf.predict_proba(X_test)
        except:
            clf = svm.SVC(C = 0.01, gamma='auto', kernel='rbf', probability=True, cache_size=800, tol = 1E0) # Cache_size is memory usage
            clf.fit(X_train, y_train)
            dump(clf, svm_file)
            print('Trained SVM ' + tag)
            prob_train = clf.predict_proba(X_train)
            prob_test = clf.predict_proba(X_test)
    
    # Return the probability of a background for training and test data
    #bkg_prob_train = prob_train[:,0]
    #bkg_prob_test = prob_test[:,0]
    
    # Return the probability of a signal for training and test data
    sig_prob_train = prob_train[:,1]
    sig_prob_test = prob_test[:,1]

    
    return sig_prob_train, sig_prob_test #, bkg_prob_train, bkg_prob_test



# Used rounding for predictions but will leave this here
'''

for i in range(len(model_1_prob)):
    if model_1_prob[i]<0.5:
        model_1_pred[i] = 0
    else:
        model_1_pred[i] = 1      
'''

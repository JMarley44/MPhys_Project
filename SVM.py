# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 13:50:38 2021

@author: James
"""

'Support vector machine program'
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from joblib import dump, load

#from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc,precision_recall_curve,roc_curve

ForceModel = False

def SVM(data, tag):
    
    # File for model save/load
    svm_file = 'SVM_'+tag+'.joblib'

    # Shuffle the data
    np.random.shuffle(data)
    #!!! Note usage in this way forwards the shuffled data back into the functions 'data' input
    # This is useful for analysis by eye
    
    # Separate into x and y
    y_pos = len(data[0])-1
    X = data[:,0:y_pos]
    y = data[:,y_pos]
    
    # Normalisation of X
    scaler = StandardScaler()
    scaler.fit(X)
    X_norm = scaler.transform(X)
    
    # Training and test splitting
    N = data.shape[0]
    N_train = int(2*N/3)
    
    X_train = X_norm[0:N_train,:]
    y_train = y[0:N_train]
    
    X_test = X_norm[N_train:,:]
    y_test = y[N_train:]
    
    # Define a blank model
    clf = None
    
    # If force model is enabled train the model every time
    if ForceModel:
        print('Force model is enabled')
        clf = svm.SVC(kernel='rbf', probability=True, cache_size=500) # Cache_size is memory usage
        clf.fit(X_train, y_train)   # , sample_weight= np.ascontiguousarray(w_train) 
        dump(clf, svm_file)
        print("Trained the SVM")
        
    # Otherwise try and load the model, and if it doesn't exist train it
    else:
        try:
            clf = load(svm_file)
            print('Loaded the SVM')
        except:
            clf = svm.SVC(kernel='rbf', probability=True, cache_size=500) # Cache_size is memory usage
            clf.fit(X_train, y_train)   # , sample_weight= np.ascontiguousarray(w_train) 
            dump(clf, svm_file)
            print("Trained the SVM")
    
    # Make predictions
    print('Classifier complete')
    prob_train = clf.predict_proba(X_train)
    prob_test = clf.predict_proba(X_test)
    
    # Return the probability of a signal for training and test data
    sig_prob_train = prob_train[:,1]
    sig_prob_test = prob_test[:,1]

    
    return sig_prob_train,sig_prob_test


# Used rounding for predictions but will leave this here
'''

for i in range(len(model_1_prob)):
    if model_1_prob[i]<0.5:
        model_1_pred[i] = 0
    else:
        model_1_pred[i] = 1      
'''

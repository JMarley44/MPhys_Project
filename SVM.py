# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 13:50:38 2021

@author: James
"""

'Support vector machine program'
from sklearn import svm
from joblib import dump, load
import numpy as np

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

def SVM_opt(X_train, y_train, X_test, y_test, w_train, w_test, C, gamma, tol, tag):    

    import Functions as f    

    try:
        C_len = len(C)
    except:
        C_len = 1
        C = np.array([C])
    
    try:
        gamma_len = len(gamma)
    except:
        gamma_len = 1
        gamma = np.array([gamma])
        
    try:
        tol_len = len(tol)
    except:
        tol_len = 1
        tol = np.array([tol])

    all_length = C_len*gamma_len*tol_len
    counter = 0
    
    for i in range(C_len):
        for j in range(gamma_len):
            for k in range(tol_len):
                print('**************')
                print('Opt. run: ' + str(counter+1) + ' / ' + str(all_length))
                print('**************')   
                
                # Define a blank model
                clf = None
            
                # The model
                clf = svm.SVC(C = C[i], gamma=gamma[j], kernel='rbf', probability=True, cache_size=1000, 
                              tol = tol[k], break_ties=True, decision_function_shape = 'ovr') # Cache_size is memory usage
                
                clf.fit(X_train, y_train)
                prob_train = clf.predict_proba(X_train)
                prob_test = clf.predict_proba(X_test)
                
                # Return the probability of a signal for training and test data
                sig_prob_train = prob_train[:,1]
                sig_prob_test = prob_test[:,1]
            
                
                f.ROC_Curve(sig_prob_train, sig_prob_test, y_train, y_test, close=True, 
                            
                            title='SVM_'+ tag +'_C=' + str(C[i]) + '_gamma=' + str(gamma[j]) + '_' +'tol=' + str(tol[k]) + '_', 
                            
                            saveas='SVM/'+ tag +'/Optimisation/C=' + str(C[i]) + '_gamma=' + str(gamma[j]) + '_' +'tol=' + str(tol[k]) + '_')
                
                f.ProbHist(sig_prob_train, sig_prob_test, y_train, y_test, 
                           w_train, w_test, 21, close=True, 
                      label=['ttZ',('ggA_'+ tag)], xtitle='Probability of signal', ytitle='Events', 
                        
                      title='SVM_'+ tag +'_C=' + str(C[i]) + '_gamma=' + str(gamma[j]) + '_' +'tol=' + str(tol[k]) + '_', 
                      
                      saveas='SVM/'+ tag +'/Optimisation/C=' + str(C[i]) + '_gamma=' + str(gamma[j]) + '_' +'tol=' + str(tol[k]) + '_')
    
                counter=counter+1



# Used rounding for predictions but will leave this here
'''

for i in range(len(model_1_prob)):
    if model_1_prob[i]<0.5:
        model_1_pred[i] = 0
    else:
        model_1_pred[i] = 1      
'''

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 12:41:01 2021

@author: James
"""
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
import keras

import numpy as np

def ML(X_train, y_train, X_test, y_test, model_length, w_train, w_test, w_norm_train, forceFit, close, type_tag, 
       epochs, batch, lr, 
       doES, ESpat,
       doRL, RLrate, RLpat,
       input_node, mid_node, extra_node=0, optimisation=False):
    
    # Try to load the ML model
    try:
        
        # If it's an optimisation run, skip to the except catch
        if optimisation:
            raise Exception
        
        # If force fit is enabled, skip to the except catch
        if forceFit:
            
            print()
            print('**************************')
            print('Forcing '+ type_tag[0] + ' Fit: ' + type_tag[1])
            print('**************************')
            print()
            raise Exception
            
        # Load the model
        model = keras.models.load_model(type_tag[0] + "_models/" + type_tag[1] + "/model")
        print(type_tag[0] + ' ' + type_tag[1] + ' model loaded')
        
        # Load the previous X data for this model and make predictions
        X_test = np.load(type_tag[0] + "_models/" + type_tag[1] + '/X_test.npy', allow_pickle=True)
        X_train = np.load(type_tag[0] + "_models/" + type_tag[1] + '/X_train.npy', allow_pickle=True)
        
        # Load the previous y and w data for this model to return to plots
        y_test = np.load(type_tag[0] + "_models/" + type_tag[1] + '/y_test.npy', allow_pickle=True)
        y_train = np.load(type_tag[0] + "_models/" + type_tag[1] + '/y_train.npy', allow_pickle=True)
        
        w_test = np.load(type_tag[0] + "_models/" + type_tag[1] + '/w_test.npy', allow_pickle=True)
        w_train = np.load(type_tag[0] + "_models/" + type_tag[1] + '/w_train.npy', allow_pickle=True)
        
        
    except:

        if not forceFit:        
            print()
            print('++++++++++++++++++++++++++')
            print('No model for '+ type_tag[0] + ' Fit: ' + type_tag[1])
            print('Retraining the model')
            print('++++++++++++++++++++++++++')
            print()
            
        # THE MODEL #
        model = Sequential()
        model.add(Dense(input_node, input_dim=model_length, activation='relu'))
        model.add(Dense(mid_node, activation='relu'))
        if extra_node != 0:
            model.add(Dense(extra_node, activation='sigmoid'))
        
        model.add(Dense(1, activation='sigmoid'))
        
        # Define the optimiser and its learning rate
        
        # Adam SGD with an adaptive moment
        if lr != 0:
            opt = keras.optimizers.Adam(learning_rate=lr)
        else:
            print('Learning rate not specified \n')
            opt = 'adam'
        
        # Stochastic grad descent optimiser, optional momentum and decay
        #opt = keras.optimizers.SGD(lr=lr, momentum=0.9)
        
        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        
        # Prepare validation data
        # N = X_train.shape[0]
        # N_val = int(0.8*N)
        
        # X_val = X_train[N_val:,:]
        # X_train_cut = X_train[0:N_val,:]
        
        # y_val = y_train[N_val:]
        # y_train_cut = y_train[0:N_val]
        
        # Convert all to flaot32
        w_train = np.asarray(w_train).astype('float32')
        w_test = np.asarray(w_test).astype('float32')
        w_norm_train = np.asarray(w_norm_train).astype('float32')
        X_train = np.asarray(X_train).astype('float32')
        y_train = np.asarray(y_train).astype('float32')

        # Models
        if doES and doRL:
            # Define early stopping
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=ESpat)
            # Define learning rate reduction
            rlrop = ReduceLROnPlateau(monitor='val_loss', factor=RLrate, patience=RLpat)
            # Fit the model with early stopping and reducable learning rate
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch,
                                        sample_weight=w_norm_train, validation_split=0.2,
                                        callbacks=[es, rlrop])
        elif doES:
            # Define early stopping
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=ESpat)
            # Fit the model with early stopping 
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch,
                                        sample_weight=w_norm_train, validation_split=0.2,
                                        callbacks=[es])
        elif doRL:
            # Define learning rate reduction
            rlrop = ReduceLROnPlateau(monitor='val_loss', factor=RLrate, patience=RLpat)
            # Fit the model with reducable learning rate
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch,
                                        sample_weight=w_norm_train, validation_split=0.2, 
                                        callbacks=[rlrop])
        else:
            # Fit the model without early stopping and reducable learning rate
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch,
                                        sample_weight=w_norm_train, validation_split=0.2)
        
        print(history.history.keys())

        if not optimisation:
            
            # Save the model for loading next time
            model.save(type_tag[0] + "_models/" + type_tag[1] + "/model")
            
            # Save the X and y data for loading (without randomising again) 
            np.save(type_tag[0] + "_models/" + type_tag[1] + '/X_test', X_test)
            np.save(type_tag[0] + "_models/" + type_tag[1] + '/X_train', X_train)
            np.save(type_tag[0] + "_models/" + type_tag[1] + '/y_test', y_test)
            np.save(type_tag[0] + "_models/" + type_tag[1] + '/y_train', y_train)
            np.save(type_tag[0] + "_models/" + type_tag[1] + '/w_train', w_train)
            np.save(type_tag[0] + "_models/" + type_tag[1] + '/w_test', w_test)
            
            # Make a plot of accuracy etc.
            
            plt.figure(figsize=(10.0,8.0))
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title(type_tag[0] + ' ' + type_tag[1] + ' Model accuracy', fontsize=40)
            plt.ylabel('Accuracy', fontsize=25)
            plt.xlabel('Epoch', fontsize=25)
            plt.legend(['Train', 'Val'], loc='best')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.savefig("Plots/" + type_tag[0] + "/" + type_tag[1] + "/" + "model_accuracy.png")
            
            if close: 
                plt.close()
            else:
                plt.show()
            
            plt.figure(figsize=(10.0,8.0))
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title(type_tag[0] + ' ' + type_tag[1] + ' Model loss', fontsize=40)
            plt.ylabel('Loss', fontsize=25)
            plt.xlabel('Epoch', fontsize=25)
            plt.legend(['Train', 'Val'], loc='best')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.savefig("Plots/" + type_tag[0] + "/" + type_tag[1] + "/" + "model_loss.png")
            
        if close: 
            plt.close()
        else:
            plt.show()
            
    # Make predictions for test and train
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    # Return the predictions and the y values (in case they are loaded)
    return pred_train, pred_test, y_train, y_test, w_train, w_test


def ML_opt(X_train, y_train, X_test, y_test, w_train, w_test, w_norm_train, model_length,
       epochs, batch, lr, 
       doES, ESpat,
       doRL, RLrate, RLpat,
       input_node, mid_node, extra_node=0,
       
       type_tag=0):
    
    import Functions as f
    
    # LR, epochs, batch, input node and mid node lengths
    # If integer set to one
    try:
        lr_len = len(lr)
    except:
        lr_len = 1
        lr = np.array([lr])
    
    try:
        epoch_len = len(epochs)
    except:
        epoch_len = 1
        epochs = np.array([epochs])
        
    try:
        batch_len = len(batch)
    except:
        batch_len = 1
        batch = np.array([batch])
    
    try:
        input_node_len = len(input_node)
    except:
        input_node_len = 1
        input_node = np.array([input_node])
    
    try:
        mid_node_len = len(mid_node)
    except:
        mid_node_len = 1
        mid_node = np.array([mid_node])
        

    try:
        extra_node_len = len(extra_node)
    except:
        extra_node_len = 1
        extra_node = np.array([extra_node])

    all_length = lr_len*epoch_len*batch_len*input_node_len*mid_node_len*extra_node_len
    #AUC_array = np.zeros(all_length)
    
    counter = 0
    
    for i in range(lr_len):
        for j in range(epoch_len):
            for k in range(batch_len):
                for l in range(input_node_len):
                    for m in range(mid_node_len):
                        for p in range(extra_node_len):
        
                            print('**************')
                            print('Opt. run: ' + str(counter+1) + ' / ' + str(all_length))
                            print('**************')                
            
                            if p != 0:
                                add_term = ' ' + ' - node3 = ' + str(extra_node[p]) 
                            else:
                                add_term = ''

            
                            ### The model ###   
                            pred_train, pred_test, y_train, y_test, w_train, w_test = ML(X_train, y_train, X_test, y_test, model_length,
                                                                        w_train, w_test, w_norm_train,
                                         forceFit=True, close=False, optimisation=True,
                                         
                                         # Epochs batch and lr
                                         epochs = epochs[j], batch = batch[k], lr = lr[i],
                                         
                                         # Early stopping
                                         doES=True, ESpat=ESpat,
                                         
                                         # Learning rate reduction
                                         doRL=False, RLrate=RLrate, RLpat=RLpat,
                                        
                                         
                                         # Nodes
                                         input_node = input_node[l], mid_node = mid_node[m], extra_node = extra_node[p],
                                         
                                         type_tag=0
                                         )
                        
                            #AUC = 
                            f.ROC_Curve(pred_train, pred_test, y_train, y_test, close=True, 
                                        title=(type_tag[0] + '_' + type_tag[1]), 
                                        
                                        saveas=(type_tag[0] + '/' + type_tag[1] + '/Optimisation/' + 'ROC ' +
                                                
                                                'lr =' + str(lr[i]) + ' ' +
                                                'epoch =' + str(epochs[j]) + ' ' +
                                                'batch =' + str(batch[k]) + ' ' +
                                                
                                                ' - node1 = ' +str(input_node[l]) +  ' ' +
                                                ' - node2 = ' + str(mid_node[m]) +  add_term
                                                
                                                ))     

                            f.ProbHist(pred_train, pred_test, y_train, y_test, w_train, w_test, 21, close=True, 
                                     label=['ttZ','ggA_600_500'], xtitle="Probability", ytitle="Events", 
                                        title=(type_tag[0] + '_' + type_tag[1]), 
                                        
                                        saveas=(type_tag[0] + '/' + type_tag[1] + '/Optimisation/' + 
                                                
                                                'lr =' + str(lr[i]) + ' ' +
                                                'epoch =' + str(epochs[j]) + ' ' +
                                                'batch =' + str(batch[k]) + ' ' +
                                                
                                                ' - node1 = ' +str(input_node[l]) +  ' ' +
                                                ' - node2 = ' + str(mid_node[m]) +  add_term
                                                
                                                ),
                                        
                                        
                                        addText = 'epoch: ' + str(epochs[j]) + '\n' + 
                                        'batch: ' + str(batch[k]))
                            
                            #AUC_array[counter] = AUC
                            counter = counter+1
                        
    #max_AUC = np.amax(AUC_array)



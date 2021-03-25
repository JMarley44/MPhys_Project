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

def ML(X_train, y_train, y_binary, X_all, model_length, forceFit, close, type_tag, 
       epochs, batch, lr, 
       doES, ESpat,
       doRL, RLrate, RLpat,
       input_node, mid_node, optimisation=False):
    
    # Try to load the ML model
    try:
        
        # If it's an optimisation run, skip to the except catch
        if optimisation:
            raise Exception
        
        # If force fit is enabled, skip to the except catch
        if forceFit:
            
            print()
            print('***********************')
            print('Forcing '+ type_tag[0] + ' Fit: ' + type_tag[1])
            print('***********************')
            print()
            raise Exception
            
        # Load the model
        model = keras.models.load_model(type_tag[0] + "_models/" + type_tag[1] + "_model")
        print(type_tag[0] + ' ' + type_tag[1] + ' model loaded')
        
        # Load the X data for this model and make a prediction
        X_load = np.load(type_tag[0] + "_models/" + type_tag[1] + '_X_all.npy', allow_pickle=True)
        pred = model.predict(X_load)
        
        # Load the y data for this model to return to plots
        y_return = np.load(type_tag[0] + "_models/" + type_tag[1] + '_y_all.npy', allow_pickle=True)
        
    except:
 
        ### MODEL PARAMETERS ###
        
        extra_node = int(input_node*(3/2))

        # THE MODEL #
        model = Sequential()
        model.add(Dense(input_node, input_dim=model_length, activation='relu'))
        
        model.add(Dense(extra_node, activation='relu'))
        
        model.add(Dense(mid_node, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        
        # Define the optimiser and its learning rate
        
        # Adam SGD with an adaptive moment
        opt = keras.optimizers.Adam(learning_rate=lr)
        
        # Stochastic grad descent optimiser, optional momentum and decay
        #opt = keras.optimizers.SGD(lr=lr, momentum=0.9)
        
        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        
        # Prepare validation data
        N = X_train.shape[0]
        N_val = int(0.7*N)
        
        X_val = X_train[N_val:,:]
        X_train = X_train[0:N_val,:]
        
        y_val = y_train[N_val:]
        y_train = y_train[0:N_val]

        # Models
        if doES and doRL:
            # Define early stopping
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=ESpat)
            # Define learning rate reduction
            rlrop = ReduceLROnPlateau(monitor='val_loss', factor=RLrate, patience=RLpat)
            # Fit the model with early stopping and reducable learning rate
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch,
                                        validation_data=(X_val,y_val), callbacks=[es, rlrop])
        elif doES:
            # Define early stopping
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=ESpat)
            # Fit the model with early stopping 
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch,
                                        validation_data=(X_val,y_val), callbacks=[es])
        elif doRL:
            # Define learning rate reduction
            rlrop = ReduceLROnPlateau(monitor='val_loss', factor=RLrate, patience=RLpat)
            # Fit the model with reducable learning rate
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch,
                                        validation_data=(X_val,y_val), callbacks=[rlrop])
        else:
            # Fit the model without early stopping and reducable learning rate
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch,
                                        validation_data=(X_val,y_val))
        
        print(history.history.keys())

        if not optimisation:
            
            # Save the model for loading next time
            model.save(type_tag[0] + "_models/" + type_tag[1] + "_model")
            
            # Save the X and y data for loading (without randomising) 
            np.save(type_tag[0] + "_models/" + type_tag[1] + '_X_all', X_all) 
            y_return = y_binary
            np.save(type_tag[0] + "_models/" + type_tag[1] + '_y_all', y_return)
            
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
        else:
            # For optimisation return an arbritray y (unused in optimisation)
            y_return = 0
            
        if close: 
            plt.close()
        else:
            plt.show()
            
        pred = model.predict(X_all)

    return pred, y_return


def ML_opt(X_train, y_train, y_binary, X_all, model_length, N_train, N_arr, weight,
       epochs, batch, lr, 
       doES, ESpat,
       doRL, RLrate, RLpat,
       input_node, mid_node,
       
       type_tag):
    
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
    
    all_length = lr_len*epoch_len*batch_len*input_node_len*mid_node_len
    AUC_array = np.zeros(all_length)
    
    counter = 0
    
    for i in range(lr_len):
        for j in range(epoch_len):
            for k in range(batch_len):
                for l in range(input_node_len):
                    for m in range(mid_node_len):
        
                        print('**************')
                        print('Opt. run: ' + str(counter+1) + ' / ' + str(all_length))
                        print('**************')                
        
                        
        
                        ### The model ###
                        pred, _ = ML(X_train, y_train, y_binary, X_all, model_length, 
                                     forceFit=True, close=False, optimisation=True,
                                     
                                     # Epochs batch and lr
                                     epochs = epochs[j], batch = batch[k], lr = lr[i],
                                     
                                     # Early stopping
                                     doES=True, ESpat=ESpat,
                                     
                                     # Learning rate reduction
                                     doRL=False, RLrate=RLrate, RLpat=RLpat,
                                    
                                     
                                     # Nodes
                                     input_node = input_node[l], mid_node = mid_node[m],
                                     
                                     type_tag=0
                                     )
                    
                        AUC = f.ROC_Curve(y_binary, pred, close=True, 
                                    title=(type_tag[0] + '_' + type_tag[1]), 
                                    
                                    saveas=(type_tag[0] + '/' + type_tag[1] + '/Optimisation/' + 'ROC ' +
                                            
                                            'lr =' + str(lr[i]) + ' ' +
                                            'epoch =' + str(epochs[j]) + ' ' +
                                            'batch =' + str(batch[k]) + ' ' +
                                            
                                            ' - node1 = ' +str(input_node[l]) +  ' ' +
                                            ' - node2 = ' + str(mid_node[m])
                                            
                                            ))
                        
                        f.ProbHist(y_binary, pred, N_train, 21, weight, N_arr, close=True, 
                                 label=['ttZ','ggA_600_500'], xtitle="Probability", ytitle="Events", 
                                    title=(type_tag[0] + '_' + type_tag[1]), 
                                    
                                    saveas=(type_tag[0] + '/' + type_tag[1] + '/Optimisation/' + 
                                            
                                            'lr =' + str(lr[i]) + ' ' +
                                            'epoch =' + str(epochs[j]) + ' ' +
                                            'batch =' + str(batch[k]) + ' ' +
                                            
                                            ' - node1 = ' +str(input_node[l]) +  ' ' +
                                            ' - node2 = ' + str(mid_node[m])
                                            
                                            ),
                                    
                                    
                                    addText = 'epoch: ' + str(epochs[j]) + '\n' + 
                                    'batch: ' + str(batch[k]))
                        
                        AUC_array[counter] = AUC
                        counter = counter+1
                        
    max_AUC = np.amax(AUC_array)
    return max_AUC



# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 12:41:01 2021

@author: James
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import keras
import matplotlib.pyplot as plt

doFit = True
doES = True

def ML(X_train, y_train, X_test, y_test, model_length, doFit):
    
    if doFit:
        print('*************************')
        print('Forcing ML Fit is enabled')
        print('*************************')
        
        # Define the model
        model = Sequential()
        model.add(Dense(32, input_dim=model_length, activation='sigmoid'))
        model.add(Dense(16, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        if doES:
            # Fit the model with early stopping
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
            history = model.fit(X_train, y_train, epochs=200, batch_size=10,
                                        validation_data=(X_test,y_test), callbacks=[es])
        else:
            # Fit the model without early stopping
            history = model.fit(X_train, y_train, epochs=150, batch_size=15,
                                        validation_data=(X_test,y_test))
        
        print(history.history.keys())
    
        model.save("ML_model")
            
        # Make a plot of accuracy etc
        plt.figure(figsize=(10.0,8.0))
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("Plots/model_accuracy.png")
        plt.figure(figsize=(10.0,8.0))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig("Plots/model_loss.png")
        
    else:
        # Try to load the ML model
        try:
            model = keras.models.load_model("ML_model")
            print('ML model is loaded')
            
        # If it doesn't work, train the model
        except:
            print('ML model is missing, training a new model')
            # Define the model
            model = Sequential()
            model.add(Dense(12, input_dim=model_length, activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            if doES:
                # Fit the model with early stopping
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=35)
                history = model.fit(X_train, y_train, epochs=100, batch_size=20,
                                            validation_data=(X_test,y_test), callbacks=[es])
            else:
                # Fit the model without early stopping
                history = model.fit(X_train, y_train, epochs=100, batch_size=20,
                                            validation_data=(X_test,y_test))
            
            print(history.history.keys())
        
            model.save("ML_model")
                
            # Make a plot of accuracy etc
            plt.figure(figsize=(10.0,8.0))
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig("Plots/model_accuracy.png")
            plt.figure(figsize=(10.0,8.0))
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.savefig("Plots/model_loss.png")
            
    return model

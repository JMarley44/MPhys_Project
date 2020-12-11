# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:18:27 2020

@author: James
"""
import numpy as np

def Extractdata(dataset, N):
    
    ### Try to load the separated data from file, if it doesn't exist separate it manually
    
    try:
        data_ttZ = np.load('Arrays/dat_ttZ.npy', allow_pickle=True)
        data_ttWm = np.load('Arrays/dat_ttWm.npy', allow_pickle=True)
        data_ttWp = np.load('Arrays/dat_ttWp.npy', allow_pickle=True)
        data_ggA_460_360 = np.load('Arrays/dat_ggA_460_360.npy', allow_pickle=True)
        data_ggA_500_360 = np.load('Arrays/dat_ggA_500_360.npy', allow_pickle=True)
        data_ggA_600_360 = np.load('Arrays/dat_ggA_600_360.npy', allow_pickle=True)
        data_ggA_600_400 = np.load('Arrays/dat_ggA_600_400.npy', allow_pickle=True)
        data_ggA_600_500 = np.load('Arrays/dat_ggA_600_500.npy', allow_pickle=True)
        data_ggA_500_400 = np.load('Arrays/dat_ggA_500_400.npy', allow_pickle=True)
        
    except FileNotFoundError:
        
        data_ttZ = np.array([])
        data_ttWm = np.array([])
        data_ttWp = np.array([])
        data_ggA_460_360 = np.array([])
        data_ggA_500_360 = np.array([])
        data_ggA_600_360 = np.array([])
        data_ggA_600_400 = np.array([])
        data_ggA_600_500 = np.array([])
        data_ggA_500_400 = np.array([])
        
        for i in range(N):
            
            if (dataset[i,0]=='ttZ'):
                data_ttZ = np.vstack([data_ttZ, dataset[i,1:47]]) if data_ttZ.size else dataset[i,1:47]
        
            if (dataset[i,0]=='ttWm'):
                data_ttWm = np.vstack([data_ttWm, dataset[i,1:47]]) if data_ttWm.size else dataset[i,1:47]
                
            if (dataset[i,0]=='ttWp'):
                data_ttWp = np.vstack([data_ttWp, dataset[i,1:47]]) if data_ttWp.size else dataset[i,1:47]
        
            if (dataset[i,0]=='ggA_460_360'):
                data_ggA_460_360 = np.vstack([data_ggA_460_360, dataset[i,1:47]]) if data_ggA_460_360.size else dataset[i,1:47]
                
            if (dataset[i,0]=='ggA_500_360'):
                data_ggA_500_360 = np.vstack([data_ggA_500_360, dataset[i,1:47]]) if data_ggA_500_360.size else dataset[i,1:47]
                
            if (dataset[i,0]=='ggA_600_360'):
                data_ggA_600_360 = np.vstack([data_ggA_600_360, dataset[i,1:47]]) if data_ggA_600_360.size else dataset[i,1:47]
        
            if (dataset[i,0]=='ggA_600_400'):
                data_ggA_600_400 = np.vstack([data_ggA_600_400, dataset[i,1:47]]) if data_ggA_600_400.size else dataset[i,1:47]

            if (dataset[i,0]=='ggA_600_500'):
                data_ggA_600_500 = np.vstack([data_ggA_600_500, dataset[i,1:47]]) if data_ggA_600_500.size else dataset[i,1:47]

            if (dataset[i,0]=='ggA_500_400'):
                data_ggA_500_400 = np.vstack([data_ggA_500_400, dataset[i,1:47]]) if data_ggA_500_400.size else dataset[i,1:47]

        np.save('Arrays/dat_ttZ.npy', data_ttZ)
        np.save('Arrays/dat_ttWm.npy', data_ttWm)
        np.save('Arrays/dat_ttWp.npy', data_ttWp)
        np.save('Arrays/dat_ggA_460_360.npy', data_ggA_460_360)
        np.save('Arrays/dat_ggA_500_360.npy', data_ggA_500_360)
        np.save('Arrays/dat_ggA_600_360.npy', data_ggA_600_360)
        np.save('Arrays/dat_ggA_600_400.npy', data_ggA_600_400)
        np.save('Arrays/dat_ggA_600_500.npy', data_ggA_600_500)
        np.save('Arrays/dat_ggA_500_400.npy', data_ggA_500_400)
        
    return data_ttZ, data_ttWm, data_ttWp, data_ggA_460_360, data_ggA_500_360, data_ggA_600_360,data_ggA_600_400,data_ggA_600_500, data_ggA_500_400



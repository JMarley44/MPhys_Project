# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 19:55:22 2020

@author: James
"""

'''
# Setup
'''
import time

start_time = time.time()

import DataExtract as d
import Functions as f
import SVM as s
import ML as m

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Close all previous plots on start
plt.close('all')

###############
#  DATA LOAD  #
###############

forceCalc = True

# Try to load previously calculated values:
try:
    
    # If force Calc is enabled, skip to the except catch
    if forceCalc:
        raise Exception
    
    N_arr = np.load('Arrays/N_arr.npy')
    
    N_ttZ = N_arr[0]
    N_ttWm = N_arr[1]
    N_ttWp = N_arr[2]
    N_ggA_460_360 = N_arr[3]
    N_ggA_500_360 = N_arr[4]
    N_ggA_600_360 = N_arr[5]
    N_ggA_600_400 = N_arr[6]
    N_ggA_600_500 = N_arr[7]
    N_ggA_500_400 = N_arr[8]
    N = N_arr[9]

    Z_inv_mass = np.load('Arrays/Z_inv_mass.npy')
    
    # Leptons
    
    lep1_four_mom = np.load('Arrays/lep1_four_mom.npy', allow_pickle=True)
    lep2_four_mom = np.load('Arrays/lep2_four_mom.npy', allow_pickle=True)
    lep3_four_mom = np.load('Arrays/lep3_four_mom.npy', allow_pickle=True)
    
    lep1_pt = np.load('Arrays/lep1_pt.npy', allow_pickle=True)
    lep2_pt = np.load('Arrays/lep2_pt.npy', allow_pickle=True)
    lep3_pt = np.load('Arrays/lep3_pt.npy', allow_pickle=True)
    
    lep_dphi = np.load('Arrays/lep_dphi.npy', allow_pickle=True)
    lep_dr = np.load('Arrays/lep_dr.npy', allow_pickle=True)

    # Jets

    jet1_four_mom = np.load('Arrays/jet1_four_mom.npy', allow_pickle=True)
    jet2_four_mom = np.load('Arrays/jet2_four_mom.npy', allow_pickle=True)

    jet1_pt = np.load('Arrays/jet1_pt.npy', allow_pickle=True)
    jet2_pt = np.load('Arrays/jet2_pt.npy', allow_pickle=True)
    
    jet_dphi = np.load('Arrays/jet_dphi.npy', allow_pickle=True)
    jet_dr = np.load('Arrays/jet_dr.npy', allow_pickle=True)
    
    # bJets
    
    bjet1_four_mom = np.load('Arrays/bjet1_four_mom.npy', allow_pickle=True)
    bjet2_four_mom = np.load('Arrays/bjet2_four_mom.npy', allow_pickle=True)
    
    bjet1_pt = np.load('Arrays/bjet1_pt.npy', allow_pickle=True)
    bjet2_pt = np.load('Arrays/bjet2_pt.npy', allow_pickle=True)
    
    # tops
    
    top1_four_mom = np.load('Arrays/top1_four_mom.npy', allow_pickle=True)
    top2_four_mom = np.load('Arrays/top2_four_mom.npy', allow_pickle=True)

    top1_pt = np.load('Arrays/top1_pt.npy', allow_pickle=True)
    top2_pt = np.load('Arrays/top2_pt.npy', allow_pickle=True)
    top1_phi = np.load('Arrays/top1_phi.npy', allow_pickle=True)
    top2_phi = np.load('Arrays/top2_phi.npy', allow_pickle=True)
    
    top12_pt = np.load('Arrays/top12_pt.npy', allow_pickle=True)
    top_dphi = np.load('Arrays/top_dphi.npy', allow_pickle=True)
    top_dr = np.load('Arrays/top_dr.npy', allow_pickle=True)
    
    # Z boson
    
    Z_four_mom = np.load('Arrays/Z_four_mom.npy', allow_pickle=True)
    Z_pz = np.load('Arrays/Z_pz.npy')
    Z_pt = np.load('Arrays/Z_pt.npy')
    
    # System masses
    
    delta_m = np.load('Arrays/delta_m.npy')
    A_pt = np.load('Arrays/A_pt.npy')
    H_pt = np.load('Arrays/H_pt.npy')
    M_tt = np.load('Arrays/M_tt.npy')
    M_ttZ = np.load('Arrays/M_ttZ.npy')
    
    met_pt = np.load('Arrays/met_pt.npy', allow_pickle=True)
    Wm_mass = np.load('Arrays/Wm_mass.npy')
    Wp_mass = np.load('Arrays/Wp_mass.npy')
    neu_four_mom = np.load('Arrays/neu_four_mom.npy')
    
    weight = np.load('Arrays/weight.npy', allow_pickle=True) 
    
    print('Arrays succesfully loaded')

# If loading fails run the calculations:
except:

    #################
    #  DATA IMPORT  #
    #################    

    # Import and manipulation
    path = "C:/Users/James/Documents/Liverpool/Year 4/PHYS498 - Project/Python files/MPhys_Project/dataset_semileptonic.csv"
    dataimport = pd.read_csv(path, header=None)
    dataset_full = pd.DataFrame(dataimport).to_numpy()
    dataset = np.delete(dataset_full,0,1)
    
    # Find the length of the states
    N_ttZ,N_ttWm,N_ttWp,N_ggA_460_360,N_ggA_500_360,N_ggA_600_360,N_ggA_600_400,N_ggA_600_500,N_ggA_500_400  = d.DataSplit(dataset_full)
    
    # Length of the dataset
    N = len(dataset)
    
    #####################
    #  INPUT VARIABLES  #
    #####################
    
    W_mass = 80.38 # GeV
    Z_mass = 91.12 # GeV
    top_mass = 172.76 # GeV
    
    jet_count = 0
    bjet_count = 0
    jet_sig_count = 0
    
    'Weights'
    weight = d.ExtractVariable(dataset,'weight',1,1)
    
    'Leptons'
    # Whole data set
    lep1_pt = d.ExtractVariable(dataset,'lep', 1, 'pt')
    lep1_eta = d.ExtractVariable(dataset,'lep', 1, 'eta')
    lep1_phi = d.ExtractVariable(dataset,'lep', 1, 'phi')
    
    lep1_q = d.ExtractVariable(dataset,'lep', 1, 'q')
    lep1_fl = d.ExtractVariable(dataset,'lep', 1, 'fl')
    
    lep2_pt = d.ExtractVariable(dataset,'lep', 2, 'pt')
    lep2_eta = d.ExtractVariable(dataset,'lep', 2, 'eta')
    lep2_phi = d.ExtractVariable(dataset,'lep', 2, 'phi')
    
    lep2_q = d.ExtractVariable(dataset,'lep', 2, 'q')
    lep2_fl = d.ExtractVariable(dataset,'lep', 2, 'fl')
    
    lep3_pt = d.ExtractVariable(dataset,'lep', 3, 'pt')
    lep3_eta = d.ExtractVariable(dataset,'lep', 3, 'eta')
    lep3_phi = d.ExtractVariable(dataset,'lep', 3, 'phi')
    
    lep3_q = d.ExtractVariable(dataset,'lep', 3, 'q')
    lep3_fl = d.ExtractVariable(dataset,'lep', 3, 'fl')

            
    for i in range(len(lep1_pt)):
        if lep1_pt[i]<25:
            print(i, ' 1 p is BAD')
        if lep2_pt[i]<25:
            print(i, ' 2 p is BAD')
        if lep3_pt[i]<25:
            print(i, ' 3 p is BAD')
    
    'Jets'
    jet1_pt = d.ExtractVariable(dataset,'jet', 1, 'pt')
    jet1_eta = d.ExtractVariable(dataset,'jet', 1, 'eta')
    jet1_phi = d.ExtractVariable(dataset,'jet', 1, 'phi')
    
    jet2_pt = d.ExtractVariable(dataset,'jet', 2, 'pt')
    jet2_eta = d.ExtractVariable(dataset,'jet', 2, 'eta')
    jet2_phi = d.ExtractVariable(dataset,'jet', 2, 'phi')
    
    'b-Jets'
    bjet1_pt = d.ExtractVariable(dataset,'bjet', 1, 'pt')
    bjet1_eta = d.ExtractVariable(dataset,'bjet', 1, 'eta')
    bjet1_phi = d.ExtractVariable(dataset,'bjet', 1, 'phi')
    
    bjet2_pt = d.ExtractVariable(dataset,'bjet', 2, 'pt')
    bjet2_eta = d.ExtractVariable(dataset,'bjet', 2, 'eta')
    bjet2_phi = d.ExtractVariable(dataset,'bjet', 2, 'phi')
    
    for i in range(len(lep1_pt)):
        if jet1_pt[i]<20:
            print(i, ' j1 p is BAD')
        if jet2_pt[i]<20:
            print(i, ' j2 p is BAD')
        if bjet1_pt[i]<20:
            print(i, ' b1 p is BAD')
        if bjet1_pt[i]<20:
            print(i, ' b2 p is BAD')

    'MET'
    met_pt = d.ExtractVariable(dataset,'met', 1, 'pt')
    met_phi = d.ExtractVariable(dataset,'met', 1, 'phi')
    
    'Tops'
    top1_pt = d.ExtractVariable(dataset,'top', 1, 'pt')
    top1_eta = d.ExtractVariable(dataset,'top', 1, 'eta')
    top1_phi = d.ExtractVariable(dataset,'top', 1, 'phi')
    top1_m = d.ExtractVariable(dataset,'top', 1, 'm')
    
    top2_pt = d.ExtractVariable(dataset,'top', 2, 'pt')
    top2_eta = d.ExtractVariable(dataset,'top', 2, 'eta')
    top2_phi = d.ExtractVariable(dataset,'top', 2, 'phi')
    top2_m = d.ExtractVariable(dataset,'top', 2, 'm')
    
    ##################
    #  EMPTY ARRAYS  #
    ##################
    
    'Leptons'
    # Four momenta
    lep1_four_mom = np.zeros((N,4))
    lep2_four_mom = np.zeros((N,4))
    lep3_four_mom = np.zeros((N,4))
    
    # Four momenta sums
    lep12_four_mom = np.zeros((N,4))
    lep13_four_mom = np.zeros((N,4))
    lep23_four_mom = np.zeros((N,4))
    
    # Invariant masses
    lep12_inv_mass = np.zeros(N)
    lep13_inv_mass = np.zeros(N)
    lep23_inv_mass = np.zeros(N)
    
    lep_dphi = np.zeros(N)
    lep_dr = np.zeros(N)
    
    'Jets'
    # Four momenta
    jet1_four_mom = np.zeros((N,4))
    jet2_four_mom = np.zeros((N,4))
    
    # Four momenta sums
    jet12_four_mom = np.zeros((N,4))
    
    # Invariant masses
    jet12_inv_mass = np.zeros(N)
    
    'b-Jets'
    # Four momenta
    bjet1_four_mom = np.zeros((N,4))
    bjet2_four_mom = np.zeros((N,4))
    
    # Four momenta sums
    jet12_four_mom = np.zeros((N,4))
    
    # Invariant masses
    
    'MET'
    #MET momenta
    met_px = np.zeros(N)
    met_py = np.zeros(N)
    
    'tops'
    Wp_top1_mass = np.zeros(N)
    Wp_top2_mass = np.zeros(N)
    
    Wm_top1_mass = np.zeros(N)
    Wm_top2_mass = np.zeros(N)
    
    top1_four_mom = np.zeros((N,4))
    top2_four_mom = np.zeros((N,4))
    
    Wm_top1 = np.zeros((N,4))
    Wm_top2 = np.zeros((N,4))
    
    Wp_top1 = np.zeros((N,4))
    Wp_top2 = np.zeros((N,4))
    
    'Neutrino'
    neu_four_mom = np.zeros((N,4))
    
    'Wm'
    Wm_four_mom = np.zeros((N,4))
    Wm_mass = np.zeros(N)
    
    # For reconstruction
    lep_mag = np.zeros(N)
    lep_sq = np.zeros((N,4))
    met_mag = np.zeros(N)
    k = np.zeros(N)
    quad = np.zeros((2,N))
    absquad = np.zeros((2,N))
    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    discriminant = np.zeros(N)
    
    'Wp'
    Wp_four_mom = np.zeros((N,4))
    Wp_mass = np.zeros(N)
    
    'Z'
    Z_four_mom = np.zeros((N,4))
    Z_inv_mass = np.zeros(N)
    Z_pt = np.zeros(N)
    Z_delete = np.zeros(N)
    
    'Systems'
    
    test1 = np.zeros(N)
    test2 = np.zeros(N)
    
    tt_four_mom = np.zeros((N,4))
    ttZ_four_mom = np.zeros((N,4))
    
    M_tt = np.zeros(N)
    M_ttZ = np.zeros(N)
    delta_m = np.zeros(N)
    
    'SVM variables'
    delta_m_actual = np.zeros(N)
    
    ###############
    #  MAIN LOOP  #
    ###############
    
    for i in range(N):
        
        'Leptons'
        # Four momenta
        lep1_four_mom [i,:] = f.four_mom(lep1_pt[i], lep1_eta[i], lep1_phi[i])
        lep2_four_mom [i,:] = f.four_mom(lep2_pt[i], lep2_eta[i], lep2_phi[i])
        lep3_four_mom [i,:] = f.four_mom(lep3_pt[i], lep3_eta[i], lep3_phi[i])
        
        # Addition of four momenta
        lep12_four_mom [i,:] = lep1_four_mom[i,:]+lep2_four_mom[i,:]
        lep13_four_mom [i,:] = lep1_four_mom[i,:]+lep3_four_mom[i,:]
        lep23_four_mom [i,:] = lep2_four_mom[i,:]+lep3_four_mom[i,:]
        
        # Invariant mass
        lep12_inv_mass [i] = f.inv_mass(lep12_four_mom[i])
        lep13_inv_mass [i] = f.inv_mass(lep13_four_mom[i])
        lep23_inv_mass [i] = f.inv_mass(lep23_four_mom[i])
        
        'Jets'
        
        # Four momenta
        jet1_four_mom [i,:] = f.four_mom(jet1_pt[i], jet1_eta[i], jet1_phi[i])
        jet2_four_mom [i,:] = f.four_mom(jet2_pt[i], jet2_eta[i], jet2_phi[i])
        
        # Addition of four momenta
        jet12_four_mom [i,:] = jet1_four_mom[i,:]+jet2_four_mom[i,:]
        
        #Invariant mass
        jet12_inv_mass [i] = f.inv_mass(jet12_four_mom[i])
        
        'b-Jets'
        
        # Four momenta
        bjet1_four_mom [i,:] = f.four_mom(bjet1_pt[i], bjet1_eta[i], bjet1_phi[i])
        bjet2_four_mom [i,:] = f.four_mom(bjet2_pt[i], bjet2_eta[i], bjet2_phi[i])   
    
        #####################
        #  RECONSTRUCTIONS  #
        #####################
    
        'Z boson'
        # Calculate test mass differences
        Zm_diff_12 = np.sqrt((lep12_inv_mass - Z_mass)**2)
        Zm_diff_13 = np.sqrt((lep13_inv_mass - Z_mass)**2)
        Zm_diff_23 = np.sqrt((lep23_inv_mass - Z_mass)**2)
        
        lep12_viable = False
        lep13_viable = False
        lep23_viable = False
        
        # Flavour and charge viability tests
        if lep1_fl[i] == lep2_fl[i] and lep1_q[i] != lep2_q[i]:
            lep12_viable = True
        if lep1_fl[i] == lep3_fl[i] and lep1_q[i] != lep3_q[i]:
            lep13_viable = True
        if lep2_fl[i] == lep3_fl[i] and lep2_q[i] != lep3_q[i]:
            lep23_viable = True
        
        try:
            # if all lep combinations are viable test to see which is best
            if lep12_viable and lep13_viable and lep23_viable:
                if Zm_diff_12[i] < Zm_diff_13[i] and Zm_diff_12[i] < Zm_diff_23[i]:
                    # 12 is the answer
                    if Zm_diff_12[i]>15:
                        raise Exception
                    Z_four_mom[i] = lep12_four_mom[i]
                    lep_dphi[i] = f.dphi(lep1_phi[i], lep2_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep1_eta[i], lep2_eta[i])
                elif Zm_diff_13[i] < Zm_diff_23[i]:
                    # 13 is the answer
                    if Zm_diff_13[i]>15:
                        raise Exception
                    Z_four_mom[i] = lep13_four_mom[i]
                    lep_dphi[i] = f.dphi(lep1_phi[i], lep3_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep1_eta[i], lep3_eta[i])
                else:
                    # 23 is the answer
                    if Zm_diff_23[i]>15:
                        raise Exception
                    Z_four_mom[i] = lep23_four_mom[i]
                    lep_dphi[i] = f.dphi(lep2_phi[i], lep3_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep2_eta[i], lep3_eta[i])
            
            # if only 2 are viable check which one is best
            elif lep12_viable and lep13_viable:
                if Zm_diff_12[i] < Zm_diff_13[i]:
                    # 12 is the answer
                    if Zm_diff_12[i]>15:
                        raise Exception
                    Z_four_mom[i] = lep12_four_mom[i]
                    lep_dphi[i] = f.dphi(lep1_phi[i], lep2_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep1_eta[i], lep2_eta[i])
                else:
                    # 13 is the answer
                    if Zm_diff_13[i]>15:
                        raise Exception
                    Z_four_mom[i] = lep13_four_mom[i]
                    lep_dphi[i] = f.dphi(lep1_phi[i], lep3_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep1_eta[i], lep3_eta[i])
                
            elif lep12_viable and lep23_viable:
                if Zm_diff_12[i] < Zm_diff_23[i]:
                    # 12 is the answer
                    if Zm_diff_12[i]>15:
                        raise Exception
                    Z_four_mom[i] = lep12_four_mom[i]
                    lep_dphi[i] = f.dphi(lep1_phi[i], lep2_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep1_eta[i], lep2_eta[i])
                else:
                    # 23 is the answer
                    if Zm_diff_23[i]>15:
                        raise Exception
                    Z_four_mom[i] = lep23_four_mom[i]
                    lep_dphi[i] = f.dphi(lep2_phi[i], lep3_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep2_eta[i], lep3_eta[i])
                
            elif lep13_viable and lep23_viable:
                if Zm_diff_13[i] < Zm_diff_23[i]:
                    # 13 is the answer
                    if Zm_diff_13[i]>15:
                        raise Exception
                    Z_four_mom[i] = lep13_four_mom[i]
                    lep_dphi[i] = f.dphi(lep1_phi[i], lep3_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep1_eta[i], lep3_eta[i])
                else:
                    # 23 is the answer
                    if Zm_diff_23[i]>15:
                        raise Exception
                    Z_four_mom[i] = lep23_four_mom[i]
                    lep_dphi[i] = f.dphi(lep2_phi[i], lep3_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep2_eta[i], lep3_eta[i])
                
            # if only one path is viable choose it
            elif lep12_viable:
                # 12 is the answer
                if Zm_diff_12[i]>15:
                    raise Exception
                else:
                    Z_four_mom[i] = lep12_four_mom[i]
                    lep_dphi[i] = f.dphi(lep1_phi[i], lep2_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep1_eta[i], lep2_eta[i])
    
            elif lep13_viable:
                # 13 is the answer
                if Zm_diff_13[i]>15:
                    raise Exception
                else:
                    Z_four_mom[i] = lep13_four_mom[i]
                    lep_dphi[i] = f.dphi(lep1_phi[i], lep3_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep1_eta[i], lep3_eta[i])
            
            elif lep23_viable:
                # 23 is the answer
                if Zm_diff_23[i]>15:
                    raise Exception
                else:
                    Z_four_mom[i] = lep23_four_mom[i]
                    lep_dphi[i] = f.dphi(lep2_phi[i], lep3_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep2_eta[i], lep3_eta[i])
                    
            # If nothing is viable skip to except catch
            else:
                raise Exception
            
            # Test the pseudorapidity
            if lep1_eta[i]>2.5:
                raise Exception
            elif lep2_eta[i]>2.5:
                raise Exception
            elif lep3_eta[i]>2.5:
                raise Exception
            elif abs(jet1_eta[i])>3:
                raise Exception
            elif abs(jet2_eta[i])>3:
                raise Exception
            elif abs(bjet1_eta[i])>2.5:
                raise Exception
            elif abs(bjet2_eta[i])>2.5:
                raise Exception
            
        except:
            # If there's no viable decay mark the event for deletion later
            Z_delete[i] = 1
            
            # Shift each N value downwards if one below it is removed
            if i < N_ttZ:
                N_ttZ = N_ttZ - 1
            if i < N_ttWm:
                N_ttWm = N_ttWm - 1
            if i < N_ttWp:
                N_ttWp = N_ttWp - 1
            if i < N_ggA_460_360:
                N_ggA_460_360 = N_ggA_460_360 - 1
            if i < N_ggA_500_360:
                N_ggA_500_360 = N_ggA_500_360 - 1
            if i < N_ggA_600_360:
                N_ggA_600_360 = N_ggA_600_360 - 1
            if i < N_ggA_600_400:
                N_ggA_600_400 = N_ggA_600_400 - 1
            if i < N_ggA_600_500:
                N_ggA_600_500 = N_ggA_600_500 - 1
            if i < N_ggA_500_400:
                N_ggA_500_400 = N_ggA_500_400 - 1
            if i > N_ggA_500_400:
                N_ggA_500_400 = N_ggA_500_400 - 1
                
            # Adjust the main N value
            N = N-1
                
        lep_dphi[i] = f.dphi(lep1_phi[i], lep2_phi[i])
        lep_dr[i] = f.drangle(lep_dphi[i], lep1_eta[i], lep2_eta[i])
                
        Z_inv_mass[i] = f.inv_mass(Z_four_mom[i])
        Z_pt[i] = np.sqrt(((Z_four_mom [i,1])**2)+((Z_four_mom [i,2])**2))
        Z_pz = Z_four_mom[:,3]
        
        #!!! Add more description
        'Wm boson'
        # Approximate neutrino transverse with MET
        neu_four_mom[i,1] = met_px [i] = met_pt[i]*np.cos(met_phi[i])
        neu_four_mom[i,2] = met_py [i] = met_pt[i]*np.sin(met_phi[i])
        met_mag[i] = (met_px [i]*met_px [i])+(met_py [i]*met_py [i])
    
        lep_sq[i,:] = lep3_four_mom[i,:]*lep3_four_mom[i,:]
        lep_mag[i] = lep_sq[i,0]-(lep_sq[i,1]+lep_sq[i,2]+lep_sq[i,3])
        
        k[i] = ((W_mass*W_mass - lep_mag[i])/2) + ((lep3_four_mom[i,1]*met_px [i]) + (lep3_four_mom[i,2] * met_py [i]))
        
        a[i] =  (lep3_four_mom[i,0]*lep3_four_mom[i,0]) - (lep3_four_mom[i,3]*lep3_four_mom[i,3])
        b[i] = -2*k[i]*lep3_four_mom[i,3]
        c[i] =  (lep3_four_mom[i,0]*lep3_four_mom[i,0] *  met_mag[i])  -  (k[i]*k[i])
        
        discriminant[i] = (b[i]*b[i]) - (4*a[i]*c[i])
        
        quad[0,i] = (- b[i] - np.sqrt(abs(discriminant[i]))) / (2 * a[i])
        quad[1,i] = (- b[i] + np.sqrt(abs(discriminant[i]))) / (2 * a[i])
        
        
        if (discriminant[i] < 0):
            neu_four_mom[i,3] = - b[i] / (2 * a[i])
        
        else:
            for n in range(2):
                  absquad[n] = abs(quad[n])
                  if (absquad[0,i] < absquad[1,i]):
                      neu_four_mom[i,3] = quad[0,i]
                  else:
                      neu_four_mom[i,3] = quad[1,i]
    
        Wm_four_mom [i,0] = lep3_four_mom[i,0]+f.mom(met_px[i], met_py[i], (neu_four_mom[i,3]))
        for j in range(1,4):
            Wm_four_mom [i,j] = lep3_four_mom[i,j]+neu_four_mom[i,j]
            
        'Wp boson'
        for j in range(4):
            Wp_four_mom [i,j] = jet1_four_mom [i,j]+jet2_four_mom [i,j]
            
        Wm_mass[i] = f.inv_mass(Wm_four_mom[i])
        Wp_mass[i] = f.inv_mass(Wp_four_mom[i])
            
        'tops'
        
        # 1st scenario
        Wp_top1[i] = Wp_four_mom[i] + bjet1_four_mom[i]
        Wm_top2[i] = Wm_four_mom[i] + bjet2_four_mom[i]
        
        # 2nd scenario
        Wm_top1[i] = Wm_four_mom[i] + bjet1_four_mom[i]
        Wp_top2[i] = Wp_four_mom[i] + bjet2_four_mom[i]
        
        # 1st scenario mass
        Wp_top1_mass[i] = f.inv_mass(Wp_top1[i])
        Wm_top2_mass[i] = f.inv_mass(Wm_top2[i])
        
        # 2nd scenario mass
        Wm_top1_mass[i] = f.inv_mass(Wm_top1[i])
        Wp_top2_mass[i] = f.inv_mass(Wp_top2[i])
        
        # Test which scenario is closer to top mass
        test1[i] = np.sqrt(((Wp_top1_mass[i]+Wm_top2_mass[i])-(2*top_mass))**2)
        test2[i] = np.sqrt(((Wp_top2_mass[i]+Wm_top1_mass[i])-(2*top_mass))**2)
        
        if test1[i] < test2[i]:
            top1_four_mom[i] = Wp_top1[i]
            top2_four_mom[i] = Wm_top2[i]
        else:
            top1_four_mom[i] = Wm_top1[i]
            top2_four_mom[i] = Wp_top2[i]
            
        'Systems'
        # Calculation of the tt and ttZ systems
        tt_four_mom[i] = top1_four_mom[i] + top2_four_mom[i]
        ttZ_four_mom[i] = tt_four_mom[i] + lep12_four_mom[i]
        
        # Calculation of the masses and delta m
        M_tt [i] = f.inv_mass(tt_four_mom[i])
        M_ttZ[i] = f.inv_mass(ttZ_four_mom[i])
        delta_m[i] = M_ttZ[i] - M_tt[i]
        
        'SVM calculations'
        neu_four_mom[i,0] = (f.mom(neu_four_mom[i,1],neu_four_mom[i,2],neu_four_mom[i,3]))**2

    dtop_four_mom = top1_four_mom - top2_four_mom
    top12_pt = np.sqrt(((dtop_four_mom [:,1])**2)+((dtop_four_mom [:,2])**2))

    #top12_pt = top1_pt-top2_pt
    top_dphi = f.dphi(top1_phi, top2_phi)
    top_dr = f.drangle(top_dphi, top1_eta, top2_eta)
    
    jet_dphi = f.dphi(jet1_phi, jet2_phi)
    jet_dr = f.drangle(jet_dphi, jet1_eta, jet2_eta)
    
    A_pt = np.sqrt(((ttZ_four_mom [:,1])**2)+((ttZ_four_mom [:,2])**2))
    H_pt = np.sqrt(((tt_four_mom [:,1])**2)+((tt_four_mom [:,2])**2))
    
    # Remove non-viable decays
    
    # Could be better optimised by starting a new loop but the effects will be minimal
    
    # Leptons
    lep1_four_mom = lep1_four_mom[Z_delete==0,:]
    lep2_four_mom = lep2_four_mom[Z_delete==0,:]
    lep3_four_mom = lep3_four_mom[Z_delete==0,:]
    
    lep1_pt = lep1_pt[Z_delete==0]
    lep2_pt = lep2_pt[Z_delete==0]
    lep3_pt = lep3_pt[Z_delete==0]
    
    lep1_eta = lep1_eta[Z_delete==0]
    lep2_eta = lep2_eta[Z_delete==0]
    lep3_eta = lep3_eta[Z_delete==0]
    
    lep1_phi = lep1_phi[Z_delete==0]
    lep2_phi = lep2_phi[Z_delete==0]
    lep3_phi = lep3_phi[Z_delete==0]
    
    lep_dphi = lep_dphi[Z_delete==0]
    lep_dr = lep_dr[Z_delete==0]
    
    # Jets
    jet1_four_mom = jet1_four_mom[Z_delete==0,:]
    jet2_four_mom = jet2_four_mom[Z_delete==0,:]
    
    jet1_pt = jet1_pt[Z_delete==0]
    jet2_pt = jet2_pt[Z_delete==0]
    
    jet1_eta = jet1_eta[Z_delete==0]
    jet2_eta = jet2_eta[Z_delete==0]
    
    jet1_phi = jet1_phi[Z_delete==0]
    jet2_phi = jet2_phi[Z_delete==0]
    
    jet_dphi = jet_dphi[Z_delete==0]
    jet_dr = jet_dr[Z_delete==0]
    
    # bJets
    bjet1_four_mom = bjet1_four_mom[Z_delete==0,:]
    bjet2_four_mom = bjet2_four_mom[Z_delete==0,:]
    
    bjet1_pt = bjet1_pt[Z_delete==0]
    bjet2_pt = bjet2_pt[Z_delete==0]
    
    bjet1_eta = bjet1_eta[Z_delete==0]
    bjet2_eta = bjet2_eta[Z_delete==0]
    
    bjet1_phi = bjet1_phi[Z_delete==0]
    bjet2_phi = bjet2_phi[Z_delete==0]
    
    # Tops
    top1_four_mom = top1_four_mom[Z_delete==0,:]
    top2_four_mom = top2_four_mom[Z_delete==0,:]
    
    top1_pt = top1_pt[Z_delete==0]
    top2_pt = top2_pt[Z_delete==0]
    
    top1_eta = top1_eta[Z_delete==0]
    top2_eta = top2_eta[Z_delete==0]
    
    top1_phi = top1_phi[Z_delete==0]
    top2_phi = top2_phi[Z_delete==0]
    
    top12_pt = top12_pt[Z_delete==0]
    top_dphi = top_dphi[Z_delete==0]
    top_dr = top_dr[Z_delete==0]
    
    # MET
    met_pt = met_pt[Z_delete==0]
    met_phi = met_phi[Z_delete==0]
    
    # Z boson
    Z_four_mom = Z_four_mom[Z_delete==0,:]

    Z_pz = Z_pz[Z_delete==0]
    Z_pt = Z_pt[Z_delete==0]
    Z_inv_mass = Z_inv_mass[Z_delete==0]
    
    # W boson
    Wm_mass = Wm_mass[Z_delete==0]
    Wp_mass = Wp_mass[Z_delete==0]
    
    # System masses
    delta_m = delta_m[Z_delete==0]
    A_pt = A_pt[Z_delete==0]
    H_pt = H_pt[Z_delete==0]
    M_tt = M_tt[Z_delete==0]
    M_ttZ = M_ttZ[Z_delete==0]
    
    # Additional discriminants
    neu_four_mom = neu_four_mom[Z_delete==0]
    
    # WEights
    weight = weight[Z_delete==0]
    
    # Make an array of the new N values (excluding the ignored values)
    N_arr = np.array([N_ttZ,N_ttWm,N_ttWp,N_ggA_460_360,N_ggA_500_360,N_ggA_600_360,
                      N_ggA_600_400,N_ggA_600_500,N_ggA_500_400,N])
    
    # Save the arrays for efficiency
    np.save('Arrays/N_arr.npy', N_arr)
    
    np.save('Arrays/Z_inv_mass.npy', Z_inv_mass)
    
    # Leptons
    np.save('Arrays/lep1_four_mom.npy', lep1_four_mom)     
    np.save('Arrays/lep2_four_mom.npy', lep2_four_mom)     
    np.save('Arrays/lep3_four_mom.npy', lep3_four_mom) 
    
    np.save('Arrays/lep1_pt.npy', lep1_pt)     
    np.save('Arrays/lep2_pt.npy', lep2_pt)     
    np.save('Arrays/lep3_pt.npy', lep3_pt)  
    
    np.save('Arrays/lep_dphi.npy', lep_dphi) # Maybe move this to SVM bunch
    np.save('Arrays/lep_dr.npy', lep_dr)
    
    # Jets
    np.save('Arrays/jet1_four_mom.npy', jet1_four_mom)     
    np.save('Arrays/jet2_four_mom.npy', jet2_four_mom)
    
    np.save('Arrays/jet1_pt.npy', jet1_pt)     
    np.save('Arrays/jet2_pt.npy', jet2_pt)    
    
    np.save('Arrays/jet_dphi.npy', jet_dphi) # Maybe move this to SVM bunch
    np.save('Arrays/jet_dr.npy', jet_dr)
    
    # bJets
    
    np.save('Arrays/bjet1_four_mom.npy', bjet1_four_mom)     
    np.save('Arrays/bjet2_four_mom.npy', bjet2_four_mom)
    
    np.save('Arrays/bjet1_pt.npy', bjet1_pt)     
    np.save('Arrays/bjet2_pt.npy', bjet2_pt)
    
    # tops
    
    np.save('Arrays/top1_four_mom.npy', top1_four_mom)     
    np.save('Arrays/top2_four_mom.npy', top2_four_mom)
    
    np.save('Arrays/top1_pt.npy', top1_pt)     
    np.save('Arrays/top2_pt.npy', top2_pt)
    np.save('Arrays/top1_phi.npy', top1_phi)     
    np.save('Arrays/top2_phi.npy', top2_phi)  
    
    np.save('Arrays/top12_pt.npy', top12_pt)   
    np.save('Arrays/top_dphi.npy', top_dphi) # Maybe move this to SVM bunch
    np.save('Arrays/top_dr.npy', top_dr)
    
    # Z boson
    
    np.save('Arrays/Z_four_mom.npy', Z_four_mom)     
    np.save('Arrays/Z_pz.npy', Z_pz)   
    np.save('Arrays/Z_pt.npy', Z_pt)     
    
    # System masses
    np.save('Arrays/delta_m.npy', delta_m)
    np.save('Arrays/A_pt.npy', A_pt)
    np.save('Arrays/H_pt.npy', H_pt)
    np.save('Arrays/M_tt.npy', M_tt)     
    np.save('Arrays/M_ttZ.npy', M_ttZ)     
        
    np.save('Arrays/met_pt.npy', met_pt)     
    np.save('Arrays/Wm_mass.npy', Wm_mass)     
    np.save('Arrays/Wp_mass.npy', Wp_mass)       
    np.save('Arrays/neu_four_mom.npy', neu_four_mom)        
    np.save('Arrays/weight.npy', weight)    

###########
#  PLOTS  #
###########

'Singular histograms'

# Z invariant mass
f.Hist(Z_inv_mass, 21, close=True, xtitle="$m_{Z}$ (GeV)", ytitle="Events", 
       title="Z boson invariant mass", xmin = 70, xmax = 110)


'Stacked signal histograms'

###########     ttZ         other
# Signals #     460_360    500_360    600_360
###########     600_400    600_500    500_400

all_signals = ['ttZ', 'other','460_360', '500_360', '600_360', '600_400', '600_500', '500_400']
select_signals = ['ttZ', 'other','500_360', '600_360', '600_500']

all_line = ['-',':','--','--','-',':']
select_line = ['--',':','--','--','-','--']


# Plots with all signals
f.SignalHist(lep1_pt, weight, 25, N_arr, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="Lepton 1 $p_T$",
             saveas="Lepton1_pt", signals = select_signals, line = select_line, scale='log')

f.SignalHist(lep2_pt, weight, 25, N_arr, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="Lepton 2 $p_T$",
             saveas="Lepton2_pt", signals = select_signals, line = select_line, scale='log')

f.SignalHist(lep3_pt, weight, 25, N_arr, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="Lepton 3 $p_T$",
             saveas="Lepton3_pt", signals = select_signals, line = select_line, scale='log')

f.SignalHist(jet1_pt, weight, 25, N_arr, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="Jet1 $p_T$",
             saveas="Jet1_pt", signals = select_signals, line = select_line, scale='log')

f.SignalHist(jet2_pt, weight, 25, N_arr, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="Jet2 $p_T$",
             saveas="Jet2_pt", signals = select_signals, line = select_line, scale='log')

f.SignalHist(bjet1_pt, weight, 25, N_arr, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="bJet1 $p_T$",
             saveas="bJet1_pt", signals = select_signals, line = select_line, scale='log')

f.SignalHist(bjet2_pt, weight, 25, N_arr, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="bJet2 $p_T$",
             saveas="bJet2_pt", signals = select_signals, line = select_line, scale='log')

# Z pt
f.SignalHist(Z_pt, weight, 25, N_arr, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="$Z_{pT}$",
             saveas="Z_pT", signals = select_signals, line = select_line)

# delta m
delta_m_bkg_count, delta_m_sig_count = f.SignalHist(delta_m, weight, 26, N_arr, close=True, xtitle="$\Delta$m (GeV)", ytitle="Events", title="$\Delta$m",
             saveas="delta_m", signals = all_signals, line = all_line, xlim=[0, 2000], addText = 'Weighted to luminosity')


'Testing discriminating variables plots'

# delta m
f.SignalHist(delta_m, weight, 26, N_arr, close=True, xtitle="$\Delta$m (GeV)", ytitle="Events", title="$\Delta$m",
             saveas="disc_delta_m", signals = select_signals, 
             shaped=True, line = select_line, xlim=[0, 1000])

# met pt
f.SignalHist(met_pt, weight, 25, N_arr, close=True, xtitle=r'$E^{T}_{miss}$ (GeV)', ytitle="Events", title=r'$E^{T}_{miss}$',
             saveas="disc_met_pt", signals = select_signals, 
             shaped = True, line = select_line, xlim=[0, 600])

# lep dr
f.SignalHist(lep_dr, weight, 26, N_arr, close=True, xtitle=r'lep angle r (rad)', ytitle="rad", title=r'lepton ${\Delta}R$',
              saveas="disc_lep_dr", signals = select_signals, 
              shaped = True, line = select_line)

# Wp mass
f.SignalHist(Wp_mass, weight, 26, N_arr, close=True, xtitle=r'$W^+$ mass (GeV)', ytitle="Events", title=r'$W^+$ mass',
             saveas="disc_Wp_mass", signals = select_signals, 
             shaped = True, line = select_line, xlim=[0, 1000])

# Wm mass
f.SignalHist(Wm_mass, weight, 25, N_arr, close=True, xtitle=r'$W^-$ mass (GeV)', ytitle="Events", title=r'$W^-$ mass',
             saveas="disc_Wm_mass", signals = select_signals, 
             shaped = True, line = select_line, xlim=[0, 600])

# Neutrino pz
f.SignalHist(neu_four_mom[:,3], weight, 26, N_arr, close=True, xtitle=r'Neutrino $p_Z$ (GeV)', ytitle="Events", title=r'Neutrino $p_Z$',
             saveas="disc_Neutrino_pz", signals = select_signals, 
             shaped = True, line = select_line, xlim=[-500, 500])

# Z pt
f.SignalHist(Z_pt, weight, 25, N_arr, close=True, xtitle=r'$p_T$ (GeV)', ytitle="Events", title=r'$Z_{p_T}$',
             saveas="disc_Z_pT", signals = select_signals, 
             shaped = True, line = select_line, xlim=[0, 600])

# Z pz
f.SignalHist(Z_pz, weight, 25, N_arr, close=True, xtitle=r'$p_z$ (GeV)', ytitle="Events", title=r'$Z_{p_z}$',
             saveas="disc_Z_pz", signals = select_signals, 
             shaped = True, line = select_line, xlim=[-1000, 1000])

# M_ttZ
f.SignalHist(M_ttZ, weight, 26, N_arr, close=True, xtitle=r'ttZ mass (GeV)', ytitle="Events", title=r'ttZ system mass',
             saveas="disc_ttZ_m", signals = select_signals, 
             shaped = True, line = select_line, xlim=[0, 3000])

# M_ttZ
f.SignalHist(M_tt, weight, 26, N_arr, close=True, xtitle=r'tt mass (GeV)', ytitle="Events", title=r'tt system mass',
             saveas="disc_tt_m", signals = select_signals, 
             shaped = True, line = select_line, xlim=[0, 3000])


# top1 pt
f.SignalHist(top1_pt, weight, 25, N_arr, close=True, xtitle=r'$p_T$ (GeV)', ytitle="Events", title=r'top1 $p_T$',
             saveas="disc_top1_pt", signals = select_signals, 
             shaped = True, line = select_line, xlim=[0, 1000])

# top2 pt
f.SignalHist(top2_pt, weight, 26, N_arr, close=True, xtitle=r'$p_T$ (GeV)', ytitle="Events", title=r'top2 $p_T$',
             saveas="disc_top2_pt", signals = select_signals, 
             shaped = True, line = select_line, xlim=[0, 800])

# top1-2 pt
f.SignalHist(top12_pt, weight, 26, N_arr, close=True, xtitle=r'$p_T$ (GeV)', ytitle="Events", title=r'top12 $p_T$',
             saveas="disc_top12_pt", signals = select_signals, 
             shaped = True, line = select_line, xlim=[0, 1200])

# top dphi
f.SignalHist(top_dphi, weight, 26, N_arr, close=True, xtitle=r't$\bar{t}$ angle \phi (rad)', ytitle="rad", title=r'top quarks $\Delta\phi$',
              saveas="disc_top_dphi", signals = select_signals, 
              shaped = True, line = select_line)

# top dr
f.SignalHist(top_dr, weight, 26, N_arr, close=True, xtitle=r't$\bar{t}$ angle r (rad)', ytitle="rad", title=r'top quarks ${\Delta}R$',
              saveas="disc_top_dr", signals = select_signals, 
              shaped = True, line = select_line)

# jet dphi
f.SignalHist(jet_dphi, weight, 26, N_arr, close=True, xtitle=r'jets angle \phi (rad)', ytitle="rad", title=r'jet $\Delta\phi$',
              saveas="disc_jet_dphi", signals = select_signals, 
              shaped = True, line = select_line)

# jet dr
f.SignalHist(jet_dr, weight, 26, N_arr, close=True, xtitle=r'jet angle r (rad)', ytitle="rad", title=r'jet ${\Delta}R$',
              saveas="disc_jet_dr", signals = select_signals, 
              shaped = True, line = select_line)

# A pt
f.SignalHist(A_pt, weight, 26, N_arr, close=True, xtitle=r'$p_T$ (GeV)', ytitle="Events", title=r'$A_{pT}$',
             saveas="disc_A_pT", signals = select_signals, 
             shaped = True, line = select_line, xlim=[0, 500])

# H pt
f.SignalHist(H_pt, weight, 26, N_arr, close=True, xtitle=r'$p_T$ (GeV)', ytitle="Events", title=r'$H_{pT}$',
             saveas="disc_H_pT", signals = select_signals, 
             shaped = True, line = select_line, xlim=[0, 600])

# top1 phi
f.SignalHist(top1_phi, weight, 26, N_arr, close=True, xtitle=r't$\bar{t}$ angle \phi (rad)', ytitle="rad", title=r'top quark \phi$',
              saveas="disc_top1_phi", signals = select_signals, 
              shaped = True, line = select_line)

# top2 phi
f.SignalHist(top2_phi, weight, 26, N_arr, close=True, xtitle=r't$\bar{t}$ angle \phi (rad)', ytitle="rad", title=r'top quark \phi$',
              saveas="disc_top2_phi", signals = select_signals, 
              shaped = True, line = select_line)

#############################
#  Machine learning aspect  #
#############################

'INFO'
# ggA_460_360 - [N_ttWp:N_ggA_460_360]              length = 758
# ggA_500_360 - [N_ggA_460_360:N_ggA_500_360]       length = 707
# ggA_600_360 - [N_ggA_500_360:N_ggA_600_360]       length = 1092
# ggA_600_400 - [N_ggA_600_360:N_ggA_600_400]       length = 1168
# ggA_600_500 - [N_ggA_600_400:N_ggA_600_500]       length = 1372
# ggA_500_400 - [N_ggA_600_500:N_ggA_500_400]       length = 2059

# Ordering of the signal types
# N_ttZ,N_ttWm,N_ttWp,N_ggA_460_360,N_ggA_500_360,N_ggA_600_360,N_ggA_600_400,N_ggA_600_500,N_ggA_500_400

# Round to obtain the predicted value
    # model_1_pred = np.around(model_1_all_prob)

#############################
#  SUPPORT VECTOR MACHINE   #
#############################

doSVM = True

forceModel1 = False
forceModel2 = False
forceModel3 = False
forceModel4 = False
forceModel5 = False

if doSVM:
    
    ###########
    # 600_500 #
    ###########
    
    ### Model 1 ###
    model_1 = (delta_m, Z_pt, H_pt, M_ttZ, top_dr)
    
    # Just m and Z
    #model_1 = (delta_m, Z_pt)
    
    # Prepare for SVM usage
    (model_1_data, model_1_data_norm, X_train_1, y_train_1, 
     X_test_1, y_test_1, y_binary_1, SVM_N_train_1) = f.data_prep(model_1, N_ttZ, N, N_ggA_600_400, N_ggA_600_500)
    
    # Build the SVM
    model_1_prob_train, model_1_prob_test = s.SVM(X_train_1, y_train_1, X_test_1, 
                                                  C=10, gamma=0.01, tol=1, tag='1', ForceModel=forceModel1)
    
    # C = 1 gamma = 2
    
    # Combine the output probabilities
    model_1_all_prob = np.concatenate((model_1_prob_train,model_1_prob_test))
    model_1_all_prob = model_1_all_prob.reshape(len(model_1_all_prob),1)
    
    ### PLOTS 1 ###
    
    # ROC Curve
    f.ROC_Curve(y_binary_1, model_1_all_prob, close=True, title="SVM_600_500_", saveas='SVM/600_500/')
    
    f.ProbHist(y_binary_1, model_1_all_prob, SVM_N_train_1, 21, weight, N_arr, close=True, 
          label=['ttZ','ggA_600_500'], xtitle="Probability of signal", ytitle="Events", 
            title="SVM_600_500", saveas='SVM/600_500/')
    
    ###########
    # 600_360 #
    ###########

    ### Model 2 ###
    model_2 = (delta_m, Z_pt, H_pt, M_ttZ, top_dr, met_pt)
    
    (model_2_data, model_2_data_norm, X_train_2, y_train_2, 
     X_test_2, y_test_2, y_binary_2, SVM_N_train_2) = f.data_prep(model_2, N_ttZ, N, N_ggA_500_360, N_ggA_600_360)
    
    model_2_prob_train, model_2_prob_test = s.SVM(X_train_2, y_train_2, X_test_2, 
                                                  C=0.1, gamma=5, tol=1, tag='2', ForceModel=forceModel2)

    model_2_all_prob = np.concatenate((model_2_prob_train,model_2_prob_test))
    model_2_all_prob = model_2_all_prob.reshape(len(model_2_all_prob),1)
    
    ### Plots 2 ###
    f.ROC_Curve(y_binary_2, model_2_all_prob, close=True, title="SVM_600_360_", saveas='SVM/600_360/')
    
    f.ProbHist(y_binary_2, model_2_all_prob, SVM_N_train_2, 21, weight, N_arr, close=True, 
          label=['ttZ','ggA_600_360'], xtitle="Probability of signal", ytitle="Events", 
            title="SVM_600_360", saveas='SVM/600_360/')
    
    ###########
    # 500_360 #
    ###########
        
    ### Model 3 ###
    model_3 = (delta_m, Z_pt, H_pt, M_ttZ, met_pt)
    
    (model_3_data, model_3_data_norm, X_train_3, y_train_3, 
     X_test_3, y_test_3, y_binary_3, SVM_N_train_3) = f.data_prep(model_3, N_ttZ, N, N_ggA_460_360, N_ggA_500_360)
    
    model_3_prob_train, model_3_prob_test = s.SVM(X_train_3, y_train_3, X_test_3, 
                                                  C=0.1, gamma=5, tol=1E-3, tag='3', ForceModel=forceModel3)
    
    model_3_all_prob = np.concatenate((model_3_prob_train,model_3_prob_test))
    model_3_all_prob = model_3_all_prob.reshape(len(model_3_all_prob),1)
    
    ### Plots 3 ###
    f.ROC_Curve(y_binary_3, model_3_all_prob, close=True, title="SVM_500_360_", saveas='SVM/500_360/')
    
    f.ProbHist(y_binary_3, model_3_all_prob, SVM_N_train_3, 21, weight, N_arr, close=True, 
          label=['ttZ','ggA_500_360'], xtitle="Probability of signal", ytitle="Events", 
            title="SVM_500_360", saveas='SVM/500_360/')

    ###########
    # 460_360 #
    ###########
       
    ### Model 4 ###
    model_4 = (delta_m, Z_pt, top1_pt, top2_pt)
    
    (model_4_data, model_4_data_norm, X_train_4, y_train_4, 
     X_test_4, y_test_4, y_binary_4, SVM_N_train_4) = f.data_prep(model_3, N_ttZ, N, N_ttWp, N_ggA_460_360)
    
    model_4_prob_train, model_4_prob_test = s.SVM(X_train_4, y_train_4, X_test_4, 
                                                  C=15, gamma=15, tol=1E-3, tag='4', ForceModel=forceModel4)
    
    model_4_all_prob = np.concatenate((model_4_prob_train,model_4_prob_test))
    model_4_all_prob = model_4_all_prob.reshape(len(model_4_all_prob),1)
    
    ### Plots 4 ###
    f.ROC_Curve(y_binary_4, model_4_all_prob, close=True, title="SVM_460_360_", saveas='SVM/460_360/')
    
    f.ProbHist(y_binary_4, model_4_all_prob, SVM_N_train_4, 21, weight, N_arr, close=True, 
          label=['ttZ','ggA_460_360'], xtitle="Probability of signal", ytitle="Events", 
            title="SVM_460_360", saveas='SVM/460_360/')

    ###########
    # 500_400 #
    ###########
    
    ### Model 5 ### 
    model_5 = (delta_m, Z_pt, H_pt)
    
    (model_5_data, model_5_data_norm, X_train_5, y_train_5, 
    X_test_5, y_test_5, y_binary_5, SVM_N_train_5) = f.data_prep(model_5, N_ttZ, N, N_ggA_600_500, N_ggA_500_400)
    
    model_5_prob_train, model_5_prob_test = s.SVM(X_train_5, y_train_5, X_test_5, 
                                                  C=1, gamma=1E-3, tol=1, tag='5', ForceModel=forceModel5)
    
    model_5_all_prob = np.concatenate((model_5_prob_train,model_5_prob_test))
    model_5_all_prob = model_5_all_prob.reshape(len(model_5_all_prob),1)

    ### Plots 5 ###
    f.ROC_Curve(y_binary_5, model_5_all_prob, close=True, title="SVM_500_400_", saveas='SVM/500_400/')

    f.ProbHist(y_binary_5, model_5_all_prob, SVM_N_train_5, 21, weight, N_arr, close=True, 
          label=['ttZ','ggA_500_400'], xtitle="Probability of signal", ytitle="Events", 
            title="SVM_500_400", saveas='SVM/500_400/')
      
    
#####################
#  SHALLOW NETWORK  #
#####################

doML = True

ForceML1 = False

ML_Opt_1 = False

if doML:
    
    ###########
    # 600_500 #
    ###########
    
    ### ML model 1 ###
    ML_model_1 = (delta_m, Z_pt, H_pt, M_ttZ, top_dr)
    
    # Prepare data for ML
    (ML_model_1_data, ML_model_1_data_norm, ML_X_train_1, ML_y_train_1, ML_X_test_1, 
     ML_y_test_1, ML_y_binary_1, ML_N_train_1) = f.data_prep(ML_model_1, N_ttZ, N, N_ggA_600_400, N_ggA_600_500)
    
    model_length = len(ML_model_1)
    
    # Extract all X data for prediction
    ML_all_1 = ML_model_1_data_norm[:][:,0:model_length]
        
    # Define the learning rate
    ML_lr_1 = 0.01
    ML_epoch_1 = 150
    ML_batch_1 = 20
    
    input_node = len(ML_X_train_1[0])
    mid_node = int(input_node*2)
    
    ### The model ###
    ML_pred_1, ML_y_binary_1 = m.ML(ML_X_train_1, ML_y_train_1, ML_y_binary_1, ML_all_1, model_length,
                 forceFit=ForceML1, close=True, type_tag = ['ML', '600_500'],
                 
                 # Epochs batch and lr
                 epochs = ML_epoch_1, batch = ML_batch_1, lr = ML_lr_1,
                 
                 # Early stopping
                 doES=True, ESpat=30,
                 
                 # Learning rate reduction
                 doRL=False, RLrate=0.1, RLpat=100,
                 
                 # Nodes
                 input_node = input_node, mid_node = mid_node)

    f.ROC_Curve(ML_y_binary_1, ML_pred_1, close=True, 
                title=('ML_600_500_'), saveas=('ML/600_500/'))
    
    f.ProbHist(ML_y_binary_1, ML_pred_1, ML_N_train_1, 21, weight, N_arr, close=True, 
             label=['ttZ','ggA_600_500'], xtitle="Probability", ytitle="Events", 
                title=('ML_600_500_'), saveas=('ML/600_500/'))

    if ML_Opt_1:
            
            # input_node = np.array([input_node*(1/2), input_node*(2/3), input_node, input_node*(4/3), input_node*(3/2)])
            # mid_node = np.array([mid_node*(1/2), mid_node*(2/3), mid_node, mid_node*(4/3), mid_node*(3/2)])
            
            input_node = np.array([input_node])
            mid_node = np.array([mid_node, mid_node*(4/3)])
            DL_epoch_1 = np.array([150])
            DL_batch_1 = np.array([10,20])
            DL_lr_1 = np.array([0.05, 0.01, 0.005, 0.001])
            
            m.ML_opt(ML_X_train_1, ML_y_train_1, ML_y_binary_1, ML_all_1, model_length, ML_N_train_1, N_arr, weight, 
               ML_epoch_1, ML_batch_1, ML_lr_1, 
               doES=True, ESpat=35,
               doRL=False, RLrate=0, RLpat=0,
               input_node=input_node, mid_node=mid_node,
               
               type_tag = ['ML', '600_500'])

##################
#  DEEP NETWORK  #
##################

doDL = True

ForceDL1 = False
ForceDL2 = True
ForceDL3 = False     # Cut test

DL_Opt_1 = False
DL_Opt_2 = False

if doDL:
    
    ###########
    # 600_500 #
    ###########
    
    DL_model_1 = (lep1_four_mom[:,0], lep1_four_mom[:,1], lep1_four_mom[:,2], lep1_four_mom[:,3],
                  lep2_four_mom[:,0], lep2_four_mom[:,1], lep2_four_mom[:,2], lep2_four_mom[:,3],
                  lep3_four_mom[:,0], lep3_four_mom[:,1], lep3_four_mom[:,2], lep3_four_mom[:,3],
                  
                  jet1_four_mom[:,0], jet1_four_mom[:,1], jet1_four_mom[:,2], jet1_four_mom[:,3],
                  jet2_four_mom[:,0], jet2_four_mom[:,1], jet2_four_mom[:,2], jet2_four_mom[:,3],
                  
                  bjet1_four_mom[:,0], bjet1_four_mom[:,1], bjet1_four_mom[:,2], bjet1_four_mom[:,3],
                  bjet2_four_mom[:,0], bjet2_four_mom[:,1], bjet2_four_mom[:,2], bjet2_four_mom[:,3],
                  
                  top1_four_mom[:,0], top1_four_mom[:,1], top1_four_mom[:,2], top1_four_mom[:,3],
                  top2_four_mom[:,0], top2_four_mom[:,1], top2_four_mom[:,2], top2_four_mom[:,3],
                  
                  Z_four_mom[:,0], Z_four_mom[:,1], Z_four_mom[:,2], Z_four_mom[:,3], 
                  
                  delta_m, Z_pt, H_pt, top_dr)

    # Prepare data for DL
    (DL_model_1_data, DL_model_1_data_norm, DL_X_train_1, DL_y_train_1, DL_X_test_1, 
     DL_y_test_1, DL_y_binary_1, DL_N_train_1) = f.data_prep(DL_model_1, N_ttZ, N, N_ggA_600_400, N_ggA_600_500)
    
    model_length = len(DL_model_1)
    
    # Extract all X data for prediction
    DL_all_1 = DL_model_1_data_norm[:][:,0:model_length]
    
    # Define the learning rate
    DL_lr_1 = 1E-4
    DL_epoch_1 = 300
    DL_batch_1 = 5
    
    input_node = len(DL_X_train_1[0])
    mid_node = int(input_node*2)
        
    ### The model ###
    DL_pred_1, DL_y_binary_1 = m.ML(DL_X_train_1, DL_y_train_1, DL_y_binary_1, DL_all_1, model_length, 
                 forceFit=ForceDL1, close=True, type_tag = ['DL', '600_500'],
                 
                 # Epochs batch and lr
                 epochs = DL_epoch_1, batch = DL_batch_1, lr = DL_lr_1,
                 
                 # Early stopping
                 doES=True, ESpat=35,
                 
                 # Learning rate reduction
                 doRL=False, RLrate=0.1, RLpat=20,
                 
                 # Nodes
                 input_node = input_node, mid_node = mid_node
                 )

    f.ROC_Curve(DL_y_binary_1, DL_pred_1, close=True, 
                title=('DL_600_500'), saveas=('DL/600_500/'))
    
    f.ProbHist(DL_y_binary_1, DL_pred_1, DL_N_train_1, 21, weight, N_arr, close=True, 
             label=['ttZ','ggA_600_500'], xtitle="Probability", ytitle="Events", 
                title=('DL_600_500_'), saveas=('DL/600_500/'))
    
    DL_1_bkg_count, DL_1_sig_count = f.ProbHist(DL_y_binary_1, DL_pred_1, 
             DL_N_train_1, 21, weight, N_arr, close=True, 
             label=['ttZ','ggA_600_500'], xtitle="Probability", ytitle="Events", 
                title=('DL_600_500_'), saveas=('DL/600_500/'), shaped=False)

    # Run optimisation if enabled
    if DL_Opt_1:
        
        # input_node = np.array([input_node*(1/2), input_node*(2/3), input_node, input_node*(4/3), input_node*(3/2)])
        # mid_node = np.array([mid_node*(1/2), mid_node*(2/3), mid_node, mid_node*(4/3), mid_node*(3/2)])
        
        input_node = np.array([input_node])
        mid_node = np.array([mid_node, mid_node*(4/3)])
        DL_epoch_1 = np.array([300])
        DL_batch_1 = np.array([15,20,32])
        DL_lr_1 = np.array([0.001, 0.0005, 0.00025, 0.0001, 0.00005 ,0.00001])
        
        m.ML_opt(DL_X_train_1, DL_y_train_1, DL_y_binary_1, DL_all_1, model_length, DL_N_train_1, N_arr, weight, 
           DL_epoch_1, DL_batch_1, DL_lr_1, 
           doES=True, ESpat=35,
           doRL=False, RLrate=0, RLpat=0,
           input_node=input_node, mid_node=mid_node,
           
           type_tag = ['DL', '600_500'])
        
        
    ###########    
    # 500_400 #
    ###########
    
    # Length 43
    DL_model_2 = (lep1_four_mom[:,0], lep1_four_mom[:,1], lep1_four_mom[:,2], lep1_four_mom[:,3],
                  lep2_four_mom[:,0], lep2_four_mom[:,1], lep2_four_mom[:,2], lep2_four_mom[:,3],
                  lep3_four_mom[:,0], lep3_four_mom[:,1], lep3_four_mom[:,2], lep3_four_mom[:,3],
                  
                  jet1_four_mom[:,0], jet1_four_mom[:,1], jet1_four_mom[:,2], jet1_four_mom[:,3],
                  jet2_four_mom[:,0], jet2_four_mom[:,1], jet2_four_mom[:,2], jet2_four_mom[:,3],
                  
                  bjet1_four_mom[:,0], bjet1_four_mom[:,1], bjet1_four_mom[:,2], bjet1_four_mom[:,3],
                  bjet2_four_mom[:,0], bjet2_four_mom[:,1], bjet2_four_mom[:,2], bjet2_four_mom[:,3],
                  
                  top1_four_mom[:,0], top1_four_mom[:,1], top1_four_mom[:,2], top1_four_mom[:,3],
                  top2_four_mom[:,0], top2_four_mom[:,1], top2_four_mom[:,2], top2_four_mom[:,3],
                  
                  Z_four_mom[:,0], Z_four_mom[:,1], Z_four_mom[:,2], Z_four_mom[:,3], 
                  
                  delta_m, Z_pt, H_pt)

    # Prepare data for DL
    (DL_model_2_data, DL_model_2_data_norm, DL_X_train_2, DL_y_train_2, DL_X_test_2, 
     DL_y_test_2, DL_y_binary_2, DL_N_train_2) = f.data_prep(DL_model_2, N_ttZ, N, N_ggA_600_500, N_ggA_500_400)
    
    model_length = len(DL_model_2)
    
    # Extract all X data for prediction
    DL_all_2 = DL_model_2_data_norm[:][:,0:model_length]
    
    # Define the learning rate
    DL_lr_2 = 5E-5
    DL_epoch_2 = 300
    DL_batch_2 = 10
    
    input_node = int(len(DL_X_train_2[0])*(4/3))
    mid_node = int(len(DL_X_train_2[0])*2*(4/3))
    

    ### The model ###
    DL_pred_2, DL_y_binary_2 = m.ML(DL_X_train_2, DL_y_train_2, DL_y_binary_2, DL_all_2, model_length, 
                 forceFit=ForceDL2, close=True, type_tag = ['DL', '500_400'],
                 
                 # Epochs batch and lr
                 epochs = DL_epoch_2, batch = DL_batch_2, lr = DL_lr_2,
                 
                 # Early stopping
                 doES=True, ESpat=30,
                 
                 # Learning rate reduction
                 doRL=False, RLrate=0.5, RLpat=8,
                 
                 # Nodes
                 input_node = input_node, mid_node = mid_node
                 )

    f.ROC_Curve(DL_y_binary_2, DL_pred_2, close=True, 
                title=('DL_500_400'), saveas=('DL/500_400/'))
    
    f.ProbHist(DL_y_binary_2, DL_pred_2, DL_N_train_2, 21, weight, N_arr, close=True, 
             label=['ttZ','ggA_500_400'], xtitle="Probability", ytitle="Events", 
                title=('DL_500_400_'), saveas=('DL/500_400/'))
    
    # Run optimisation if enabled
    if DL_Opt_2:
        
        # input_node = np.array([input_node*(1/2), input_node*(2/3), input_node, input_node*(4/3), input_node*(3/2)])
        # mid_node = np.array([mid_node*(1/2), mid_node*(2/3), mid_node, mid_node*(4/3), mid_node*(3/2)])
        
        input_node = np.array([input_node*(4/3)])
        mid_node = np.array([mid_node*(4/3), mid_node*(3/2)])
        
        DL_epoch_2 = np.array([300])
        DL_batch_2 = np.array([10,20])
        DL_lr_2 = np.array([1E-3, 1E-4, 5E-4, 1E-5])
        
        m.ML_opt(DL_X_train_2, DL_y_train_2, DL_y_binary_2, DL_all_2, model_length, DL_N_train_2, N_arr, weight, 
           DL_epoch_2, DL_batch_2, DL_lr_2, 
           doES=True, ESpat=35,
           doRL=False, RLrate=0, RLpat=0,
           input_node=input_node, mid_node=mid_node,
           
           type_tag = ['DL', '500_400'])
    
    ##########################
    # !!!!!!!!!!!!!!!!!!!!!! #
    # DL 3 is 600_500 repeat #
    ##########################
    
    # Prepare data for DL (using cut percentage)
    (DL_model_3_data, DL_model_3_data_norm, DL_X_train_3, DL_y_train_3, DL_X_test_3, 
     DL_y_test_3, DL_y_binary_3, DL_N_train_3) = f.data_prep(DL_model_2, N_ttZ, N, N_ggA_600_500, N_ggA_500_400,
                                                                  cut_percentage = 0.5)
    
    model_length = len(DL_model_2)
    
    # Extract all X data for prediction
    DL_all_3 = DL_model_3_data_norm[:][:,0:model_length]
    
    # Define the learning rate
    DL_lr_3 = 5E-5
    DL_epoch_3 = 300
    DL_batch_3 = 10
    
    input_node = int(len(DL_X_train_3[0])*(4/3))
    mid_node = int(len(DL_X_train_3[0])*2*(4/3))

    ### The model ###
    DL_pred_3, DL_y_binary_3 = m.ML(DL_X_train_3, DL_y_train_3, DL_y_binary_3, DL_all_3, model_length, 
                  forceFit=ForceDL3, close=True, type_tag = ['DL', '500_400_Cut1'],

    
                  # Epochs batch and lr
                  epochs = DL_epoch_3, batch = DL_batch_3, lr = DL_lr_3,
                 
                  # Early stopping
                  doES=True, ESpat=35,
                 
                  # Learning rate reduction
                  doRL=False, RLrate=0.1, RLpat=20,
                 
                  # Nodes
                  input_node = input_node, mid_node = mid_node
                  )

    f.ROC_Curve(DL_y_binary_3, DL_pred_3, close=True, 
                title=('DL_500_400_Cut_1_'), saveas=('DL/500_400_Cut1/'))
    
    f.ProbHist(DL_y_binary_3, DL_pred_3, DL_N_train_3, 21, weight, N_arr, close=True, 
              label=['ttZ','ggA_500_400'], xtitle="Probability", ytitle="Events", 
                title=('DL_500_400_Cut_1_'), saveas=('DL/500_400_Cut1/'))

######################
#  LIMIT ESTIMATION  #
######################

#svm_sig_count = svm_sig_count*(300/139)
#svm_bkg_count = svm_bkg_count*(300/139)

# if doSVM:
#     limit_svm = f.getLimit(svm_sig_count, svm_bkg_count, confidenceLevel=0.95, method=0, err=0.05)
#     print('SVM limit is: ', limit_svm)

delta_m_sig_count = delta_m_sig_count*(300/139)
delta_m_bkg_count = delta_m_bkg_count*(300/139)



limit_delta_m = f.getLimit(delta_m_sig_count, delta_m_bkg_count, confidenceLevel=0.95, method=0, err=0.05)
print('delta m limit is: ', limit_delta_m)

if doDL:
    # Adjust to 300 fb^-1
    #!!! Needs to be weighted in future
    DL_1_sig_count = DL_1_sig_count*(300/139)
    DL_1_bkg_count = DL_1_bkg_count*(300/139)
    
    limit_DL_1 = f.getLimit(DL_1_sig_count, DL_1_bkg_count, confidenceLevel=0.95, method=0, err=0.05)
    print('DL 600_500 limit is: ', limit_DL_1)





# !!! DL limit needs fixing, weights in the function is broken

# DL_bkg_count, DL_sig_count = f.LimitCount(DL_y_binary_1, DL_pred_1, 21, weight, N_arr, label=['ttZ','ggA_600_500'], xlim=[0,1])

# limit_DL = f.getLimit(DL_sig_count, DL_bkg_count, confidenceLevel=0.95, method=0, err=0.05)
# print('DL limit is: ', limit_DL)


###############
### Runtime ###
###############

print('Runtime: {:.2f} seconds'.format(time.time() - start_time))
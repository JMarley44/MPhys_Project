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

import sys
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

forceCalc = False

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
    N_ggA_500_400_1 = N_arr[9]
    N = N_arr[10]

    Z_inv_mass = np.load('Arrays/Z_inv_mass.npy')
    
    # Leptons
    lep1_four_mom = np.load('Arrays/lep1_four_mom.npy', allow_pickle=True)
    lep2_four_mom = np.load('Arrays/lep2_four_mom.npy', allow_pickle=True)
    lep3_four_mom = np.load('Arrays/lep3_four_mom.npy', allow_pickle=True)
    
    lep1_pt = np.load('Arrays/lep1_pt.npy', allow_pickle=True)
    lep2_pt = np.load('Arrays/lep2_pt.npy', allow_pickle=True)
    lep3_pt = np.load('Arrays/lep3_pt.npy', allow_pickle=True)
    
    lep1_eta = np.load('Arrays/lep1_eta.npy', allow_pickle=True)
    lep2_eta = np.load('Arrays/lep2_eta.npy', allow_pickle=True)
    lep3_eta = np.load('Arrays/lep3_eta.npy', allow_pickle=True)
    
    lep1_phi = np.load('Arrays/lep1_phi.npy', allow_pickle=True)
    lep2_phi = np.load('Arrays/lep2_phi.npy', allow_pickle=True)
    lep3_phi = np.load('Arrays/lep3_phi.npy', allow_pickle=True)
    
    lep_dphi = np.load('Arrays/lep_dphi.npy', allow_pickle=True)
    lep_dr = np.load('Arrays/lep_dr.npy', allow_pickle=True)

    # Jets
    jet1_four_mom = np.load('Arrays/jet1_four_mom.npy', allow_pickle=True)
    jet2_four_mom = np.load('Arrays/jet2_four_mom.npy', allow_pickle=True)

    jet1_pt = np.load('Arrays/jet1_pt.npy', allow_pickle=True)
    jet2_pt = np.load('Arrays/jet2_pt.npy', allow_pickle=True)
    
    jet1_eta = np.load('Arrays/jet1_eta.npy', allow_pickle=True)
    jet2_eta = np.load('Arrays/jet2_eta.npy', allow_pickle=True)
    
    jet1_phi = np.load('Arrays/jet1_phi.npy', allow_pickle=True)
    jet2_phi = np.load('Arrays/jet2_phi.npy', allow_pickle=True)
    
    jet_dphi = np.load('Arrays/jet_dphi.npy', allow_pickle=True)
    jet_dr = np.load('Arrays/jet_dr.npy', allow_pickle=True)
    
    # bJets
    bjet1_four_mom = np.load('Arrays/bjet1_four_mom.npy', allow_pickle=True)
    bjet2_four_mom = np.load('Arrays/bjet2_four_mom.npy', allow_pickle=True)
    
    bjet1_pt = np.load('Arrays/bjet1_pt.npy', allow_pickle=True)
    bjet2_pt = np.load('Arrays/bjet2_pt.npy', allow_pickle=True)
    
    bjet1_eta = np.load('Arrays/bjet1_eta.npy', allow_pickle=True)
    bjet2_eta = np.load('Arrays/bjet2_eta.npy', allow_pickle=True)
    
    bjet1_phi = np.load('Arrays/bjet1_phi.npy', allow_pickle=True)
    bjet2_phi = np.load('Arrays/bjet2_phi.npy', allow_pickle=True)
    
    # tops
    top1_four_mom = np.load('Arrays/top1_four_mom.npy', allow_pickle=True)
    top2_four_mom = np.load('Arrays/top2_four_mom.npy', allow_pickle=True)

    top1_pt = np.load('Arrays/top1_pt.npy', allow_pickle=True)
    top2_pt = np.load('Arrays/top2_pt.npy', allow_pickle=True)
    
    top1_eta = np.load('Arrays/top1_eta.npy', allow_pickle=True)
    top2_eta = np.load('Arrays/top2_eta.npy', allow_pickle=True)
    
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
    met_phi = np.load('Arrays/met_phi.npy', allow_pickle=True)
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
    main_data = pd.DataFrame(dataimport).to_numpy()
    
    path1 = "C:/Users/James/Documents/Liverpool/Year 4/PHYS498 - Project/Python files/MPhys_Project/events_AZH_500_400_139ifb.csv"
    extra_sig_import = pd.read_csv(path1, header=None)
    extra_sig_data = pd.DataFrame(extra_sig_import).to_numpy()
    
    path2 = "C:/Users/James/Documents/Liverpool/Year 4/PHYS498 - Project/Python files/MPhys_Project/events_ttZ_139ifb.csv"
    extra_bkg_import = pd.read_csv(path2, header=None)
    extra_bkg_data = pd.DataFrame(extra_bkg_import).to_numpy()
    
    # Add extra signal to the end
    dataset_full = np.vstack((main_data,extra_sig_data))
    
    # Add extra bkg to the start
    dataset_full = np.vstack((extra_bkg_data,dataset_full))
    
    # Delete the column titles
    dataset = np.delete(dataset_full,0,1)
    
    # Find the length of the states
    N_ttZ,N_ttWm,N_ttWp,N_ggA_460_360,N_ggA_500_360,N_ggA_600_360,N_ggA_600_400,N_ggA_600_500,N_ggA_500_400,N_ggA_500_400_1  = d.DataSplit(dataset_full)
    
    # Length of the dataset
    N = len(dataset)
    
    #####################
    #  INPUT VARIABLES  #
    #####################
    
    W_mass = 80.38 # GeV
    Z_mass = 91.12 # GeV
    top_mass = 172.76 # GeV
    
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
    
    #!!! Do I need this?
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
    Wm_lep_four_mom = np.zeros((N,4))
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
    
    N_ttZ_test = N_ttZ
    N_ttWm_test = N_ttWm
    N_ttWp_test = N_ttWp
    N_ggA_460_360_test = N_ggA_460_360
    N_ggA_500_360_test = N_ggA_500_360
    N_ggA_600_360_test = N_ggA_600_360
    N_ggA_600_400_test = N_ggA_600_400
    N_ggA_600_500_test = N_ggA_600_500
    N_ggA_500_400_test = N_ggA_500_400
    N_ggA_500_400_1_test = N_ggA_500_400_1
    
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
    
    'Counters'
    j_count = 0
    z_count = 0
    print_count = 0
    
    ###############
    #  MAIN LOOP  #
    ###############
    
    print('Executing main loop.', end='')
    
    for i in range(N):
        
        if (print_count % 5000==0):
            print('.', end='')
        
        print_count += 1
        
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
        # Also included here is the lepton dr (angle between Z leptons)
        
        # Calculate test mass differences
        Zm_diff_12 = np.sqrt((lep12_inv_mass - Z_mass)**2)
        Zm_diff_13 = np.sqrt((lep13_inv_mass - Z_mass)**2)
        Zm_diff_23 = np.sqrt((lep23_inv_mass - Z_mass)**2)
        
        # Initialise viability bools
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
                        z_count = z_count +1
                        raise Exception
                    Z_four_mom[i] = lep12_four_mom[i]
                    lep_dphi[i] = f.dphi(lep1_phi[i], lep2_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep1_eta[i], lep2_eta[i])
                    Wm_lep_four_mom[i] = lep3_four_mom[i]
                elif Zm_diff_13[i] < Zm_diff_23[i]:
                    # 13 is the answer
                    if Zm_diff_13[i]>15:
                        z_count = z_count +1
                        raise Exception
                    Z_four_mom[i] = lep13_four_mom[i]
                    lep_dphi[i] = f.dphi(lep1_phi[i], lep3_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep1_eta[i], lep3_eta[i])
                    Wm_lep_four_mom[i] = lep2_four_mom[i]
                else:
                    # 23 is the answer
                    if Zm_diff_23[i]>15:
                        z_count = z_count +1
                        raise Exception
                    Z_four_mom[i] = lep23_four_mom[i]
                    lep_dphi[i] = f.dphi(lep2_phi[i], lep3_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep2_eta[i], lep3_eta[i])
                    Wm_lep_four_mom[i] = lep1_four_mom[i]
            
            # if only 2 are viable check which one is best
            elif lep12_viable and lep13_viable:
                if Zm_diff_12[i] < Zm_diff_13[i]:
                    # 12 is the answer
                    if Zm_diff_12[i]>15:
                        z_count = z_count +1
                        raise Exception
                    Z_four_mom[i] = lep12_four_mom[i]
                    lep_dphi[i] = f.dphi(lep1_phi[i], lep2_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep1_eta[i], lep2_eta[i])
                    Wm_lep_four_mom[i] = lep3_four_mom[i]
                else:
                    # 13 is the answer
                    if Zm_diff_13[i]>15:
                        z_count = z_count +1
                        raise Exception
                    Z_four_mom[i] = lep13_four_mom[i]
                    lep_dphi[i] = f.dphi(lep1_phi[i], lep3_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep1_eta[i], lep3_eta[i])
                    Wm_lep_four_mom[i] = lep2_four_mom[i]
                
            elif lep12_viable and lep23_viable:
                if Zm_diff_12[i] < Zm_diff_23[i]:
                    # 12 is the answer
                    if Zm_diff_12[i]>15:
                        z_count = z_count +1
                        raise Exception
                    Z_four_mom[i] = lep12_four_mom[i]
                    lep_dphi[i] = f.dphi(lep1_phi[i], lep2_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep1_eta[i], lep2_eta[i])
                    Wm_lep_four_mom[i] = lep3_four_mom[i]
                else:
                    # 23 is the answer
                    if Zm_diff_23[i]>15:
                        z_count = z_count +1
                        raise Exception
                    Z_four_mom[i] = lep23_four_mom[i]
                    lep_dphi[i] = f.dphi(lep2_phi[i], lep3_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep2_eta[i], lep3_eta[i])
                    Wm_lep_four_mom[i] = lep1_four_mom[i]
                
            elif lep13_viable and lep23_viable:
                if Zm_diff_13[i] < Zm_diff_23[i]:
                    # 13 is the answer
                    if Zm_diff_13[i]>15:
                        z_count = z_count +1
                        raise Exception
                    Z_four_mom[i] = lep13_four_mom[i]
                    lep_dphi[i] = f.dphi(lep1_phi[i], lep3_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep1_eta[i], lep3_eta[i])
                    Wm_lep_four_mom[i] = lep2_four_mom[i]
                else:
                    # 23 is the answer
                    if Zm_diff_23[i]>15:
                        z_count = z_count +1
                        raise Exception
                    Z_four_mom[i] = lep23_four_mom[i]
                    lep_dphi[i] = f.dphi(lep2_phi[i], lep3_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep2_eta[i], lep3_eta[i])
                    Wm_lep_four_mom[i] = lep1_four_mom[i]
                
            # if only one path is viable choose it
            elif lep12_viable:
                # 12 is the answer
                if Zm_diff_12[i]>15:
                    z_count = z_count +1
                    raise Exception
                else:
                    Z_four_mom[i] = lep12_four_mom[i]
                    lep_dphi[i] = f.dphi(lep1_phi[i], lep2_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep1_eta[i], lep2_eta[i])
                    Wm_lep_four_mom[i] = lep3_four_mom[i]
    
            elif lep13_viable:
                # 13 is the answer
                if Zm_diff_13[i]>15:
                    z_count = z_count +1
                    raise Exception
                else:
                    Z_four_mom[i] = lep13_four_mom[i]
                    lep_dphi[i] = f.dphi(lep1_phi[i], lep3_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep1_eta[i], lep3_eta[i])
                    Wm_lep_four_mom[i] = lep2_four_mom[i]
            
            elif lep23_viable:
                # 23 is the answer
                if Zm_diff_23[i]>15:
                    z_count = z_count +1
                    raise Exception
                else:
                    Z_four_mom[i] = lep23_four_mom[i]
                    lep_dphi[i] = f.dphi(lep2_phi[i], lep3_phi[i])
                    lep_dr[i] = f.drangle(lep_dphi[i], lep2_eta[i], lep3_eta[i])
                    Wm_lep_four_mom[i] = lep1_four_mom[i]
                    
            # If nothing is viable skip to except catch
            else:
                z_count = z_count +1
                raise Exception
                
            # Test the pseudorapidity
            if abs(lep1_eta[i])>2.5:
                j_count = j_count+1
                raise Exception
                
            elif abs(lep2_eta[i])>2.5:
                j_count = j_count+1
                raise Exception
                
            elif abs(lep3_eta[i])>2.5:
                j_count = j_count+1
                raise Exception

            elif abs(jet1_eta[i])>2.5:
                j_count = j_count+1
                raise Exception

            elif abs(jet2_eta[i])>2.5:
                j_count = j_count+1
                raise Exception

            elif abs(bjet1_eta[i])>2.5:
                j_count = j_count+1
                raise Exception

            elif abs(bjet2_eta[i])>2.5:
                j_count = j_count+1
                raise Exception

            
        except:
            # If there's no viable decay mark the event for deletion later
            Z_delete[i] = 1
            
            # Shift each N value downwards if one below it is removed
            if i <= N_ttZ_test:
                N_ttZ = N_ttZ - 1
            if i <= N_ttWm_test:
                N_ttWm = N_ttWm - 1
            if i <= N_ttWp_test:
                N_ttWp = N_ttWp - 1
            if i <= N_ggA_460_360_test:
                N_ggA_460_360 = N_ggA_460_360 - 1
            if i <= N_ggA_500_360_test:
                N_ggA_500_360 = N_ggA_500_360 - 1
            if i <= N_ggA_600_360_test:
                N_ggA_600_360 = N_ggA_600_360 - 1
            if i <= N_ggA_600_400_test:
                N_ggA_600_400 = N_ggA_600_400 - 1
            if i <= N_ggA_600_500_test:
                N_ggA_600_500 = N_ggA_600_500 - 1
            if i <= N_ggA_500_400_test:
                N_ggA_500_400 = N_ggA_500_400 - 1
            if i <= N_ggA_500_400_1_test:
                N_ggA_500_400_1 = N_ggA_500_400_1 - 1
            if i >= N_ggA_500_400_1_test:
                N_ggA_500_400_1 = N_ggA_500_400_1 - 1
                
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
    
        lep_sq[i,:] = Wm_lep_four_mom[i,:]*Wm_lep_four_mom[i,:]
        lep_mag[i] = lep_sq[i,0]-(lep_sq[i,1]+lep_sq[i,2]+lep_sq[i,3])
        
        k[i] = ((W_mass*W_mass - lep_mag[i])/2) + ((Wm_lep_four_mom[i,1]*met_px [i]) + (Wm_lep_four_mom[i,2] * met_py [i]))
        
        a[i] =  (Wm_lep_four_mom[i,0]*Wm_lep_four_mom[i,0]) - (Wm_lep_four_mom[i,3]*Wm_lep_four_mom[i,3])
        b[i] = -2*k[i]*Wm_lep_four_mom[i,3]
        c[i] =  (Wm_lep_four_mom[i,0]*Wm_lep_four_mom[i,0] *  met_mag[i])  -  (k[i]*k[i])
        
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
    
        Wm_four_mom [i,0] = Wm_lep_four_mom[i,0]+f.mom(met_px[i], met_py[i], (neu_four_mom[i,3]))
        for j in range(1,4):
            Wm_four_mom [i,j] = Wm_lep_four_mom[i,j]+neu_four_mom[i,j]
            
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
        ttZ_four_mom[i] = tt_four_mom[i] + Z_four_mom[i]
        
        # Calculation of the masses and delta m
        M_tt [i] = f.inv_mass(tt_four_mom[i])
        M_ttZ[i] = f.inv_mass(ttZ_four_mom[i])
        delta_m[i] = M_ttZ[i] - M_tt[i]
        
        'SVM calculations'
        neu_four_mom[i,0] = (f.mom(neu_four_mom[i,1],neu_four_mom[i,2],neu_four_mom[i,3]))**2

    ####################
    # End of Main loop #
    ####################
    
    print(' - Done!')
    print(j_count + z_count,' non-viable decays removed\n')

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
                      N_ggA_600_400,N_ggA_600_500,N_ggA_500_400,N_ggA_500_400_1,N])
    
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
    
    np.save('Arrays/lep1_eta.npy', lep1_eta)     
    np.save('Arrays/lep2_eta.npy', lep2_eta)     
    np.save('Arrays/lep3_eta.npy', lep3_eta)
    
    np.save('Arrays/lep1_phi.npy', lep1_phi)     
    np.save('Arrays/lep2_phi.npy', lep2_phi)     
    np.save('Arrays/lep3_phi.npy', lep3_phi) 
    
    np.save('Arrays/lep_dphi.npy', lep_dphi) # Maybe move this to SVM bunch
    np.save('Arrays/lep_dr.npy', lep_dr)
    
    # Jets
    np.save('Arrays/jet1_four_mom.npy', jet1_four_mom)     
    np.save('Arrays/jet2_four_mom.npy', jet2_four_mom)
    
    np.save('Arrays/jet1_pt.npy', jet1_pt)     
    np.save('Arrays/jet2_pt.npy', jet2_pt)

    np.save('Arrays/jet1_eta.npy', jet1_eta)     
    np.save('Arrays/jet2_eta.npy', jet2_eta)

    np.save('Arrays/jet1_phi.npy', jet1_phi)     
    np.save('Arrays/jet2_phi.npy', jet2_phi)        
    
    np.save('Arrays/jet_dphi.npy', jet_dphi) # Maybe move this to SVM bunch
    np.save('Arrays/jet_dr.npy', jet_dr)
    
    # bJets
    
    np.save('Arrays/bjet1_four_mom.npy', bjet1_four_mom)     
    np.save('Arrays/bjet2_four_mom.npy', bjet2_four_mom)
    
    np.save('Arrays/bjet1_pt.npy', bjet1_pt)     
    np.save('Arrays/bjet2_pt.npy', bjet2_pt)
    
    np.save('Arrays/bjet1_eta.npy', bjet1_eta)     
    np.save('Arrays/bjet2_eta.npy', bjet2_eta)
    
    np.save('Arrays/bjet1_phi.npy', bjet1_phi)     
    np.save('Arrays/bjet2_phi.npy', bjet2_phi)
    
    # tops
    
    np.save('Arrays/top1_four_mom.npy', top1_four_mom)     
    np.save('Arrays/top2_four_mom.npy', top2_four_mom)
    
    np.save('Arrays/top1_pt.npy', top1_pt)     
    np.save('Arrays/top2_pt.npy', top2_pt)
    
    np.save('Arrays/top1_eta.npy', top1_eta)     
    np.save('Arrays/top2_eta.npy', top2_eta)
    
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
    np.save('Arrays/met_phi.npy', met_phi)  
    np.save('Arrays/Wm_mass.npy', Wm_mass)     
    np.save('Arrays/Wp_mass.npy', Wp_mass)       
    np.save('Arrays/neu_four_mom.npy', neu_four_mom)        
    np.save('Arrays/weight.npy', weight)    

###########
#  PLOTS  #
###########
doPlots = False

if doPlots:
    
    print('Plotting plots.', end='')
    
    ############
    # Singular #
    ############
    
    # Z invariant mass
    f.Hist(Z_inv_mass, weight, 16, close=True, N_ttZ=N_ttZ, xtitle="$m_{Z}$ (GeV)", ytitle="Events", 
           title="Z boson invariant mass", xmin = 75, xmax = 105)
    
    f.Hist(jet1_eta, weight, 21, close=True, N_ttZ=N_ttZ, xtitle="Pseudorapidity eta", ytitle="Events", 
            title="jet_1_eta")
    
    f.Hist(jet2_eta, weight, 21, close=True, N_ttZ=N_ttZ, xtitle="Pseudorapidity eta", ytitle="Events", 
            title="jet_2_eta")

    ###########
    # Signals #
    ###########
    
    # All signals for the limit calculation
    all_signals = ['ttZ', 'other','460_360', '500_360', '600_360', '600_400', '600_500', '500_400','500_400_1']
    
    # Selected signals, choose one
    select_signals = ['ttZ', 'other', '460_360', '500_360', '600_360']
    #select_signals = ['ttZ', 'other', '600_400', '500_400_1', '600_500']
    
    all_line = ['-',':','--','--','-',':','-']
    select_line = ['--',':','--','--','-','--',':']
    
    ##########
    # Events #
    ##########
    
    # Input variables
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
    f.SignalHist(delta_m, weight, 26, N_arr, close=True, xtitle="$\Delta$m (GeV)", ytitle="Events", title="$\Delta$m",
                 saveas="delta_m", signals = select_signals, line = select_line, scale='log')
    
    ##########
    # Limits #
    ##########
    
    # delta m
    delta_m_bkg_count, delta_m_sig_count = f.SignalHist(delta_m, weight, 26, N_arr, close=True, xtitle="$\Delta$m (GeV)", ytitle="Events", title="$\Delta$m",
                 saveas="delta_m", signals = all_signals, line = all_line, xlim=[0, 2000], addText = 'Weighted to luminosity', limit=True)
    
    # Z pt
    Z_pt_bkg_count, Z_pt_sig_count = f.SignalHist(delta_m, weight, 25, N_arr, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="$Z_{pT}$",
                 saveas="Z_pT", signals = all_signals, line = all_line, addText = 'Weighted to luminosity', limit=True)
    
    print('.', end='')
    
    ### Save the limit counts for when plots are disabled
    np.save('Arrays/delta_m_limit_bkg.npy', delta_m_bkg_count)
    np.save('Arrays/delta_m_limit_sig.npy', delta_m_sig_count) 
    
    np.save('Arrays/Z_limit_bkg.npy', Z_pt_bkg_count)
    np.save('Arrays/Z_limit_sig.npy', Z_pt_sig_count) 
    
    ##################
    # Shallow Shapes #
    ##################
    
    # delta m
    f.SignalHist(delta_m, weight, 26, N_arr, close=True, xtitle="$\Delta$m (GeV)", ytitle="Events", title="$\Delta$m",
                 saveas="Shallow/delta_m", signals = select_signals, 
                 shaped=True, line = select_line, xlim=[0, 1000])
    
    # met pt
    f.SignalHist(met_pt, weight, 25, N_arr, close=True, xtitle=r'$E^{T}_{miss}$ (GeV)', ytitle="Events", title=r'$E^{T}_{miss}$',
                  saveas="Shallow/met_pt", signals = select_signals, 
                  shaped = True, line = select_line, xlim=[0, 600])
    
    # met phi
    f.SignalHist(met_phi, weight, 25, N_arr, close=True, xtitle=r'$E^{T}_{miss}$ $\phi$ (rad)', ytitle="rad", title=r'$E^{T}_{miss}$ $\phi$',
                  saveas="Shallow/met_phi", signals = select_signals, 
                  shaped = True, line = select_line)

    # lep dr
    f.SignalHist(lep_dr, weight, 26, N_arr, close=True, xtitle=r'lep angle r (rad)', ytitle="rad", title=r'lepton ${\Delta}R$',
                  saveas="Shallow/lep_dr", signals = select_signals, 
                  shaped = True, line = select_line)
    
    # Neutrino pz
    f.SignalHist(neu_four_mom[:,3], weight, 26, N_arr, close=True, xtitle=r'Neutrino $p_Z$ (GeV)', ytitle="Events", title=r'Neutrino $p_Z$',
                 saveas="Shallow/Neutrino_pz", signals = select_signals, 
                 shaped = True, line = select_line, xlim=[-500, 500])
    
    # Z pt
    f.SignalHist(Z_pt, weight, 25, N_arr, close=True, xtitle=r'$p_T$ (GeV)', ytitle="Events", title=r'$Z_{p_T}$',
                 saveas="Shallow/Z_pT", signals = select_signals, 
                 shaped = True, line = select_line, xlim=[0, 600])
    
    # Z pz
    f.SignalHist(Z_pz, weight, 25, N_arr, close=True, xtitle=r'$p_z$ (GeV)', ytitle="Events", title=r'$Z_{p_z}$',
                 saveas="Shallow/Z_pz", signals = select_signals, 
                 shaped = True, line = select_line, xlim=[-1000, 1000])
    
    # M_ttZ
    f.SignalHist(M_ttZ, weight, 26, N_arr, close=True, xtitle=r'ttZ mass (GeV)', ytitle="Events", title=r'ttZ system mass',
                 saveas="Shallow/ttZ_m", signals = select_signals, 
                 shaped = True, line = select_line, xlim=[0, 3000])
    
    # M_ttZ
    f.SignalHist(M_tt, weight, 26, N_arr, close=True, xtitle=r'tt mass (GeV)', ytitle="Events", title=r'tt system mass',
                 saveas="Shallow/tt_m", signals = select_signals, 
                 shaped = True, line = select_line, xlim=[0, 3000])
    
    # top1-2 pt
    f.SignalHist(top12_pt, weight, 26, N_arr, close=True, xtitle=r'$p_T$ (GeV)', ytitle="Events", title=r'top12 $p_T$',
                 saveas="Shallow/top12_pt", signals = select_signals, 
                 shaped = True, line = select_line, xlim=[0, 1200])
    
    # top dphi
    f.SignalHist(top_dphi, weight, 26, N_arr, close=True, xtitle=r't$\bar{t}$ angle \phi (rad)', ytitle="rad", title=r'top quarks $\Delta\phi$',
                  saveas="Shallow/top_dphi", signals = select_signals, 
                  shaped = True, line = select_line)
    
    # top dr
    f.SignalHist(top_dr, weight, 26, N_arr, close=True, xtitle=r't$\bar{t}$ angle r (rad)', ytitle="rad", title=r'top quarks ${\Delta}R$',
                  saveas="Shallow/top_dr", signals = select_signals, 
                  shaped = True, line = select_line)
    
    # jet dphi
    f.SignalHist(jet_dphi, weight, 26, N_arr, close=True, xtitle=r'jets angle \phi (rad)', ytitle="rad", title=r'jet $\Delta\phi$',
                  saveas="Shallow/jet_dphi", signals = select_signals, 
                  shaped = True, line = select_line)
    
    # jet dr
    f.SignalHist(jet_dr, weight, 26, N_arr, close=True, xtitle=r'jet angle r (rad)', ytitle="rad", title=r'jet ${\Delta}R$',
                  saveas="Shallow/jet_dr", signals = select_signals, 
                  shaped = True, line = select_line)
    
    # H pt
    f.SignalHist(H_pt, weight, 26, N_arr, close=True, xtitle=r'$p_T$ (GeV)', ytitle="Events", title=r'$H_{pT}$',
                 saveas="Shallow/H_pT", signals = select_signals, 
                 shaped = True, line = select_line, xlim=[0, 600])

    print('.', end='')
    
    ###############
    # Deep shapes #
    ###############
    
    ### Leps
    
    # lep pt
    f.SignalHist(lep1_pt, weight, 25, N_arr, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="lep 1 $p_T$",
                 saveas="Deep/lep1/pt", signals = select_signals, line = select_line, shaped=True, xlim=[0, 500])
    
    f.SignalHist(lep2_pt, weight, 25, N_arr, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="lep 2 $p_T$",
                 saveas="Deep/lep2/pt", signals = select_signals, line = select_line, shaped=True, xlim=[0, 250])
    
    f.SignalHist(lep3_pt, weight, 25, N_arr, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="lep 3 $p_T$",
                 saveas="Deep/lep3/pt", signals = select_signals, line = select_line, shaped=True, xlim=[0, 250])
    
    # lep E
    f.SignalHist(lep1_four_mom[:,0], weight, 25, N_arr, close=True, xtitle="E (GeV)", ytitle="Events", title="lep 1 E",
                 saveas="Deep/lep1/E", signals = select_signals, line = select_line, shaped=True)
    
    f.SignalHist(lep2_four_mom[:,0], weight, 25, N_arr, close=True, xtitle="E (GeV)", ytitle="Events", title="lep 2 E",
                 saveas="Deep/lep2/E", signals = select_signals, line = select_line, shaped=True)
    
    f.SignalHist(lep3_four_mom[:,0], weight, 25, N_arr, close=True, xtitle="E (GeV)", ytitle="Events", title="lep 3 E",
                 saveas="Deep/lep3/E", signals = select_signals, line = select_line, shaped=True)
    
    # lep px
    f.SignalHist(lep1_four_mom[:,1], weight, 25, N_arr, close=True, xtitle="$p_x$ (GeV)", ytitle="Events", title="lep 1 $p_x$",
                 saveas="Deep/lep1/px", signals = select_signals, line = select_line, shaped=True)
    
    f.SignalHist(lep2_four_mom[:,1], weight, 25, N_arr, close=True, xtitle="$p_x$ (GeV)", ytitle="Events", title="lep 2 $p_x$",
                 saveas="Deep/lep2/px", signals = select_signals, line = select_line, shaped=True)
    
    f.SignalHist(lep3_four_mom[:,1], weight, 25, N_arr, close=True, xtitle="$p_x$ (GeV)", ytitle="Events", title="lep 3 $p_x$",
                 saveas="Deep/lep3/px", signals = select_signals, line = select_line, shaped=True)
    
    # lep py
    f.SignalHist(lep1_four_mom[:,2], weight, 25, N_arr, close=True, xtitle="$p_y$ (GeV)", ytitle="Events", title="lep 1 $p_y$",
                 saveas="Deep/lep1/py", signals = select_signals, line = select_line, shaped=True)
    
    f.SignalHist(lep2_four_mom[:,2], weight, 25, N_arr, close=True, xtitle="$p_y$ (GeV)", ytitle="Events", title="lep 2 $p_y$",
                 saveas="Deep/lep2/py", signals = select_signals, line = select_line, shaped=True)
    
    f.SignalHist(lep3_four_mom[:,2], weight, 25, N_arr, close=True, xtitle="$p_y$ (GeV)", ytitle="Events", title="lep 3 $p_y$",
                 saveas="Deep/lep3/py", signals = select_signals, line = select_line, shaped=True)
    
    # lep pz
    f.SignalHist(lep1_four_mom[:,3], weight, 25, N_arr, close=True, xtitle="$p_z$ (GeV)", ytitle="Events", title="lep 1 $p_z$",
                 saveas="Deep/lep1/pz", signals = select_signals, line = select_line, shaped=True)
    
    f.SignalHist(lep2_four_mom[:,3], weight, 25, N_arr, close=True, xtitle="$p_z$ (GeV)", ytitle="Events", title="lep 2 $p_z$",
                 saveas="Deep/lep2/pz", signals = select_signals, line = select_line, shaped=True)
    
    f.SignalHist(lep3_four_mom[:,3], weight, 25, N_arr, close=True, xtitle="$p_z$ (GeV)", ytitle="Events", title="lep 3 $p_z$",
                 saveas="Deep/lep3/pz", signals = select_signals, line = select_line, shaped=True)
    
    # lep eta
    f.SignalHist(lep1_eta, weight, 25, N_arr, close=True, xtitle="pseudorapidity", ytitle="Events", title="lep 1 eta",
                 saveas="Deep/lep1/eta", signals = select_signals, line = select_line, shaped=True)
    
    f.SignalHist(lep2_eta, weight, 25, N_arr, close=True, xtitle="pseudorapidity", ytitle="Events", title="lep 2 eta",
                 saveas="Deep/lep2/eta", signals = select_signals, line = select_line, shaped=True)
    
    f.SignalHist(lep3_eta, weight, 25, N_arr, close=True, xtitle="pseudorapidity", ytitle="Events", title="lep 3 eta",
                 saveas="Deep/lep3/eta", signals = select_signals, line = select_line, shaped=True)
    
    # lep phi
    f.SignalHist(lep1_phi, weight, 25, N_arr, close=True, xtitle="$\phi$ (rad)", ytitle="rad", title="lep 1 $\phi$",
                 saveas="Deep/lep1/phi", signals = select_signals, line = select_line, shaped=True)
    
    f.SignalHist(lep2_phi, weight, 25, N_arr, close=True, xtitle="$\phi$ (rad)", ytitle="rad", title="lep 2 $\phi$",
                 saveas="Deep/lep2/phi", signals = select_signals, line = select_line, shaped=True)
    
    f.SignalHist(lep3_phi, weight, 25, N_arr, close=True, xtitle="$\phi$ (rad)", ytitle="rad", title="lep 3 $\phi$",
                 saveas="Deep/lep3/phi", signals = select_signals, line = select_line, shaped=True)

    ### Jets

    # jet pt
    f.SignalHist(jet1_pt, weight, 25, N_arr, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="jet 1 $p_T$",
                 saveas="Deep/jet1/pt", signals = select_signals, line = select_line, shaped=True, xlim=[0, 500])
    f.SignalHist(jet2_pt, weight, 25, N_arr, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="jet 2 $p_T$",
                 saveas="Deep/jet2/pt", signals = select_signals, line = select_line, shaped=True, xlim=[0, 250])
    
    # jet px
    f.SignalHist(jet1_four_mom[:,1], weight, 25, N_arr, close=True, xtitle="$p_x$ (GeV)", ytitle="Events", title="jet 1 $p_x$",
                 saveas="Deep/jet1/px", signals = select_signals, line = select_line, shaped=True)
    f.SignalHist(jet2_four_mom[:,1], weight, 25, N_arr, close=True, xtitle="$p_x$ (GeV)", ytitle="Events", title="jet 2 $p_x$",
                 saveas="Deep/jet2/px", signals = select_signals, line = select_line, shaped=True)
    
    # jet py
    f.SignalHist(jet1_four_mom[:,2], weight, 25, N_arr, close=True, xtitle="$p_y$ (GeV)", ytitle="Events", title="jet 1 $p_y$",
                 saveas="Deep/jet1/py", signals = select_signals, line = select_line, shaped=True)
    f.SignalHist(jet2_four_mom[:,2], weight, 25, N_arr, close=True, xtitle="$p_y$ (GeV)", ytitle="Events", title="jet 2 $p_y$",
                 saveas="Deep/jet2/py", signals = select_signals, line = select_line, shaped=True)

    # jet pz
    f.SignalHist(jet1_four_mom[:,3], weight, 25, N_arr, close=True, xtitle="$p_z$ (GeV)", ytitle="Events", title="jet 1 $p_z$",
                 saveas="Deep/jet1/pz", signals = select_signals, line = select_line, shaped=True)
    f.SignalHist(jet2_four_mom[:,3], weight, 25, N_arr, close=True, xtitle="$p_z$ (GeV)", ytitle="Events", title="jet 2 $p_z$",
                 saveas="Deep/jet2/pz", signals = select_signals, line = select_line, shaped=True)
    
    # jet eta
    f.SignalHist(jet1_eta, weight, 25, N_arr, close=True, xtitle="pseudorapidity", ytitle="Events", title="jet 1 eta",
                 saveas="Deep/jet1/eta", signals = select_signals, line = select_line, shaped=True)
    f.SignalHist(jet2_eta, weight, 25, N_arr, close=True, xtitle="pseudorapidity", ytitle="Events", title="jet 2 eta",
                 saveas="Deep/jet2/eta", signals = select_signals, line = select_line, shaped=True)
    
    # jet phi
    f.SignalHist(jet1_phi, weight, 25, N_arr, close=True, xtitle="$\phi$ (rad)", ytitle="rad", title="jet 1 $\phi$",
                 saveas="Deep/jet1/phi", signals = select_signals, line = select_line, shaped=True)
    
    f.SignalHist(jet2_phi, weight, 25, N_arr, close=True, xtitle="$\phi$ (rad)", ytitle="rad", title="jet 2 $\phi$",
                 saveas="Deep/jet2/phi", signals = select_signals, line = select_line, shaped=True)
    
    ### bjets
    
    # bjet pt
    f.SignalHist(bjet1_pt, weight, 25, N_arr, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="bjet 1 $p_T$",
                 saveas="Deep/bjet1/pt", signals = select_signals, line = select_line, shaped=True)
    
    f.SignalHist(bjet2_pt, weight, 25, N_arr, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="bjet 2 $p_T$",
             saveas="Deep/bjet2/pt", signals = select_signals, line = select_line, shaped=True)
    
    # bjet px
    f.SignalHist(bjet1_four_mom[:,1], weight, 25, N_arr, close=True, xtitle="$p_x$ (GeV)", ytitle="Events", title="bjet 1 $p_x$",
                 saveas="Deep/bjet1/px", signals = select_signals, line = select_line, shaped=True)
    f.SignalHist(bjet2_four_mom[:,1], weight, 25, N_arr, close=True, xtitle="$p_x$ (GeV)", ytitle="Events", title="bjet 2 $p_x$",
                 saveas="Deep/bjet2/px", signals = select_signals, line = select_line, shaped=True)
    
    # bjet py
    f.SignalHist(bjet1_four_mom[:,2], weight, 25, N_arr, close=True, xtitle="$p_y$ (GeV)", ytitle="Events", title="bjet 1 $p_y$",
                 saveas="Deep/bjet1/py", signals = select_signals, line = select_line, shaped=True)
    f.SignalHist(bjet2_four_mom[:,2], weight, 25, N_arr, close=True, xtitle="$p_y$ (GeV)", ytitle="Events", title="bjet 2 $p_y$",
                 saveas="Deep/bjet2/py", signals = select_signals, line = select_line, shaped=True)

    # bjet pz
    f.SignalHist(bjet1_four_mom[:,3], weight, 25, N_arr, close=True, xtitle="$p_z$ (GeV)", ytitle="Events", title="bjet 1 $p_z$",
                 saveas="Deep/bjet1/pz", signals = select_signals, line = select_line, shaped=True)
    f.SignalHist(bjet2_four_mom[:,3], weight, 25, N_arr, close=True, xtitle="$p_z$ (GeV)", ytitle="Events", title="bjet 2 $p_z$",
                 saveas="Deep/bjet2/pz", signals = select_signals, line = select_line, shaped=True)
    
    # bjet eta
    f.SignalHist(bjet1_eta, weight, 25, N_arr, close=True, xtitle="pseudorapidity", ytitle="Events", title="bjet 1 eta",
                 saveas="Deep/bjet1/eta", signals = select_signals, line = select_line, shaped=True)
    f.SignalHist(bjet2_eta, weight, 25, N_arr, close=True, xtitle="pseudorapidity", ytitle="Events", title="bjet 2 eta",
                 saveas="Deep/bjet2/eta", signals = select_signals, line = select_line, shaped=True)
    
    # bjet phi
    f.SignalHist(bjet1_phi, weight, 25, N_arr, close=True, xtitle="$\phi$ (rad)", ytitle="rad", title="bjet 1 $\phi$",
                 saveas="Deep/bjet1/phi", signals = select_signals, line = select_line, shaped=True)
    
    f.SignalHist(bjet2_phi, weight, 25, N_arr, close=True, xtitle="$\phi$ (rad)", ytitle="rad", title="bjet 2 $\phi$",
                 saveas="Deep/bjet2/phi", signals = select_signals, line = select_line, shaped=True)
    
    ### Tops
    
    # top pt
    f.SignalHist(top1_pt, weight, 25, N_arr, close=True, xtitle=r'$p_T$ (GeV)', ytitle="Events", title=r'top1 $p_T$',
                 saveas="Deep/top1/pt", signals = select_signals, 
                 shaped = True, line = select_line, xlim=[0, 1000])
    f.SignalHist(top2_pt, weight, 26, N_arr, close=True, xtitle=r'$p_T$ (GeV)', ytitle="Events", title=r'top2 $p_T$',
                 saveas="Deep/top2/pt", signals = select_signals, 
                 shaped = True, line = select_line, xlim=[0, 800])
    
    # top px
    f.SignalHist(top1_four_mom[:,1], weight, 25, N_arr, close=True, xtitle="$p_x$ (GeV)", ytitle="Events", title="top 1 $p_x$",
                 saveas="Deep/top1/px", signals = select_signals, line = select_line, shaped=True)
    f.SignalHist(top2_four_mom[:,1], weight, 25, N_arr, close=True, xtitle="$p_x$ (GeV)", ytitle="Events", title="top 2 $p_x$",
                 saveas="Deep/top2/px", signals = select_signals, line = select_line, shaped=True)
    
    # top py
    f.SignalHist(top1_four_mom[:,2], weight, 25, N_arr, close=True, xtitle="$p_y$ (GeV)", ytitle="Events", title="top 1 $p_y$",
                 saveas="Deep/top1/py", signals = select_signals, line = select_line, shaped=True)
    f.SignalHist(top2_four_mom[:,2], weight, 25, N_arr, close=True, xtitle="$p_y$ (GeV)", ytitle="Events", title="top 2 $p_y$",
                 saveas="Deep/top2/py", signals = select_signals, line = select_line, shaped=True)

    # top pz
    f.SignalHist(top1_four_mom[:,3], weight, 25, N_arr, close=True, xtitle="$p_z$ (GeV)", ytitle="Events", title="top 1 $p_z$",
                 saveas="Deep/top1/pz", signals = select_signals, line = select_line, shaped=True)
    f.SignalHist(top2_four_mom[:,3], weight, 25, N_arr, close=True, xtitle="$p_z$ (GeV)", ytitle="Events", title="top 2 $p_z$",
                 saveas="Deep/top2/pz", signals = select_signals, line = select_line, shaped=True)

    # top phi
    f.SignalHist(top1_phi, weight, 26, N_arr, close=True, xtitle=r't$\bar{t}$ angle \phi (rad)', ytitle="rad", title=r'top quark $\phi$',
                  saveas="Deep/top1/phi", signals = select_signals, 
                  shaped = True, line = select_line)
    f.SignalHist(top2_phi, weight, 26, N_arr, close=True, xtitle=r't$\bar{t}$ angle \phi (rad)', ytitle="rad", title=r'top quark $\phi$',
                  saveas="Deep/top2/phi", signals = select_signals, 
                  shaped = True, line = select_line)
    
    # top eta
    f.SignalHist(top1_eta, weight, 25, N_arr, close=True, xtitle="pseudorapidity", ytitle="Events", title="top 1 eta",
                 saveas="Deep/top1/eta", signals = select_signals, line = select_line, shaped=True)
    f.SignalHist(top2_eta, weight, 25, N_arr, close=True, xtitle="pseudorapidity", ytitle="Events", title="top 2 eta",
             saveas="Deep/top2/eta", signals = select_signals, line = select_line, shaped=True)
    
    ### MET
    
    # MET pt
    f.SignalHist(met_pt, weight, 25, N_arr, close=True, xtitle="$E^{T}_{miss}$ (GeV)", ytitle="Events", title="$E^{T}_{miss}$",
                 saveas="Deep/met/pt", signals = select_signals, line = select_line, shaped=True)
    # MET phi
    f.SignalHist(met_phi, weight, 25, N_arr, close=True, xtitle="$E^{T}_{miss}$ $\phi$ (rad)", ytitle="rad", title="$E^{T}_{miss}$ $\phi$",
             saveas="Deep/met/phi", signals = select_signals, line = select_line, shaped=True)
    
    ### Finishing statement
    print(' - Done!')
    
#############################
#  Machine learning aspect  #
#############################

'INFO'

# Ordering of the signal types
# N_ttZ,N_ttWm,N_ttWp,N_ggA_460_360,N_ggA_500_360,N_ggA_600_360,N_ggA_600_400,N_ggA_600_500,N_ggA_500_400,N_ggA_500_400_1

# Signal lengths post event removal
# print('Length of 460_360: ', N_ggA_460_360-N_ttWp)              # ggA_460_360   - [N_ttWp:N_ggA_460_360]
# print('500_360: ', N_ggA_500_360-N_ggA_460_360)       # ggA_500_360   - [N_ggA_460_360:N_ggA_500_360]
# print('600_360: ', N_ggA_600_360-N_ggA_500_360)       # ggA_600_360   - [N_ggA_500_360:N_ggA_600_360]
# print('600_400: ', N_ggA_600_400-N_ggA_600_360)       # ggA_600_400   - [N_ggA_600_360:N_ggA_600_400]
# print('600_500: ', N_ggA_600_500-N_ggA_600_400)       # ggA_600_500   - [N_ggA_600_400:N_ggA_600_500]
# print('500_400: ', N_ggA_500_400-N_ggA_600_500)       # ggA_500_400   - [N_ggA_600_500:N_ggA_500_400]
# print('500_400_1: ', N_ggA_500_400_1-N_ggA_500_400)   # ggA_500_400_1 - [N_ggA_500_400:N_ggA_500_400_1]


#############################
#  SUPPORT VECTOR MACHINE   #
#############################

doSVM = True

# 460_360
forceModel1 = True

# 500_360
forceModel2 = True

#600_360
forceModel3 = True

# 600_400
forceModel4 = False

# 600_500
forceModel5 = False

# 500_400_1
forceModel6 = False

SVM_Opt1 = False
SVM_Opt2 = False
SVM_Opt3 = False
SVM_Opt4 = False
SVM_Opt5 = False
SVM_Opt6 = False

if doSVM:
    
    ###########
    # 460_360 #
    ###########
    
    ### Model 1 ###
    model_1 = (delta_m, Z_pt, H_pt, top_dr, met_pt)
    
    # Prepare for SVM usage
    (model_1_data, model_1_data_norm, X_train_1, y_train_1, 
     X_test_1, y_test_1, w_train_1, w_test_1, w_norm_train_1, y_binary_1, 
     SVM_N_train_1) = f.data_prep(model_1, N_ttZ, N, weight, N_ttWp, N_ggA_460_360)
    
    y_train_1 = y_train_1.astype(int)
    
    # Initial time for the runtime
    SVM_460_time = time.time()
    
    # Build the SVM
    model_1_prob_train, model_1_prob_test = s.SVM(X_train_1, y_train_1, X_test_1, 
                                                  C=0.1, gamma=0.1, tol=0.1, tag='1', ForceModel=forceModel1)
    
    # Runtime for computational efficiency section
    SVM_460_time = time.time() - SVM_460_time
    
    ### PLOTS 1 ###
    
    # ROC Curve
    f.ROC_Curve(model_1_prob_train, model_1_prob_test, y_train_1, y_test_1, 
                close=True, title="SVM $m_A$=460 $m_H$=360", saveas='SVM/460_360/')
    
    f.ProbHist(model_1_prob_train, model_1_prob_test, y_train_1, y_test_1, 
               w_train_1, w_test_1, 21, close=True, 
               label=['ttZ','ggA_460_360'], xtitle="Probability of signal", ytitle="Events", 
            title="SVM $m_A$=460 $m_H$=360", saveas='SVM/460_360/')
    
    SVM_1_bkg_count, SVM_1_sig_count = f.ProbLimitCount(model_1_prob_train, model_1_prob_test, y_train_1, y_test_1, 
               w_train_1, w_test_1, 21, close=True, 
              label=['ttZ','ggA_460_360'], xtitle="Probability of signal", ytitle="Events", 
            title="SVM $m_A$=460 $m_H$=360", saveas='SVM/460_360/')

    C = (1000,100,10,1,0.1,0.01,0.001,0.0001)
    gamma = (1000,100,10,1,0.1,0.01,0.001,0.0001)
    tol = (1, 0.5, 0.1, 0.01, 0.001)
    
    if SVM_Opt1:
        s.SVM_opt(X_train_1, y_train_1, X_test_1, y_test_1,  w_train_1, w_test_1, C=C, gamma=gamma, tol=tol, tag='460_360')   
    
    ###########
    # 500_360 #
    ###########
    
    ### Model 2 ###
    model_2 = (delta_m, Z_pt, H_pt, M_ttZ, top_dr, lep_dr)
    
    # Prepare for SVM usage
    (model_2_data, model_2_data_norm, X_train_2, y_train_2, 
     X_test_2, y_test_2, w_train_2, w_test_2, w_norm_train_2, y_binary_2, 
     SVM_N_train_2) = f.data_prep(model_2, N_ttZ, N, weight, N_ggA_460_360, N_ggA_500_360)
    
    y_train_2 = y_train_2.astype(int)
    
    # Build the SVM
    model_2_prob_train, model_2_prob_test = s.SVM(X_train_2, y_train_2, X_test_2, 
                                                  C=1, gamma=1, tol=1, tag='2', ForceModel=forceModel2)
    
    ### PLOTS 2 ###
    
    # ROC Curve
    f.ROC_Curve(model_2_prob_train, model_2_prob_test, y_train_2, y_test_2, 
                close=True, title="SVM $m_A$=500 $m_H$=360", saveas='SVM/500_360/')
    
    f.ProbHist(model_2_prob_train, model_2_prob_test, y_train_2, y_test_2, 
               w_train_2, w_test_2, 21, close=True, 
               label=['ttZ','ggA_500_360'], xtitle="Probability of signal", ytitle="Events", 
            title=r"SVM $m_A$=500 $m_H$=360", saveas='SVM/500_360/')
    
    SVM_2_bkg_count, SVM_2_sig_count = f.ProbLimitCount(model_2_prob_train, model_2_prob_test, y_train_2, y_test_2, 
               w_train_2, w_test_2, 21, close=True, 
              label=['ttZ','ggA_500_360'], xtitle="Probability of signal", ytitle="Events", 
            title="SVM $m_A$=500 $m_H$=360", saveas='SVM/500_360/')

    if SVM_Opt2:
        s.SVM_opt(X_train_2, y_train_2, X_test_2, y_test_2, w_train_2, w_test_2, C=C, gamma=gamma, tol=tol, tag='500_360')   
        
    ###########
    # 600_360 #
    ###########
    
    ### Model 3 ###
    model_3 = (delta_m, Z_pt, H_pt, M_ttZ, top_dr, lep_dr)
    
    # Prepare for SVM usage
    (model_3_data, model_3_data_norm, X_train_3, y_train_3, 
     X_test_3, y_test_3, w_train_3, w_test_3, w_norm_train_3, y_binary_3, 
     SVM_N_train_3) = f.data_prep(model_3, N_ttZ, N, weight, N_ggA_500_360, N_ggA_600_360)
    
    y_train_3 = y_train_3.astype(int)
    
    # Build the SVM
    model_3_prob_train, model_3_prob_test = s.SVM(X_train_3, y_train_3, X_test_3, 
                                                  C=0.1, gamma=0.1, tol=0.001, tag='3', ForceModel=forceModel3)
    
    ### PLOTS 3 ###
    
    # ROC Curve
    f.ROC_Curve(model_3_prob_train, model_3_prob_test, y_train_3, y_test_3, 
                close=True, title="SVM $m_A$=600 $m_H$=360", saveas='SVM/600_360/')
    
    f.ProbHist(model_3_prob_train, model_3_prob_test, y_train_3, y_test_3, 
               w_train_3, w_test_3, 21, close=True, 
               label=['ttZ','ggA_600_360'], xtitle="Probability of signal", ytitle="Events", 
            title="SVM $m_A$=600 $m_H$=360", saveas='SVM/600_360/')
    
    SVM_3_bkg_count, SVM_3_sig_count = f.ProbLimitCount(model_3_prob_train, model_3_prob_test, y_train_3, y_test_3, 
               w_train_3, w_test_3, 21, close=True, 
              label=['ttZ','ggA_600_360'], xtitle="Probability of signal", ytitle="Events", 
            title="SVM $m_A$=600 $m_H$=360", saveas='SVM/600_360/')
    
    if SVM_Opt3:
        s.SVM_opt(X_train_3, y_train_3, X_test_3, y_test_3, w_train_3, w_test_3, C=C, gamma=gamma, tol=tol, tag='600_360')   
        
    ###########
    # 600_400 #
    ###########
    
    ### Model 4 ###
    model_4 = (delta_m, Z_pt, H_pt, M_ttZ, top_dr, lep_dr)
    
    # Prepare for SVM usage
    (model_4_data, model_4_data_norm, X_train_4, y_train_4, 
     X_test_4, y_test_4, w_train_4, w_test_4, w_norm_train_4, y_binary_4, 
     SVM_N_train_4) = f.data_prep(model_4, N_ttZ, N, weight, N_ggA_600_360, N_ggA_600_400)
    
    y_train_4 = y_train_4.astype(int)
    
    # Build the SVM
    model_4_prob_train, model_4_prob_test = s.SVM(X_train_4, y_train_4, X_test_4, 
                                                  C=10, gamma=0.01, tol=1, tag='4', ForceModel=forceModel4)
    
    ### PLOTS 4 ###
    
    # ROC Curve
    f.ROC_Curve(model_4_prob_train, model_4_prob_test, y_train_4, y_test_4, 
                close=True, title="SVM $m_A$=600 $m_H$=400", saveas='SVM/600_400/')
    
    f.ProbHist(model_4_prob_train, model_4_prob_test, y_train_4, y_test_4, 
               w_train_4, w_test_4, 21, close=True, 
               label=['ttZ','ggA_600_400'], xtitle="Probability of signal", ytitle="Events", 
            title="SVM $m_A$=600 $m_H$=400", saveas='SVM/600_400/')
    
    SVM_4_bkg_count, SVM_4_sig_count = f.ProbLimitCount(model_4_prob_train, model_4_prob_test, y_train_4, y_test_4, 
               w_train_4, w_test_4, 21, close=True, 
              label=['ttZ','ggA_600_400'], xtitle="Probability of signal", ytitle="Events", 
            title="SVM $m_A$=600 $m_H$=400", saveas='SVM/600_400/')
    
    if SVM_Opt4:
        s.SVM_opt(X_train_4, y_train_4, X_test_4, y_test_4, w_train_4, w_test_4, C=C, gamma=gamma, tol=tol, tag='600_400')   
        
    ###########
    # 600_500 #
    ###########
    
    ### Model 5 ###
    model_5 = (delta_m, Z_pt, H_pt, M_ttZ, top_dr)
    
    # Prepare for SVM usage
    (model_5_data, model_5_data_norm, X_train_5, y_train_5, 
     X_test_5, y_test_5, w_train_5, w_test_5, w_norm_train_5, y_binary_5, 
     SVM_N_train_5) = f.data_prep(model_5, N_ttZ, N, weight, N_ggA_600_400, N_ggA_600_500)
    
    y_train_5 = y_train_5.astype(int)
    
    # Build the SVM
    model_5_prob_train, model_5_prob_test = s.SVM(X_train_5, y_train_5, X_test_5, 
                                                  C=10, gamma=0.01, tol=1, tag='5', ForceModel=forceModel5)
    
    ### PLOTS 5 ###
    
    # ROC Curve
    f.ROC_Curve(model_5_prob_train, model_5_prob_test, y_train_5, y_test_5, 
                close=True, title="SVM $m_A$=600 $m_H$=500", saveas='SVM/600_500/')
    
    f.ProbHist(model_5_prob_train, model_5_prob_test, y_train_5, y_test_5, 
               w_train_5, w_test_5, 21, close=True, 
               label=['ttZ','ggA_600_500'], xtitle="Probability of signal", ytitle="Events", 
            title="SVM $m_A$=600 $m_H$=500", saveas='SVM/600_500/')
    
    SVM_5_bkg_count, SVM_5_sig_count = f.ProbLimitCount(model_5_prob_train, model_5_prob_test, y_train_5, y_test_5, 
               w_train_5, w_test_5, 21, close=True, 
              label=['ttZ','ggA_600_500'], xtitle="Probability of signal", ytitle="Events", 
            title="SVM $m_A$=600 $m_H$=500", saveas='SVM/600_500/')
    
    if SVM_Opt5:
        s.SVM_opt(X_train_5, y_train_5, X_test_5, y_test_5, w_train_5, w_test_5, C=C, gamma=gamma, tol=tol, tag='600_500')   
        
    #############
    # 500_400_1 #
    #############

    ### Model 6 ###
    model_6 = (delta_m, Z_pt, H_pt, top_dr, met_pt)
    
    # Prepare for SVM usage
    (model_6_data, model_6_data_norm, X_train_6, y_train_6, 
     X_test_6, y_test_6, w_train_6, w_test_6, w_norm_train_6, y_binary_6, 
     SVM_N_train_6) = f.data_prep(model_6, N_ttZ, N, weight, N_ggA_500_400, N_ggA_500_400_1, cut_percent=0)
    
    y_train_6 = y_train_6.astype(int)
    
    # Build the SVM
    model_6_prob_train, model_6_prob_test = s.SVM(X_train_6, y_train_6, X_test_6, 
                                                  C=0.01, gamma=0.01, tol=0.1, tag='6', ForceModel=forceModel6)
    
    ### PLOTS 6 ###
    
    # ROC Curve
    SVM_uncut_test_AUC = f.ROC_Curve(model_6_prob_train, model_6_prob_test, y_train_6, y_test_6, 
                                     close=True, title="SVM $m_A$=500 $m_H$=400", saveas='SVM/500_400_1/')
    
    f.ProbHist(model_6_prob_train, model_6_prob_test, y_train_6, y_test_6, 
               w_train_6, w_test_6, 21, close=True, 
               label=['ttZ','ggA_500_400_1'], xtitle="Probability of signal", ytitle="Events", 
            title="SVM $m_A$=500 $m_H$=400", saveas='SVM/500_400_1/')
    
    SVM_6_bkg_count, SVM_6_sig_count = f.ProbLimitCount(model_6_prob_train, model_6_prob_test, y_train_6, y_test_6, 
               w_train_6, w_test_6, 21, close=True, 
              label=['ttZ','ggA_500_400_1'], xtitle="Probability of signal", ytitle="Events", 
            title="SVM $m_A$=500 $m_H$=400", saveas='SVM/500_400_1/')
    
    if SVM_Opt6:
        s.SVM_opt(X_train_6, y_train_6, X_test_6, y_test_6, w_train_6, w_test_6, C=C, gamma=gamma, tol=tol, tag='500_400_1')   

#####################
#  SHALLOW NETWORK  #
#####################

doSL = True

# 460_360
ForceSL1 = True 

# 500_360
ForceSL2 = True

# 600_360
ForceSL3 = True

# 600_400
ForceSL4 = False

# 600_500
ForceSL5 = False

# 500_400_1
ForceSL6 = False

### OPT
SL_Opt_1 = False
SL_Opt_2 = False
SL_Opt_3 = False
SL_Opt_4 = False
SL_Opt_5 = False

if doSL:
    
    ###########
    # 460_360 #
    ###########
    
    ### SL model 1 ###
    SL_model_1 = (delta_m, Z_pt, H_pt, top_dr, lep_dr)
    
    # Prepare data for SL
    (SL_model_1_data, SL_model_1_data_norm, SL_X_train_1, SL_y_train_1, SL_X_test_1, 
     SL_y_test_1, SL_w_train_1, SL_w_test_1, SL_w_norm_train_1,  SL_y_binary_1, 
     SL_N_train_1) = f.data_prep(SL_model_1, N_ttZ, N, weight, N_ttWp, N_ggA_460_360)
    
    model_length = len(SL_model_1)
    
    # Extract all X data for prediction
    SL_all_1 = SL_model_1_data_norm[:][:,0:model_length]
        
    # Define the learning rate
    SL_lr_1 = 0.0001
    SL_epoch_1 = 200
    SL_batch_1 = 5
    
    input_node = 8#int(len(SL_X_train_1[0]))
    mid_node = 8#int(input_node*2)
    
    # Initial time for the runtime
    SL_460_time = time.time()
    
    ### The model ###
    SL_pred_1_train, SL_pred_1_test, SL_y_train_1, SL_y_test_1, SL_w_train_1, SL_w_test_1 = m.ML(
        SL_X_train_1, SL_y_train_1, SL_X_test_1, SL_y_test_1, model_length,
                 SL_w_train_1, SL_w_test_1, SL_w_norm_train_1, forceFit=ForceSL1, close=True, type_tag = ['SL', '460_360'],
                 
                 # Epochs batch and lr
                 epochs = SL_epoch_1, batch = SL_batch_1, lr = SL_lr_1,
                 
                 # Early stopping
                 doES=True, ESpat=30,
                 
                 # Learning rate reduction
                 doRL=False, RLrate=0.1, RLpat=100,
                 
                 # Nodes
                 input_node = input_node, mid_node = mid_node)

    # Runtime for computational efficiency section
    SL_460_time = time.time() - SL_460_time

    f.ROC_Curve(SL_pred_1_train, SL_pred_1_test, SL_y_train_1, SL_y_test_1, close=True, 
                title=('SL $m_A$=460 $m_H$=360'), saveas=('SL/460_360/'))
    
    f.ProbHist(SL_pred_1_train, SL_pred_1_test, SL_y_train_1, SL_y_test_1, 
               SL_w_train_1, SL_w_test_1, 21, close=True, 
             label=['ttZ','ggA_460_360'], xtitle="Probability of signal", ytitle="Events", 
                title=('SL $m_A$=460 $m_H$=360'), saveas=('SL/460_360/'))

    SL_1_bkg_count, SL_1_sig_count = f.ProbLimitCount(SL_pred_1_train, SL_pred_1_test, SL_y_train_1, SL_y_test_1, 
               SL_w_train_1, SL_w_test_1, 21, close=True, 
              label=['ttZ','ggA_460_360'], xtitle="Probability of signal", ytitle="Events", 
                title=('SL $m_A$=460 $m_H$=360'), saveas=('SL/460_360/'))

    if SL_Opt_1:
            
            (SL_model_1_data, SL_model_1_data_norm, SL_X_train_1, SL_y_train_1, SL_X_test_1, 
             SL_y_test_1, SL_w_train_1, SL_w_test_1, SL_w_norm_train_1,  SL_y_binary_1, 
             SL_N_train_1) = f.data_prep(SL_model_1, N_ttZ, N, weight, N_ttWp, N_ggA_460_360)
            
            input_node = np.array([8,12])
            mid_node = np.array([8,12,16])
            SL_epoch_1 = np.array([200])
            SL_batch_1 = np.array([5])
            SL_lr_1 = np.array([0.01, 0.001, 0.0001])
            
            m.ML_opt(SL_X_train_1, SL_y_train_1, SL_X_test_1, SL_y_test_1, 
                     SL_w_train_1, SL_w_test_1, SL_w_norm_train_1, model_length,
               SL_epoch_1, SL_batch_1, SL_lr_1, 
               doES=True, ESpat=35,
               doRL=False, RLrate=0, RLpat=0,
               input_node=input_node, mid_node=mid_node,
               
               type_tag = ['SL', '460_360'])
            
    ###########
    # 500_360 #
    ###########
    
    ### SL model 2 ###
    SL_model_2 = (delta_m, Z_pt, H_pt, top_dr, lep_dr)
    
    # Prepare data for SL
    (SL_model_2_data, SL_model_2_data_norm, SL_X_train_2, SL_y_train_2, SL_X_test_2, 
      SL_y_test_2, SL_w_train_2, SL_w_test_2, SL_w_norm_train_2, SL_y_binary_2, 
      SL_N_train_2) = f.data_prep(SL_model_2, N_ttZ, N, weight, N_ggA_460_360, N_ggA_500_360)
    
    model_length = len(SL_model_2)
    
    # Extract all X data for prediction
    SL_all_2 = SL_model_2_data_norm[:][:,0:model_length]
        
    # Define the learning rate
    SL_lr_2 = 0.0001
    SL_epoch_2 = 200
    SL_batch_2 = 5
    
    input_node = 8#int(len(SL_X_train_2[0]))
    mid_node = 12#int(input_node*2)
    
    ### The model ###
    SL_pred_2_train, SL_pred_2_test, SL_y_train_2, SL_y_test_2, SL_w_train_2, SL_w_test_2 = m.ML(
        SL_X_train_2, SL_y_train_2, SL_X_test_2, SL_y_test_2, model_length, 
        SL_w_train_2, SL_w_test_2, SL_w_norm_train_2, forceFit=ForceSL2, close=True, type_tag = ['SL', '500_360'],
                 
                  # Epochs batch and lr
                  epochs = SL_epoch_2, batch = SL_batch_2, lr = SL_lr_2,
                 
                  # Early stopping
                  doES=True, ESpat=30,
                 
                  # Learning rate reduction
                  doRL=False, RLrate=0.1, RLpat=100,
                 
                  # Nodes
                  input_node = input_node, mid_node = mid_node)

    f.ROC_Curve(SL_pred_2_train, SL_pred_2_test, SL_y_train_2, SL_y_test_2, close=True, 
                title=('SL $m_A$=500 $m_H$=360'), saveas=('SL/500_360/'))
    
    f.ProbHist(SL_pred_2_train, SL_pred_2_test, SL_y_train_2, SL_y_test_2,
               SL_w_train_2, SL_w_test_2, 21, close=True, 
              label=['ttZ','ggA_500_360'], xtitle="Probability of signal", ytitle="Events", 
                title=('SL $m_A$=500 $m_H$=360'), saveas=('SL/500_360/'))
    
    SL_2_bkg_count, SL_2_sig_count = f.ProbLimitCount(SL_pred_2_train, SL_pred_2_test, SL_y_train_2, SL_y_test_2, 
               SL_w_train_2, SL_w_test_2, 21, close=True, 
              label=['ttZ','ggA_500_360'], xtitle="Probability of signal", ytitle="Events", 
                title=('SL $m_A$=500 $m_H$=360'), saveas=('SL/500_360/'))
    
    if SL_Opt_2:
            
            (SL_model_2_data, SL_model_2_data_norm, SL_X_train_2, SL_y_train_2, SL_X_test_2, 
              SL_y_test_2, SL_w_train_2, SL_w_test_2, SL_w_norm_train_2, SL_y_binary_2, 
              SL_N_train_2) = f.data_prep(SL_model_2, N_ttZ, N, weight, N_ggA_460_360, N_ggA_500_360)
        
            input_node = np.array([8,12])
            mid_node = np.array([8,12,16])
            SL_epoch_2 = np.array([200])
            SL_batch_2 = np.array([5])
            SL_lr_2 = np.array([0.01, 0.001, 0.0001])
            
            m.ML_opt(SL_X_train_2, SL_y_train_2, SL_X_test_2, SL_y_test_2, 
                     SL_w_train_2, SL_w_test_2, SL_w_norm_train_2, model_length,
               SL_epoch_2, SL_batch_2, SL_lr_2, 
               doES=True, ESpat=35,
               doRL=False, RLrate=0, RLpat=0,
               input_node=input_node, mid_node=mid_node,
               
               type_tag = ['SL', '500_360'])

    ###########
    # 600_360 #
    ###########
    
    ### SL model 3 ###
    SL_model_3 = (delta_m, Z_pt, H_pt, top_dr, lep_dr)
    
    # Prepare data for SL
    (SL_model_3_data, SL_model_3_data_norm, SL_X_train_3, SL_y_train_3, SL_X_test_3, 
      SL_y_test_3, SL_w_train_3, SL_w_test_3, SL_w_norm_train_3, SL_y_binary_3, 
      SL_N_train_3) = f.data_prep(SL_model_3, N_ttZ, N, weight, N_ggA_500_360, N_ggA_600_360)
    
    model_length = len(SL_model_3)
    
    # Extract all X data for prediction
    SL_all_3 = SL_model_3_data_norm[:][:,0:model_length]
        
    # Define the learning rate
    SL_lr_3 = 0.001
    SL_epoch_3 = 200
    SL_batch_3 = 5
    
    input_node = 12#int(len(SL_X_train_3[0]))
    mid_node = 8#int(input_node*3)
    
    ### The model ###
    SL_pred_3_train, SL_pred_3_test, SL_y_train_3, SL_y_test_3, SL_w_train_3, SL_w_test_3 = m.ML(
        SL_X_train_3, SL_y_train_3, SL_X_test_3, SL_y_test_3, model_length, 
        SL_w_train_3, SL_w_test_3, SL_w_norm_train_3, forceFit=ForceSL3, close=True, type_tag = ['SL', '600_360'],
                 
                  # Epochs batch and lr
                  epochs = SL_epoch_3, batch = SL_batch_3, lr = SL_lr_3,
                 
                  # Early stopping
                  doES=True, ESpat=35,
                 
                  # Learning rate reduction
                  doRL=False, RLrate=0.1, RLpat=100,
                 
                  # Nodes
                  input_node = input_node, mid_node = mid_node)

    f.ROC_Curve(SL_pred_3_train, SL_pred_3_test, SL_y_train_3, SL_y_test_3, close=True, 
                title=('SL $m_A$=600 $m_H$=360'), saveas=('SL/600_360/'))
    
    f.ProbHist(SL_pred_3_train, SL_pred_3_test, SL_y_train_3, SL_y_test_3, 
               SL_w_train_3, SL_w_test_3, 21, close=True, 
              label=['ttZ','ggA_600_360'], xtitle="Probability of signal", ytitle="Events", 
                title=('SL $m_A$=600 $m_H$=360'), saveas=('SL/600_360/'))
    
    SL_3_bkg_count, SL_3_sig_count = f.ProbLimitCount(SL_pred_3_train, SL_pred_3_test, SL_y_train_3, SL_y_test_3, 
               SL_w_train_3, SL_w_test_3, 21, close=True, 
              label=['ttZ','ggA_600_360'], xtitle="Probability of signal", ytitle="Events", 
                title=('SL $m_A$=600 $m_H$=360'), saveas=('SL/600_360/'))
    
    if SL_Opt_3:
            
            (SL_model_3_data, SL_model_3_data_norm, SL_X_train_3, SL_y_train_3, SL_X_test_3, 
              SL_y_test_3, SL_w_train_3, SL_w_test_3, SL_w_norm_train_3, SL_y_binary_3, 
              SL_N_train_3) = f.data_prep(SL_model_3, N_ttZ, N, weight, N_ggA_500_360, N_ggA_600_360)
            
            input_node = np.array([8,12])
            mid_node = np.array([8,12,16])
            SL_epoch_3 = np.array([200])
            SL_batch_3 = np.array([5])
            SL_lr_3 = np.array([0.01, 0.001, 0.0001])
            
            m.ML_opt(SL_X_train_3, SL_y_train_3, SL_X_test_3, SL_y_test_3, 
                     SL_w_train_3, SL_w_test_3, SL_w_norm_train_3, model_length,
               SL_epoch_3, SL_batch_3, SL_lr_3, 
               doES=True, ESpat=35,
               doRL=False, RLrate=0, RLpat=0,
               input_node=input_node, mid_node=mid_node,
               
               type_tag = ['SL', '600_360'])
    
    ###########
    # 600_400 #
    ###########
    
    ### SL model 4 ###
    SL_model_4 = (delta_m, Z_pt, H_pt, M_ttZ, top_dr, lep_dr)
    
    # Prepare data for SL
    (SL_model_4_data, SL_model_4_data_norm, SL_X_train_4, SL_y_train_4, SL_X_test_4, 
      SL_y_test_4, SL_w_train_4, SL_w_test_4, SL_w_norm_train_4, SL_y_binary_4, 
      SL_N_train_4) = f.data_prep(SL_model_4, N_ttZ, N, weight, N_ggA_600_360, N_ggA_600_400)
    
    model_length = len(SL_model_4)
    
    # Extract all X data for prediction
    SL_all_4 = SL_model_4_data_norm[:][:,0:model_length]
        
    # Define the learning rate
    SL_lr_4 = 0
    SL_epoch_4 = 200
    SL_batch_4 = 5
    
    input_node = 8#int(len(SL_X_train_4[0]))
    mid_node = 8#int(input_node*4)
    
    ### The model ###
    SL_pred_4_train, SL_pred_4_test, SL_y_train_4, SL_y_test_4, SL_w_train_4, SL_w_test_4 = m.ML(
        SL_X_train_4, SL_y_train_4, SL_X_test_4, SL_y_test_4, model_length, 
        SL_w_train_4, SL_w_test_4, SL_w_norm_train_4, forceFit=ForceSL4, close=True, type_tag = ['SL', '600_400'],
                 
                  # Epochs batch and lr
                  epochs = SL_epoch_4, batch = SL_batch_4, lr = SL_lr_4,
                 
                  # Early stopping
                  doES=True, ESpat=40,
                 
                  # Learning rate reduction
                  doRL=False, RLrate=0.1, RLpat=100,
                 
                  # Nodes
                  input_node = input_node, mid_node = mid_node)

    f.ROC_Curve(SL_pred_4_train, SL_pred_4_test, SL_y_train_4, SL_y_test_4, close=True, 
                title=('SL $m_A$=600 $m_H$=400'), saveas=('SL/600_400/'))
    
    f.ProbHist(SL_pred_4_train, SL_pred_4_test, SL_y_train_4, SL_y_test_4, 
               SL_w_train_4, SL_w_test_4, 21, close=True, 
              label=['ttZ','ggA_600_400'], xtitle="Probability of signal", ytitle="Events", 
                title=('SL $m_A$=600 $m_H$=400'), saveas=('SL/600_400/'))
    
    SL_4_bkg_count, SL_4_sig_count = f.ProbLimitCount(SL_pred_4_train, SL_pred_4_test, SL_y_train_4, SL_y_test_4, 
               SL_w_train_4, SL_w_test_4, 21, close=True, 
              label=['ttZ','ggA_600_400'], xtitle="Probability of signal", ytitle="Events", 
                title=('SL $m_A$=600 $m_H$=400'), saveas=('SL/600_400/'))
    
    if SL_Opt_4:
            
            (SL_model_4_data, SL_model_4_data_norm, SL_X_train_4, SL_y_train_4, SL_X_test_4, 
              SL_y_test_4, SL_w_train_4, SL_w_test_4, SL_w_norm_train_4, SL_y_binary_4, 
              SL_N_train_4) = f.data_prep(SL_model_4, N_ttZ, N, weight, N_ggA_600_360, N_ggA_600_400)
            
            input_node = np.array([input_node])
            mid_node = np.array([mid_node, mid_node*(4/3)])
            SL_epoch_4 = np.array([150])
            SL_batch_4 = np.array([10,20])
            SL_lr_4 = np.array([0.01, 0.001])
            
            m.SL_opt(SL_X_train_4, SL_y_train_4, SL_X_test_4, SL_y_test_4, 
                     SL_w_train_4, SL_w_test_4, SL_w_norm_train_4, model_length,
               SL_epoch_4, SL_batch_4, SL_lr_4, 
               doES=True, ESpat=35,
               doRL=False, RLrate=0, RLpat=0,
               input_node=input_node, mid_node=mid_node,
               
               type_tag = ['SL', '600_400'])
    
    ###########
    # 600_500 #
    ###########
    
    ### SL model 5 ###
    SL_model_5 = (delta_m, Z_pt, H_pt, M_ttZ, top_dr, lep_dr, top1_pt, top2_pt, met_pt, met_phi)
    
    # Prepare data for SL
    (SL_model_5_data, SL_model_5_data_norm, SL_X_train_5, SL_y_train_5, SL_X_test_5, 
      SL_y_test_5, SL_w_train_5, SL_w_test_5, SL_w_norm_train_5, SL_y_binary_5, 
      SL_N_train_5) = f.data_prep(SL_model_5, N_ttZ, N, weight, N_ggA_600_400, N_ggA_600_500)
    
    model_length = len(SL_model_5)
    
    # Extract all X data for prediction
    SL_all_5 = SL_model_5_data_norm[:][:,0:model_length]
        
    # Define the learning rate
    SL_lr_5 = 0
    SL_epoch_5 = 200
    SL_batch_5 = 5
    
    input_node = 8#int(len(SL_X_train_5[0]))
    mid_node = 8#int(input_node*5)
    
    ### The model ###
    SL_pred_5_train, SL_pred_5_test, SL_y_train_5, SL_y_test_5, SL_w_train_5, SL_w_test_5 = m.ML(
        SL_X_train_5, SL_y_train_5, SL_X_test_5, SL_y_test_5, model_length, 
        SL_w_train_5, SL_w_test_5, SL_w_norm_train_5, forceFit=ForceSL5, close=True, type_tag = ['SL', '600_500'],
                 
                  # Epochs batch and lr
                  epochs = SL_epoch_5, batch = SL_batch_5, lr = SL_lr_5,
                 
                  # Early stopping
                  doES=True, ESpat=50,
                 
                  # Learning rate reduction
                  doRL=False, RLrate=0.1, RLpat=100,
                 
                  # Nodes
                  input_node = input_node, mid_node = mid_node)

    f.ROC_Curve(SL_pred_5_train, SL_pred_5_test, SL_y_train_5, SL_y_test_5, close=True, 
                title=('SL $m_A$=600 $m_H$=500'), saveas=('SL/600_500/'))
    
    f.ProbHist(SL_pred_5_train, SL_pred_5_test, SL_y_train_5, SL_y_test_5, 
               SL_w_train_5, SL_w_test_5, 21, close=True, 
              label=['ttZ','ggA_600_500'], xtitle="Probability of signal", ytitle="Events", 
                title=('SL $m_A$=600 $m_H$=500'), saveas=('SL/600_500/'))
    
    SL_5_bkg_count, SL_5_sig_count = f.ProbLimitCount(SL_pred_5_train, SL_pred_5_test, SL_y_train_5, SL_y_test_5, 
               SL_w_train_5, SL_w_test_5, 21, close=True, 
              label=['ttZ','ggA_600_500'], xtitle="Probability of signal", ytitle="Events", 
                title=('SL $m_A$=600 $m_H$=500'), saveas=('SL/600_500/'))
    
    if SL_Opt_5:
            
            (SL_model_5_data, SL_model_5_data_norm, SL_X_train_5, SL_y_train_5, SL_X_test_5, 
              SL_y_test_5, SL_w_train_5, SL_w_test_5, SL_w_norm_train_5, SL_y_binary_5, 
              SL_N_train_5) = f.data_prep(SL_model_5, N_ttZ, N, weight, N_ggA_600_400, N_ggA_600_500)
                    
            input_node = np.array([input_node])
            mid_node = np.array([mid_node, mid_node*(4/3)])
            SL_epoch_5 = np.array([150])
            SL_batch_5 = np.array([10,20])
            SL_lr_5 = np.array([0.01, 0.001])
            
            m.SL_opt(SL_X_train_5, SL_y_train_5, SL_X_test_5, SL_y_test_5, 
                     SL_w_train_5, SL_w_test_5, SL_w_norm_train_5, model_length,
               SL_epoch_5, SL_batch_5, SL_lr_5, 
               doES=True, ESpat=35,
               doRL=False, RLrate=0, RLpat=0,
               input_node=input_node, mid_node=mid_node,
               
               type_tag = ['SL', '600_500'])
    
    #############
    # 500_400_1 #
    #############
    
    ### SL model 6 ###
    SL_model_6 = (delta_m, Z_pt, H_pt, top_dr, met_pt)
    #SL_model_6 = (delta_m, Z_pt, H_pt, M_ttZ, top_dr, lep_dr)
    
    # Prepare data for SL
    (SL_model_6_data, SL_model_6_data_norm, SL_X_train_6, SL_y_train_6, SL_X_test_6, 
      SL_y_test_6, SL_w_train_6, SL_w_test_6, SL_w_norm_train_6, SL_y_binary_6, 
      SL_N_train_6) = f.data_prep(SL_model_6, N_ttZ, N, weight, N_ggA_500_400, N_ggA_500_400_1)
    
    model_length = len(SL_model_6)
    
    # Extract all X data for prediction
    SL_all_6 = SL_model_6_data_norm[:][:,0:model_length]
        
    # Define the learning rate
    SL_lr_6 = 0
    SL_epoch_6 = 200
    SL_batch_6 = 5
    
    input_node = 8#int(len(SL_X_train_6[0]))
    mid_node = 8#int(input_node*6)
    
    ### The model ###
    SL_pred_6_train, SL_pred_6_test, SL_y_train_6, SL_y_test_6, SL_w_train_6, SL_w_test_6 = m.ML(
        SL_X_train_6, SL_y_train_6, SL_X_test_6, SL_y_test_6, model_length, 
        SL_w_train_6, SL_w_test_6, SL_w_norm_train_6, forceFit=ForceSL6, close=True, type_tag = ['SL', '500_400_1'],
                 
                  # Epochs batch and lr
                  epochs = SL_epoch_6, batch = SL_batch_6, lr = SL_lr_6,
                 
                  # Early stopping
                  doES=True, ESpat=30,
                 
                  # Learning rate reduction
                  doRL=False, RLrate=0.1, RLpat=100,
                 
                  # Nodes
                  input_node = input_node, mid_node = mid_node)

    SL_uncut_test_AUC = f.ROC_Curve(SL_pred_6_train, SL_pred_6_test, SL_y_train_6, SL_y_test_6, close=True, 
                title=('SL $m_A$=500 $m_H$=400'), saveas=('SL/500_400_1/'))
    
    f.ProbHist(SL_pred_6_train, SL_pred_6_test, SL_y_train_6, SL_y_test_6, 
               SL_w_train_6, SL_w_test_6, 21, close=True, 
              label=['ttZ','ggA_500_400_1'], xtitle="Probability of signal", ytitle="Events", 
                title=('SL $m_A$=500 $m_H$=400'), saveas=('SL/500_400_1/'))
    
    SL_6_bkg_count, SL_6_sig_count = f.ProbLimitCount(SL_pred_6_train, SL_pred_6_test, SL_y_train_6, SL_y_test_6, 
               SL_w_train_6, SL_w_test_6, 21, close=True, 
              label=['ttZ','ggA_500_400_1'], xtitle="Probability of signal", ytitle="Events", 
                title=('SL $m_A$=500 $m_H$=400'), saveas=('SL/500_400_1/'))

##################
#  DEEP NETWORK  #
##################

doDL = True

# 460_360
ForceDL1 = True

# 500_360
ForceDL2 = True

# 600_360
ForceDL3 = True

# 600_400
ForceDL4 = False

# 600_500
ForceDL5 = False

# 500_400_1
ForceDL6 = False

# Optimisations
DL_Opt_1 = False
DL_Opt_2 = False
DL_Opt_3 = False
DL_Opt_4 = False
DL_Opt_5 = False
DL_Opt_6 = False

if doDL:
    
    ###########
    # 460_360 #
    ###########
    
    DL_model_1 = (lep1_pt, lep2_pt, lep3_pt,
                  
                  bjet1_pt, bjet2_pt,
                  
                  top1_pt, top2_pt,
                  
                  lep3_phi, met_pt, met_phi)
    
    # Prepare data for DL
    (DL_model_1_data, DL_model_1_data_norm, DL_X_train_1, DL_y_train_1, DL_X_test_1, 
     DL_y_test_1, DL_w_train_1, DL_w_test_1, DL_w_norm_train_1, DL_y_binary_1, 
     DL_N_train_1) = f.data_prep(DL_model_1, N_ttZ, N, weight, N_ttWp, N_ggA_460_360)

    model_length = len(DL_model_1)
    
    # Extract all X data for prediction
    DL_all_1 = DL_model_1_data_norm[:][:,0:model_length]
    
    # Define the learning rate
    #!!! Change this back
    #DL_lr_1 = 0.0001
    
    DL_lr_1 = 0.01
    DL_epoch_1 = 200
    DL_batch_1 = 5
    
    input_node = 12#len(DL_X_train_1[0])
    mid_node = 12#int(input_node*2)*(4/3)
    extra_node = 16#int(input_node*2)*(3/2)
    
    # Initial time for the runtime
    DL_460_time = time.time()
    
    ### The model ###
    DL_pred_1_train, DL_pred_1_test, DL_y_train_1, DL_y_test_1, DL_w_train_1, DL_w_test_1 = m.ML(
        DL_X_train_1, DL_y_train_1, DL_X_test_1, DL_y_test_1, model_length, 
        DL_w_train_1, DL_w_test_1, DL_w_norm_train_1, forceFit=ForceDL1, close=True, type_tag = ['DL', '460_360'],
                 
                 # Epochs batch and lr
                 epochs = DL_epoch_1, batch = DL_batch_1, lr = DL_lr_1,
                 
                 # Early stopping
                 doES=True, ESpat=35,
                 
                 # Learning rate reduction
                 doRL=False, RLrate=0.1, RLpat=20,
                 
                 # Nodes
                 input_node = input_node, mid_node = mid_node, extra_node = extra_node
                 )

    # Runtime for computational efficiency section
    DL_460_time = time.time() - DL_460_time

    f.ROC_Curve(DL_pred_1_train, DL_pred_1_test, DL_y_train_1, DL_y_test_1, close=True, 
                title=('DL $m_A$=460 $m_H$=360'), saveas=('DL/460_360/'))
    
    f.ProbHist(DL_pred_1_train, DL_pred_1_test, DL_y_train_1, DL_y_test_1, 
               DL_w_train_1, DL_w_test_1, 21, close=True, 
             label=['ttZ','ggA_460_360'], xtitle="Probability of signal", ytitle="Events", 
                title=('DL $m_A$=460 $m_H$=360'), saveas=('DL/460_360/'))
    
    DL_1_bkg_count, DL_1_sig_count = f.ProbLimitCount(DL_pred_1_train, DL_pred_1_test, DL_y_train_1, DL_y_test_1, 
               DL_w_train_1, DL_w_test_1, 21, close=True, 
             label=['ttZ','ggA_460_360'], xtitle="Probability of signal", ytitle="Events", 
                title=('DL $m_A$=460 $m_H$=360'), saveas=('DL/460_360/'))

    # Run optimisation if enabled
    if DL_Opt_1:
    
        input_node = np.array([8,12])
        mid_node = np.array([8,12])
        extra_node = np.array([12,16])
        
        DL_epoch_1 = np.array([200])
        DL_batch_1 = np.array([5])
        DL_lr_1 = np.array([0.0001])
        
        (DL_model_1_data, DL_model_1_data_norm, DL_X_train_1, DL_y_train_1, DL_X_test_1, 
         DL_y_test_1, DL_w_train_1, DL_w_test_1, DL_w_norm_train_1, DL_y_binary_1, 
         DL_N_train_1) = f.data_prep(DL_model_1, N_ttZ, N, weight, N_ttWp, N_ggA_460_360)
        
        m.ML_opt(DL_X_train_1, DL_y_train_1, DL_X_test_1, DL_y_test_1, DL_w_train_1, DL_w_test_1, 
                 DL_w_norm_train_1, model_length,
           epochs = DL_epoch_1, batch = DL_batch_1, lr = DL_lr_1, 
           doES=True, ESpat=35,
           doRL=False, RLrate=0, RLpat=0,
           input_node=input_node, mid_node=mid_node, extra_node=extra_node,
           
           type_tag = ['DL', '460_360'])

    ###########
    # 500_360 #
    ###########

    DL_model_2 = (lep1_pt, lep2_pt, lep3_pt,
                  
                  bjet1_pt, bjet2_pt,
                  
                  top1_pt, top2_pt,
                  
                  lep3_phi, met_pt, met_phi)
    
    # Prepare data for DL
    (DL_model_2_data, DL_model_2_data_norm, DL_X_train_2, DL_y_train_2, DL_X_test_2, 
     DL_y_test_2, DL_w_train_2, DL_w_test_2, DL_w_norm_train_2, DL_y_binary_2, 
     DL_N_train_2) = f.data_prep(DL_model_2, N_ttZ, N, weight, N_ggA_460_360, N_ggA_500_360)

    model_length = len(DL_model_2)
    
    # Extract all X data for prediction
    DL_all_2 = DL_model_2_data_norm[:][:,0:model_length]
    
    # Define the learning rate
    DL_lr_2 = 0.00005
    DL_epoch_2 = 200
    DL_batch_2 = 5
    
    input_node = 12#len(DL_X_train_2[0])
    mid_node = 18#int(input_node*2)*(4/3)
    extra_node = 8#int(input_node*2)*(3/2)
    
    ### The model ###
    DL_pred_2_train, DL_pred_2_test, DL_y_train_2, DL_y_test_2, DL_w_train_2, DL_w_test_2 = m.ML(
        DL_X_train_2, DL_y_train_2, DL_X_test_2, DL_y_test_2, model_length, 
        DL_w_train_2, DL_w_test_2, DL_w_norm_train_2, forceFit=ForceDL2, close=True, type_tag = ['DL', '500_360'],
                 
                 # Epochs batch and lr
                 epochs = DL_epoch_2, batch = DL_batch_2, lr = DL_lr_2,
                 
                 # Early stopping
                 doES=True, ESpat=35,
                 
                 # Learning rate reduction
                 doRL=False, RLrate=0.2, RLpat=20,
                 
                 # Nodes
                 input_node = input_node, mid_node = mid_node, extra_node = extra_node
                 )

    f.ROC_Curve(DL_pred_2_train, DL_pred_2_test, DL_y_train_2, DL_y_test_2, close=True, 
                title=('DL $m_A$=500 $m_H$=360'), saveas=('DL/500_360/'))
    
    f.ProbHist(DL_pred_2_train, DL_pred_2_test, DL_y_train_2, DL_y_test_2, 
               DL_w_train_2, DL_w_test_2, 21, close=True, 
             label=['ttZ','ggA_500_360'], xtitle="Probability of signal", ytitle="Events", 
                title=('DL $m_A$=500 $m_H$=360'), saveas=('DL/500_360/'))
    
    DL_2_bkg_count, DL_2_sig_count = f.ProbLimitCount(DL_pred_2_train, DL_pred_2_test, DL_y_train_2, DL_y_test_2, 
               DL_w_train_2, DL_w_test_2, 21, close=True, 
             label=['ttZ','ggA_500_360'], xtitle="Probability of signal", ytitle="Events", 
                title=('DL $m_A$=500 $m_H$=360'), saveas=('DL/500_360/'))

    # Run optimisation if enabled
    if DL_Opt_2:
        
        (DL_model_2_data, DL_model_2_data_norm, DL_X_train_2, DL_y_train_2, DL_X_test_2, 
         DL_y_test_2, DL_w_train_2, DL_w_test_2, DL_w_norm_train_2, DL_y_binary_2, 
         DL_N_train_2) = f.data_prep(DL_model_2, N_ttZ, N, weight, N_ggA_460_360, N_ggA_500_360)
    
        input_node = np.array([6,8,12])
        mid_node = np.array([6,8,12,18])
        extra_node = np.array([8,10,12,16])
        
        DL_epoch_2 = np.array([200])
        DL_batch_2 = np.array([5])
        DL_lr_2 = np.array([0.0005, 0.0001,0.00005])
        
        m.ML_opt(DL_X_train_2, DL_y_train_2, DL_X_test_2, DL_y_test_2, DL_w_train_2, DL_w_test_2, 
                 DL_w_norm_train_2, model_length,
           DL_epoch_2, DL_batch_2, DL_lr_2, 
           doES=True, ESpat=35,
           doRL=False, RLrate=0, RLpat=0,
           input_node=input_node, mid_node=mid_node, extra_node=extra_node,
           
           type_tag = ['DL', '500_360'])

    ###########
    # 600_360 #
    ###########
    
    DL_model_3 = (lep1_pt, lep2_pt, lep3_pt,
                  
                  bjet1_pt, bjet2_pt,
                  
                  top1_pt, top2_pt,
                  
                  met_pt, met_phi)
    
    # Prepare data for DL
    (DL_model_3_data, DL_model_3_data_norm, DL_X_train_3, DL_y_train_3, DL_X_test_3, 
     DL_y_test_3, DL_w_train_3, DL_w_test_3, DL_w_norm_train_3, DL_y_binary_3, 
     DL_N_train_3) = f.data_prep(DL_model_3, N_ttZ, N, weight, N_ggA_500_360, N_ggA_600_360)

    model_length = len(DL_model_3)
    
    # Extract all X data for prediction
    DL_all_3 = DL_model_3_data_norm[:][:,0:model_length]
    
    # Define the learning rate
    DL_lr_3 = 0.0001
    DL_epoch_3 = 200
    DL_batch_3 = 5
    
    input_node = 8#len(DL_X_train_3[0])
    mid_node = 12#int(input_node*2)*(4/3)
    extra_node = 12#int(input_node*2)*(3/2)
    
    ### The model ###
    DL_pred_3_train, DL_pred_3_test, DL_y_train_3, DL_y_test_3, DL_w_train_3, DL_w_test_3 = m.ML(
        DL_X_train_3, DL_y_train_3, DL_X_test_3, DL_y_test_3, model_length, 
        DL_w_train_3, DL_w_test_3, DL_w_norm_train_3, forceFit=ForceDL3, close=True, type_tag = ['DL', '600_360'],
                 
                 # Epochs batch and lr
                 epochs = DL_epoch_3, batch = DL_batch_3, lr = DL_lr_3,
                 
                 # Early stopping
                 doES=True, ESpat=35,
                 
                 # Learning rate reduction
                 doRL=False, RLrate=0.1, RLpat=20,
                 
                 # Nodes
                 input_node = input_node, mid_node = mid_node, extra_node = extra_node
                 )

    f.ROC_Curve(DL_pred_3_train, DL_pred_3_test, DL_y_train_3, DL_y_test_3, close=True, 
                title=('DL $m_A$=600 $m_H$=360'), saveas=('DL/600_360/'))
    
    f.ProbHist(DL_pred_3_train, DL_pred_3_test, DL_y_train_3, DL_y_test_3, 
               DL_w_train_3, DL_w_test_3, 21, close=True, 
             label=['ttZ','ggA_600_360'], xtitle="Probability of signal", ytitle="Events", 
                title=('DL $m_A$=600 $m_H$=360'), saveas=('DL/600_360/'))
    
    DL_3_bkg_count, DL_3_sig_count = f.ProbLimitCount(DL_pred_3_train, DL_pred_3_test, DL_y_train_3, DL_y_test_3, 
               DL_w_train_3, DL_w_test_3, 21, close=True, 
             label=['ttZ','ggA_600_360'], xtitle="Probability of signal", ytitle="Events", 
                title=('DL $m_A$=600 $m_H$=360'), saveas=('DL/600_360/'))
    
    # Run optimisation if enabled
    if DL_Opt_3:
    
        (DL_model_3_data, DL_model_3_data_norm, DL_X_train_3, DL_y_train_3, DL_X_test_3, 
        DL_y_test_3, DL_w_train_3, DL_w_test_3, DL_w_norm_train_3, DL_y_binary_3, 
        DL_N_train_3) = f.data_prep(DL_model_3, N_ttZ, N, weight, N_ggA_500_360, N_ggA_600_360)    
    
        input_node = np.array([8,12])
        mid_node = np.array([8,12])
        extra_node = np.array([12,16])
        
        DL_epoch_3 = np.array([200])
        DL_batch_3 = np.array([5])
        DL_lr_3 = np.array([0.01, 0.001, 0.0001])
        
        m.ML_opt(DL_X_train_3, DL_y_train_3, DL_X_test_3, DL_y_test_3, DL_w_train_3, DL_w_test_3, 
                 DL_w_norm_train_3, model_length,
           DL_epoch_3, DL_batch_3, DL_lr_3, 
           doES=True, ESpat=35,
           doRL=False, RLrate=0, RLpat=0,
           input_node=input_node, mid_node=mid_node, extra_node=extra_node,
           
           type_tag = ['DL', '600_360'])
    
    ###########
    # 600_400 #
    ###########

    DL_model_4 = (lep1_four_mom[:,0], lep1_pt, lep1_four_mom[:,3],
                  lep2_four_mom[:,0], lep2_pt, lep2_four_mom[:,3],
                  
                  jet1_four_mom[:,0], jet1_pt,  jet1_four_mom[:,3],
                  jet2_four_mom[:,0], jet2_pt,  jet2_four_mom[:,3],
                  
                  met_pt, met_phi)
    
    # Prepare data for DL
    (DL_model_4_data, DL_model_4_data_norm, DL_X_train_4, DL_y_train_4, DL_X_test_4, 
     DL_y_test_4, DL_w_train_4, DL_w_test_4, DL_w_norm_train_4, DL_y_binary_4, 
     DL_N_train_4) = f.data_prep(DL_model_4, N_ttZ, N, weight, N_ggA_600_360, N_ggA_600_400)

    model_length = len(DL_model_4)
    
    # Extract all X data for prediction
    DL_all_4 = DL_model_4_data_norm[:][:,0:model_length]
    
    # Define the learning rate
    DL_lr_4 = 0.0001
    DL_epoch_4 = 300
    DL_batch_4 = 5
    
    input_node = 8#len(DL_X_train_4[0])
    mid_node = 8#int(input_node*2)*(4/3)
    extra_node = 12#int(input_node*2)*(3/2)
    
    ### The model ###
    DL_pred_4_train, DL_pred_4_test, DL_y_train_4, DL_y_test_4, DL_w_train_4, DL_w_test_4 = m.ML(
        DL_X_train_4, DL_y_train_4, DL_X_test_4, DL_y_test_4, model_length, 
        DL_w_train_4, DL_w_test_4, DL_w_norm_train_4, forceFit=ForceDL4, close=True, type_tag = ['DL', '600_400'],
                 
                 # Epochs batch and lr
                 epochs = DL_epoch_4, batch = DL_batch_4, lr = DL_lr_4,
                 
                 # Early stopping
                 doES=True, ESpat=35,
                 
                 # Learning rate reduction
                 doRL=False, RLrate=0.1, RLpat=20,
                 
                 # Nodes
                 input_node = input_node, mid_node = mid_node, extra_node = extra_node
                 )

    f.ROC_Curve(DL_pred_4_train, DL_pred_4_test, DL_y_train_4, DL_y_test_4, close=True, 
                title=('DL $m_A$=600 $m_H$=400'), saveas=('DL/600_400/'))
    
    f.ProbHist(DL_pred_4_train, DL_pred_4_test, DL_y_train_4, DL_y_test_4, 
               DL_w_train_4, DL_w_test_4, 21, close=True, 
             label=['ttZ','ggA_600_400'], xtitle="Probability of signal", ytitle="Events", 
                title=('DL $m_A$=600 $m_H$=400'), saveas=('DL/600_400/'))
    
    DL_4_bkg_count, DL_4_sig_count = f.ProbLimitCount(DL_pred_4_train, DL_pred_4_test, DL_y_train_4, DL_y_test_4, 
               DL_w_train_4, DL_w_test_4, 21, close=True, 
             label=['ttZ','ggA_600_400'], xtitle="Probability of signal", ytitle="Events", 
                title=('DL $m_A$=600 $m_H$=400'), saveas=('DL/600_400/'))
    
    # Run optimisation if enabled
    if DL_Opt_4:
    
        (DL_model_4_data, DL_model_4_data_norm, DL_X_train_4, DL_y_train_4, DL_X_test_4, 
     DL_y_test_4, DL_w_train_4, DL_w_test_4, DL_w_norm_train_4, DL_y_binary_4, 
     DL_N_train_4) = f.data_prep(DL_model_4, N_ttZ, N, weight, N_ggA_600_360, N_ggA_600_400)  
    
        input_node = np.array([8])
        mid_node = np.array([8])
        extra_node = np.array([12])
        
        DL_epoch_4 = np.array([200])
        DL_batch_4 = np.array([5])
        DL_lr_4 = np.array([0.01,0.001,0.0005,0.0001])
        
        m.ML_opt(DL_X_train_4, DL_y_train_4, DL_X_test_4, DL_y_test_4, DL_w_train_4, DL_w_test_4, 
                 DL_w_norm_train_4, model_length,
           DL_epoch_4, DL_batch_4, DL_lr_4, 
           doES=True, ESpat=35,
           doRL=False, RLrate=0, RLpat=0,
           input_node=input_node, mid_node=mid_node, extra_node=extra_node,
           
           type_tag = ['DL', '600_400'])

    
    ###########
    # 600_500 #
    ###########

    DL_model_5 = (lep1_four_mom[:,0], lep1_pt, lep1_four_mom[:,3],
                  lep2_four_mom[:,0], lep2_pt, lep2_four_mom[:,3],
                  
                  jet1_four_mom[:,0], jet1_pt,  jet1_four_mom[:,3],
                  jet2_four_mom[:,0], jet2_pt,  jet2_four_mom[:,3],
                  
                  met_pt, met_phi)
    
    # Prepare data for DL
    (DL_model_5_data, DL_model_5_data_norm, DL_X_train_5, DL_y_train_5, DL_X_test_5, 
     DL_y_test_5, DL_w_train_5, DL_w_test_5, DL_w_norm_train_5, DL_y_binary_5, 
     DL_N_train_5) = f.data_prep(DL_model_5, N_ttZ, N, weight, N_ggA_600_400, N_ggA_600_500)

    model_length = len(DL_model_5)
    
    # Extract all X data for prediction
    DL_all_5 = DL_model_5_data_norm[:][:,0:model_length]
    
    # Define the learning rate
    DL_lr_5 = 0.0001
    DL_epoch_5 = 300
    DL_batch_5 = 5
    
    input_node = 8#len(DL_X_train_5[0])
    mid_node = 8#int(input_node*2)*(4/3)
    extra_node = 12#int(input_node*2)*(3/2)
    
    ### The model ###
    DL_pred_5_train, DL_pred_5_test, DL_y_train_5, DL_y_test_5, DL_w_train_5, DL_w_test_5 = m.ML(
        DL_X_train_5, DL_y_train_5, DL_X_test_5, DL_y_test_5, model_length, 
        DL_w_train_5, DL_w_test_5, DL_w_norm_train_5, forceFit=ForceDL5, close=True, type_tag = ['DL', '600_500'],
                 
                 # Epochs batch and lr
                 epochs = DL_epoch_5, batch = DL_batch_5, lr = DL_lr_5,
                 
                 # Early stopping
                 doES=True, ESpat=35,
                 
                 # Learning rate reduction
                 doRL=False, RLrate=0.1, RLpat=20,
                 
                 # Nodes
                 input_node = input_node, mid_node = mid_node, extra_node = extra_node
                 )

    f.ROC_Curve(DL_pred_5_train, DL_pred_5_test, DL_y_train_5, DL_y_test_5, close=True, 
                title=('DL $m_A$=600 $m_H$=500'), saveas=('DL/600_500/'))
    
    f.ProbHist(DL_pred_5_train, DL_pred_5_test, DL_y_train_5, DL_y_test_5, 
               DL_w_train_5, DL_w_test_5, 21, close=True, 
             label=['ttZ','ggA_600_500'], xtitle="Probability of signal", ytitle="Events", 
                title=('DL $m_A$=600 $m_H$=500'), saveas=('DL/600_500/'))
    
    DL_5_bkg_count, DL_5_sig_count = f.ProbLimitCount(DL_pred_5_train, DL_pred_5_test, DL_y_train_5, DL_y_test_5, 
               DL_w_train_5, DL_w_test_5, 21, close=True, 
             label=['ttZ','ggA_600_500'], xtitle="Probability of signal", ytitle="Events", 
                title=('DL $m_A$=600 $m_H$=500'), saveas=('DL/600_500/'))
    
    # Run optimisation if enabled
    if DL_Opt_5:
    
        (DL_model_5_data, DL_model_5_data_norm, DL_X_train_5, DL_y_train_5, DL_X_test_5, 
         DL_y_test_5, DL_w_train_5, DL_w_test_5, DL_w_norm_train_5, DL_y_binary_5, 
         DL_N_train_5) = f.data_prep(DL_model_5, N_ttZ, N, weight, N_ggA_600_400, N_ggA_600_500)
    
        input_node = np.array([8])
        mid_node = np.array([8])
        extra_node = np.array([12])
        
        DL_epoch_5 = np.array([200])
        DL_batch_5 = np.array([5])
        DL_lr_5 = np.array([0.01,0.001,0.0005,0.0001])
        
        m.ML_opt(DL_X_train_5, DL_y_train_5, DL_X_test_5, DL_y_test_5, DL_w_train_5, DL_w_test_5, 
                 DL_w_norm_train_5, model_length,
           DL_epoch_5, DL_batch_5, DL_lr_5, 
           doES=True, ESpat=35,
           doRL=False, RLrate=0, RLpat=0,
           input_node=input_node, mid_node=mid_node, extra_node=extra_node,
           
           type_tag = ['DL', '600_500'])
        
    #############    
    # 500_400_1 #
    #############
     
    DL_model_6 = (lep1_pt,
                  lep2_pt,
                  lep3_pt, 
              
                    jet1_pt,
                    jet2_pt,
                    
                    bjet1_pt,
                    bjet2_pt,
              
                    met_pt
                    )

    # Prepare data for DL
    (DL_model_6_data, DL_model_6_data_norm, DL_X_train_6, DL_y_train_6, DL_X_test_6, 
     DL_y_test_6, DL_w_train_6, DL_w_test_6, DL_w_norm_train_6, DL_y_binary_6, 
     DL_N_train_6) = f.data_prep(DL_model_6, N_ttZ, N, weight, N_ggA_500_400, N_ggA_500_400_1,
                                                             cut_percent=0)
    
    model_length = len(DL_model_6)
    
    # Extract all X data for prediction
    DL_all_6 = DL_model_6_data_norm[:][:,0:model_length]
    
    # Define the learning rate
    DL_lr_6 = 0.001
    DL_epoch_6 = 200
    DL_batch_6 = 5
    
    input_node = 8# int(len(DL_X_train_2[0]))
    mid_node = 12#int(len(DL_X_train_2[0])*2)
    extra_node = 12#int(len(DL_X_train_2[0])*(3/2))
    
    ### The model ###
    DL_pred_6_train, DL_pred_6_test, DL_y_train_6, DL_y_test_6, DL_w_train_6, DL_w_test_6 = m.ML(
        DL_X_train_6, DL_y_train_6, DL_X_test_6, DL_y_test_6, model_length, 
        DL_w_train_6, DL_w_test_6, DL_w_norm_train_6, forceFit=ForceDL6, close=True, type_tag = ['DL', '500_400_1'],
                 
                 # Epochs batch and lr
                 epochs = DL_epoch_6, batch = DL_batch_6, lr = DL_lr_6,
                 
                 # Early stopping
                 doES=True, ESpat=30,
                 
                 # Learning rate reduction
                 doRL=False, RLrate=0.6, RLpat=16,
                 
                 # Nodes
                 input_node = input_node, mid_node = mid_node, extra_node=extra_node
                 )

    DL_uncut_test_AUC = f.ROC_Curve(DL_pred_6_train, DL_pred_6_test, DL_y_train_6, DL_y_test_6, close=True, 
                title=('DL $m_A$=500 $m_H$=400'), saveas=('DL/500_400_1/'))
    
    f.ProbHist(DL_pred_6_train, DL_pred_6_test, DL_y_train_6, DL_y_test_6, 
               DL_w_train_6, DL_w_test_6, 21, close=True, 
             label=['ttZ','ggA_500_400_1'], xtitle="Probability of signal", ytitle="Events", 
                title=('DL $m_A$=500 $m_H$=400'), saveas=('DL/500_400_1/'))
    
    DL_6_bkg_count, DL_6_sig_count = f.ProbLimitCount(DL_pred_6_train, DL_pred_6_test, DL_y_train_6, DL_y_test_6, 
               DL_w_train_6, DL_w_test_6, 21, close=True, 
             label=['ttZ','ggA_500_400_1'], xtitle="Probability of signal", ytitle="Events", 
                title=('DL $m_A$=500 $m_H$=400'), saveas=('DL/500_400_1/'))

    # Run optimisation if enabled
    if DL_Opt_6:
    
        (DL_model_6_data, DL_model_6_data_norm, DL_X_train_6, DL_y_train_6, DL_X_test_6, 
         DL_y_test_6, DL_w_train_6, DL_w_test_6, DL_w_norm_train_6, DL_y_binary_6, 
         DL_N_train_6) = f.data_prep(DL_model_6, N_ttZ, N, weight, N_ggA_500_400, N_ggA_500_400_1,
                                                             cut_percent=0)
    
        input_node = np.array([8])
        mid_node = np.array([8,12])
        extra_node = np.array([12])
        
        DL_epoch_6 = np.array([200])
        DL_batch_6 = np.array([5])
        DL_lr_6 = np.array([0.01,0.001,0.0001])
        
        m.ML_opt(DL_X_train_6, DL_y_train_6, DL_X_test_6, DL_y_test_6, DL_w_train_6, DL_w_test_6, 
                 DL_w_norm_train_6, model_length,
           DL_epoch_6, DL_batch_6, DL_lr_6, 
           doES=True, ESpat=35,
           doRL=False, RLrate=0, RLpat=0,
           input_node=input_node, mid_node=mid_node, extra_node=extra_node,
           
           type_tag = ['DL', '500_400_1'])

#############
# STAT TEST #
#############

### WARNING
### Statistical analysis is computationally heavy and runs/loads multiple models
doStat = True

ForceStat_SVM = False

ForceStat_SL = False

ForceStat_DL = False 

if doStat:
    
    try:
        N_uncut = np.load('Arrays/N_uncut.npy', allow_pickle=True)    
        SVM_test_AUC_mean = np.load('Arrays/SVM_test_AUC_mean.npy', allow_pickle=True)
        SVM_test_AUC_stdev = np.load('Arrays/SVM_test_AUC_stdev.npy', allow_pickle=True)   
        SL_test_AUC_mean = np.load('Arrays/SL_test_AUC_mean.npy', allow_pickle=True)
        SL_test_AUC_stdev = np.load('Arrays/SL_test_AUC_stdev.npy', allow_pickle=True)   
        DL_test_AUC_mean = np.load('Arrays/DL_test_AUC_mean.npy', allow_pickle=True)
        DL_test_AUC_stdev = np.load('Arrays/DL_test_AUC_stdev.npy', allow_pickle=True)
    
    except:
    
        # Number of trials
        cut_trials = 10
        
        # Define cuts and labels
        cut = np.array([0.25,0.5,0.75,0.9])
        
        # Define number of data points after cut
        N_uncut = (1-cut)*(N_ggA_500_400_1-N_ggA_500_400)
        
        SVM_test_AUC = np.zeros((len(cut), cut_trials))
        SL_test_AUC = np.zeros((len(cut), cut_trials))
        DL_test_AUC = np.zeros((len(cut), cut_trials))
        
        SVM_test_AUC_mean = np.zeros(len(cut))
        SL_test_AUC_mean = np.zeros(len(cut))
        DL_test_AUC_mean = np.zeros(len(cut))
        
        SVM_test_AUC_dev = np.zeros((len(cut), cut_trials))
        SL_test_AUC_dev = np.zeros((len(cut), cut_trials))
        DL_test_AUC_dev = np.zeros((len(cut), cut_trials))
        
        SVM_test_AUC_stdev = np.zeros(len(cut))
        SL_test_AUC_stdev = np.zeros(len(cut))
        DL_test_AUC_stdev = np.zeros(len(cut))
        
    
        # for i in range(len(cut)):
        #     # Create the label based on cut numbers
        #     cut_label.append('_Cut=' + str(int(cut[i]*100)) +'%')
            
        for i in range(len(cut)):
            for j in range(cut_trials):
                # Add j to the string to form different models for each trial
                #cut_label.append('_Cut=' + str(int(cut[i]*100)) +'%'+'_'+str(j))
                
                cut_label = '_Cut=' + str(int(cut[i]*100)) +'%'+'_'+str(j)
                
                ############
                # STAT SVM #
                ############        
        
                ### Model 6 ###
                model_6 = (delta_m, Z_pt, H_pt, M_ttZ, top_dr)
                
                # Prepare for SVM usage
                (model_6_data, model_6_data_norm, X_train_6, y_train_6, 
                 X_test_6, y_test_6, w_train_6, w_test_6, w_norm_train_6, y_binary_6, 
                 SVM_N_train_6) = f.data_prep(model_6, N_ttZ, N, weight, N_ggA_500_400, N_ggA_500_400_1, cut_percent=cut[i])
                
                y_train_6 = y_train_6.astype(int)
                
                # Build the SVM
                model_6_prob_train, model_6_prob_test = s.SVM(X_train_6, y_train_6, X_test_6, 
                                                              C=0.01, gamma=0.01, tol=0.1, tag='6'+cut_label, ForceModel=ForceStat_SVM)
                
                ### PLOTS 6 ###
                
                # ROC Curve
                SVM_test_AUC[i,j] = f.ROC_Curve(model_6_prob_train, model_6_prob_test, y_train_6, y_test_6, 
                            close=True, title="SVM_500_400_1"+cut_label, saveas='Statistics/SVM/500_400_1'+cut_label+'_')
                
                f.ProbHist(model_6_prob_train, model_6_prob_test, y_train_6, y_test_6, 
                           w_train_6, w_test_6, 21, close=True, 
                           label=['ttZ','ggA_500_400_1'], xtitle="Probability of signal", ytitle="Events", 
                        title="SVM_500_400_1"+cut_label, saveas='Statistics/SVM/500_400_1'+cut_label)
                
                # SVM_6_bkg_count, SVM_6_sig_count = f.ProbLimitCount(model_6_prob_train, model_6_prob_test, y_train_6, y_test_6, 
                #            w_train_6, w_test_6, 21, close=True, 
                #           label=['ttZ','ggA_500_400_1'], xtitle="Probability of signal", ytitle="Events", 
                #         title="SVM_500_400_1"+cut_label, saveas='Statistics/SVM/500_400_1'+cut_label)
            
                ###########
                # STAT SL #
                ###########
            
                # Prepare data for SL
                (SL_model_6_data, SL_model_6_data_norm, SL_X_train_6, SL_y_train_6, SL_X_test_6, 
                  SL_y_test_6, SL_w_train_6, SL_w_test_6, SL_w_norm_train_6, SL_y_binary_6, 
                  SL_N_train_6) = f.data_prep(SL_model_6, N_ttZ, N, weight, N_ggA_500_400, N_ggA_500_400_1,
                                                                         cut_percent=cut[i])
                
                model_length = len(SL_model_6)
                
                # Extract all X data for prediction
                SL_all_6 = SL_model_6_data_norm[:][:,0:model_length]
                
                input_node = 8#int(len(SL_X_train_6[0]))
                mid_node = 8#int(input_node*6)
                
                ### The model ###
                SL_pred_6_train, SL_pred_6_test, SL_y_train_6, SL_y_test_6, SL_w_train_6, SL_w_test_6 = m.ML(
                    SL_X_train_6, SL_y_train_6, SL_X_test_6, SL_y_test_6, model_length, 
                    SL_w_train_6, SL_w_test_6, SL_w_norm_train_6, forceFit=ForceStat_SL, close=True, type_tag = ['Statistics', '500_400_1'+cut_label,'SL'],
                    
                              # Epochs batch and lr
                              epochs = SL_epoch_6, batch = SL_batch_6, lr = SL_lr_6,
                             
                              # Early stopping
                              doES=True, ESpat=30,
                             
                              # Learning rate reduction
                              doRL=False, RLrate=0.1, RLpat=100,
                             
                              # Nodes
                              input_node = input_node, mid_node = mid_node)
            
                SL_test_AUC[i,j] = f.ROC_Curve(SL_pred_6_train, SL_pred_6_test, SL_y_train_6, SL_y_test_6, close=True, 
                            title=('SL_500_400_1'+cut_label), saveas=('Statistics/SL/500_400_1'+cut_label))
                
                f.ProbHist(SL_pred_6_train, SL_pred_6_test, SL_y_train_6, SL_y_test_6, 
                           SL_w_train_6, SL_w_test_6, 21, close=True, 
                          label=['ttZ','ggA_500_400_1'], xtitle="Probability", ytitle="Events", 
                            title=('SL_500_400_1'+cut_label), saveas=('Statistics/SL/500_400_1'+cut_label))
                
                # SL_6_bkg_count, SL_6_sig_count = f.ProbLimitCount(SL_pred_6_train, SL_pred_6_test, SL_y_train_6, SL_y_test_6, 
                #            SL_w_train_6, SL_w_test_6, 21, close=True, 
                #           label=['ttZ','ggA_500_400_1'], xtitle="Probability", ytitle="Events", 
                #             title=('SL_500_400_1'+cut_label), saveas=('Statistics/SL/'))
            
                ###########    
                # STAT DL #
                ###########
            
                # Prepare data for DL
                (DL_model_6_data, DL_model_6_data_norm, DL_X_train_6, DL_y_train_6, DL_X_test_6, 
                 DL_y_test_6, DL_w_train_6, DL_w_test_6, DL_w_norm_train_6, DL_y_binary_6, 
                 DL_N_train_6) = f.data_prep(DL_model_6, N_ttZ, N, weight, N_ggA_500_400, N_ggA_500_400_1,
                                                                         cut_percent=cut[i])
                model_length = len(DL_model_6)
                
                # Extract all X data for prediction
                DL_all_6 = DL_model_6_data_norm[:][:,0:model_length]
                
                input_node = 8# int(len(DL_X_train_2[0]))
                mid_node = 8#int(len(DL_X_train_2[0])*2)
                extra_node = 12#int(len(DL_X_train_2[0])*(3/2))
                
                ### The model ###
                DL_pred_6_train, DL_pred_6_test, DL_y_train_6, DL_y_test_6, DL_w_train_6, DL_w_test_6 = m.ML(
                    DL_X_train_6, DL_y_train_6, DL_X_test_6, DL_y_test_6, model_length, 
                    DL_w_train_6, DL_w_test_6, DL_w_norm_train_6, forceFit=ForceStat_DL, close=True, type_tag = ['Statistics', '500_400_1'+cut_label,'DL'],
                             
                             # Epochs batch and lr
                             epochs = DL_epoch_6, batch = DL_batch_6, lr = DL_lr_6,
                             
                             # Early stopping
                             doES=True, ESpat=30,
                             
                             # Learning rate reduction
                             doRL=False, RLrate=0.6, RLpat=16,
                             
                             # Nodes
                             input_node = input_node, mid_node = mid_node, extra_node=extra_node
                             )
    
                DL_test_AUC[i,j] = f.ROC_Curve(DL_pred_6_train, DL_pred_6_test, DL_y_train_6, DL_y_test_6, close=True, 
                            title=('DL_500_400_1'+cut_label), saveas=('Statistics/DL/500_400_1'+cut_label))
                
                f.ProbHist(DL_pred_6_train, DL_pred_6_test, DL_y_train_6, DL_y_test_6, 
                           DL_w_train_6, DL_w_test_6, 21, close=True, 
                         label=['ttZ','ggA_500_400_1'], xtitle="Probability", ytitle="Events", 
                            title=('DL_500_400_1'+cut_label), saveas=('Statistics/DL/500_400_1'+cut_label))
                
                # DL_6_bkg_count, DL_6_sig_count = f.ProbLimitCount(DL_pred_6_train, DL_pred_6_test, DL_y_train_6, DL_y_test_6, 
                #            DL_w_train_6, DL_w_test_6, 21, close=True, 
                #          label=['ttZ','ggA_500_400_1'], xtitle="Probability", ytitle="Events", 
                #             title=('DL_500_400_1'+cut_label[i]), saveas=('Statistics/DL/'))
                
                #!!! Do I need to include limit signal counts here?
        
            SVM_test_AUC_mean[i] = np.mean(SVM_test_AUC[i])
            SL_test_AUC_mean[i] = np.mean(SL_test_AUC[i])
            DL_test_AUC_mean[i] = np.mean(DL_test_AUC[i])
            
            for j in range(cut_trials):
                SVM_test_AUC_dev[i,j] = (SVM_test_AUC_mean[i]-SVM_test_AUC[i,j])**2
                SL_test_AUC_dev[i,j] = (SL_test_AUC_mean[i]-SL_test_AUC[i,j])**2
                DL_test_AUC_dev[i,j] = (DL_test_AUC_mean[i]-DL_test_AUC[i,j])**2
                
                
            # Should be divided by n-1 (bessels correction) but has little difference for n=10
            SVM_test_AUC_stdev[i] = np.sqrt(np.mean(SVM_test_AUC_dev[i]))
            SL_test_AUC_stdev[i] = np.sqrt(np.mean(SL_test_AUC_dev[i]))
            DL_test_AUC_stdev[i] = np.sqrt(np.mean(DL_test_AUC_dev[i]))
            
            np.save('Arrays/N_uncut.npy', N_uncut)    
            np.save('Arrays/SVM_test_AUC_mean.npy', SVM_test_AUC_mean)
            np.save('Arrays/SVM_test_AUC_stdev.npy', SVM_test_AUC_stdev)
            np.save('Arrays/SL_test_AUC_mean.npy', SL_test_AUC_mean)
            np.save('Arrays/SL_test_AUC_stdev.npy', SL_test_AUC_stdev)   
            np.save('Arrays/DL_test_AUC_mean.npy', DL_test_AUC_mean)
            np.save('Arrays/DL_test_AUC_stdev.npy', DL_test_AUC_stdev)   
        
        
    # # Add the uncut data from before for the plots
    # SVM_test_AUC = np.insert(SVM_test_AUC, 0, SVM_uncut_test_AUC, axis=0)
    # SL_test_AUC = np.insert(SL_test_AUC, 0, SL_uncut_test_AUC, axis=0)
    # DL_test_AUC = np.insert(DL_test_AUC, 0, DL_uncut_test_AUC, axis=0)
    # #N_uncut = np.insert(N_uncut, 0, (N_ggA_500_400_1-N_ggA_500_400), axis=0)
    # cut = np.insert(cut, 0, 0, axis=0)


    f.Line(N_uncut, SVM_test_AUC_mean, 'AUC', error=SVM_test_AUC_stdev, close=True, doString=False, xtitle='Number of signal events', 
            ytitle="AUC (Test data)", title='SVM AUC vs Events',saveas="SVM_AUC_Cut", x_lim=[0,4500], y_lim=[0,1])
    
    f.Line(N_uncut, SL_test_AUC_mean, 'AUC', error=SL_test_AUC_stdev,  close=True, doString=False, xtitle='Number of signal events', 
            ytitle="AUC (Test data)", title='SL AUC vs Events',saveas="SL_AUC_Cut", x_lim=[0,4500])
    
    f.Line(N_uncut, DL_test_AUC_mean, 'AUC', error=DL_test_AUC_stdev, close=True, doString=False, xtitle='Number of signal events', 
            ytitle="AUC (Test data)", title='DL AUC vs Events',saveas="DL_AUC_Cut", x_lim=[0,4500])


######################
#  LIMIT ESTIMATION  #
######################

doLimits = True
printLimits = False

### Ordering of signal count returned from plots 
# 0 - 460_360
# 1 - 500_360
# 2 - 600_360
# 3 - 600_400
# 4 - 600_500
# 5 - 500_400  - NOT USED
# 6 - 500_400_1

if doLimits:
    
    # A bar to separate prints
    textbar = '----------------------------------------------'
    
    # If plots aren't enabled load the previous cut based for limit calculations
    if not doPlots:
        delta_m_bkg_count = np.load('Arrays/delta_m_limit_bkg.npy', allow_pickle=True)
        delta_m_sig_count = np.load('Arrays/delta_m_limit_sig.npy', allow_pickle=True)
        
        Z_pt_bkg_count = np.load('Arrays/Z_limit_bkg.npy', allow_pickle=True)
        Z_pt_sig_count = np.load('Arrays/Z_limit_sig.npy', allow_pickle=True)
    
    
    ### Cut based limit calculations

    # 460_360
    limit_460_360_delta_m = f.getLimit(delta_m_bkg_count, delta_m_sig_count[0,:], 
                               confidenceLevel=0.95, method=0, err=0.05)
    limit_460_360_Z_pt = f.getLimit(Z_pt_bkg_count, Z_pt_sig_count[0,:], 
                               confidenceLevel=0.95, method=0, err=0.05)
    # 500_360
    limit_500_360_delta_m = f.getLimit(delta_m_bkg_count, delta_m_sig_count[1,:], 
                               confidenceLevel=0.95, method=0, err=0.05)
    limit_500_360_Z_pt = f.getLimit(Z_pt_bkg_count, Z_pt_sig_count[1,:], 
                               confidenceLevel=0.95, method=0, err=0.05)
    # 600_360
    limit_600_360_delta_m = f.getLimit(delta_m_bkg_count, delta_m_sig_count[2,:], 
                               confidenceLevel=0.95, method=0, err=0.05)
    limit_600_360_Z_pt = f.getLimit(Z_pt_bkg_count, Z_pt_sig_count[2,:], 
                               confidenceLevel=0.95, method=0, err=0.05)
    # 600_400
    limit_600_400_delta_m = f.getLimit(delta_m_bkg_count, delta_m_sig_count[3,:], 
                               confidenceLevel=0.95, method=0, err=0.05)
    limit_600_400_Z_pt = f.getLimit(Z_pt_bkg_count, Z_pt_sig_count[3,:], 
                               confidenceLevel=0.95, method=0, err=0.05)
    # 600_500
    limit_600_500_delta_m = f.getLimit(delta_m_bkg_count, delta_m_sig_count[4,:], 
                               confidenceLevel=0.95, method=0, err=0.05)
    limit_600_500_Z_pt = f.getLimit(Z_pt_bkg_count, Z_pt_sig_count[4,:], 
                               confidenceLevel=0.95, method=0, err=0.05)
    # 500_400_1
    limit_500_400_1_delta_m = f.getLimit(delta_m_bkg_count, delta_m_sig_count[6,:], 
                               confidenceLevel=0.95, method=0, err=0.05)
    limit_500_400_1_Z_pt = f.getLimit(Z_pt_bkg_count, Z_pt_sig_count[6,:], 
                               confidenceLevel=0.95, method=0, err=0.05)
    
    ### Model limit calculations
    if doSVM:
    
        limit_SVM_1 = f.getLimit(SVM_1_bkg_count, SVM_1_sig_count, confidenceLevel=0.95, method=0, err=0.05)
        limit_SVM_2 = f.getLimit(SVM_2_bkg_count, SVM_2_sig_count, confidenceLevel=0.95, method=0, err=0.05)
        limit_SVM_3 = f.getLimit(SVM_3_bkg_count, SVM_3_sig_count, confidenceLevel=0.95, method=0, err=0.05)
        limit_SVM_4 = f.getLimit(SVM_4_bkg_count, SVM_4_sig_count, confidenceLevel=0.95, method=0, err=0.05)
        limit_SVM_5 = f.getLimit(SVM_5_bkg_count, SVM_5_sig_count, confidenceLevel=0.95, method=0, err=0.05)
        limit_SVM_6 = f.getLimit(SVM_6_bkg_count, SVM_6_sig_count, confidenceLevel=0.95, method=0, err=0.05)
    
    if doSL:
        
        limit_SL_1 = f.getLimit(SL_1_bkg_count, SL_1_sig_count, confidenceLevel=0.95, method=0, err=0.05)
        limit_SL_2 = f.getLimit(SL_2_bkg_count, SL_2_sig_count, confidenceLevel=0.95, method=0, err=0.05)
        limit_SL_3 = f.getLimit(SL_3_bkg_count, SL_3_sig_count, confidenceLevel=0.95, method=0, err=0.05)
        limit_SL_4 = f.getLimit(SL_4_bkg_count, SL_4_sig_count, confidenceLevel=0.95, method=0, err=0.05)
        limit_SL_5 = f.getLimit(SL_5_bkg_count, SL_5_sig_count, confidenceLevel=0.95, method=0, err=0.05)
        limit_SL_6 = f.getLimit(SL_6_bkg_count, SL_6_sig_count, confidenceLevel=0.95, method=0, err=0.05)
    
    
    if doDL:

        limit_DL_1 = f.getLimit(DL_1_bkg_count, DL_1_sig_count, confidenceLevel=0.95, method=0, err=0.05)
        limit_DL_2 = f.getLimit(DL_2_bkg_count, DL_2_sig_count, confidenceLevel=0.95, method=0, err=0.05)
        limit_DL_3 = f.getLimit(DL_3_bkg_count, DL_3_sig_count, confidenceLevel=0.95, method=0, err=0.05)
        limit_DL_4 = f.getLimit(DL_4_bkg_count, DL_4_sig_count, confidenceLevel=0.95, method=0, err=0.05)
        limit_DL_5 = f.getLimit(DL_5_bkg_count, DL_5_sig_count, confidenceLevel=0.95, method=0, err=0.05)
        limit_DL_6 = f.getLimit(DL_6_bkg_count, DL_6_sig_count, confidenceLevel=0.95, method=0, err=0.05)
        
    ### Print statements
    if printLimits:
        print('\n' + textbar + '\nCut-based Limits\n' + textbar)
        print('460_360 delta m limit is: ', limit_460_360_delta_m)
        print('460_360 Z_pt limit is: ', limit_460_360_Z_pt)
        print('500_360 delta m limit is: ', limit_500_360_delta_m)
        print('500_360 Z_pt limit is: ', limit_500_360_Z_pt)
        print('600_360 delta m limit is: ', limit_600_360_delta_m)
        print('600_360 Z_pt limit is: ', limit_600_360_Z_pt)
        print('600_400 delta m limit is: ', limit_600_400_delta_m)
        print('600_400 Z_pt limit is: ', limit_600_400_Z_pt)
        print('600_500 delta m limit is: ', limit_600_500_delta_m)
        print('600_500 Z_pt limit is: ', limit_600_500_Z_pt)
        print('500_400_1 delta m limit is: ', limit_500_400_1_delta_m)
        print('500_400_1 Z_pt limit is: ', limit_500_400_1_Z_pt)
    
        if doSVM:
            print('\n' + textbar + '\nSVM Limits\n' + textbar)
            print('460_360 SVM limit is: ', limit_SVM_1)
            print('500_360 SVM limit is: ', limit_SVM_2)
            print('600_360 SVM limit is: ', limit_SVM_3)
            print('600_400 SVM limit is: ', limit_SVM_4)
            print('600_500 SVM limit is: ', limit_SVM_5)
            print('500_400_1 SVM limit is: ', limit_SVM_6)
            
        if doSL:
            print('\n' + textbar + '\nSL Limits\n' + textbar)
            print('460_360 SL limit is: ', limit_SL_1)
            print('500_360 SL limit is: ', limit_SL_2)
            print('600_360 SL limit is: ', limit_SL_3)
            print('600_400 SL limit is: ', limit_SL_4)
            print('600_500 SL limit is: ', limit_SL_5)
            print('500_400_1 SL limit is: ', limit_SL_6)
            
        if doDL:
            print('\n' + textbar + '\nDL Limits\n' + textbar)
            print('460_360 DL limit is: ', limit_DL_1)
            print('500_360 DL limit is: ', limit_DL_2)
            print('600_360 DL limit is: ', limit_DL_3)
            print('600_400 DL limit is: ', limit_DL_4)
            print('600_500 DL limit is: ', limit_DL_5)
            print('500_400_1 DL limit is: ', limit_DL_6)
    
        if doSL and doDL and doSVM:
            print('\n' + textbar + '\n460_360 Limits\n' + textbar)
            print('delta_m: ',limit_460_360_delta_m)
            print('Z_pt: ',limit_460_360_Z_pt)
            print('SVM: ',limit_SVM_1)
            print('SL: ',limit_SL_1)
            print('DL: ',limit_DL_1)
            
            print('\n' + textbar + '\n500_360 Limits\n' + textbar)
            print('delta_m: ',limit_500_360_delta_m)
            print('Z_pt: ',limit_500_360_Z_pt)
            print('SVM: ',limit_SVM_2)
            print('SL: ',limit_SL_2)
            print('DL: ',limit_DL_2)
        
            print('\n' + textbar + '\n600_360 Limits\n' + textbar)
            print('delta_m: ',limit_600_360_delta_m)
            print('Z_pt: ',limit_600_360_Z_pt)
            print('SVM: ',limit_SVM_3)
            print('SL: ',limit_SL_3)
            print('DL: ',limit_DL_3)
            
            print('\n' + textbar + '\n600_400 Limits\n' + textbar)
            print('delta_m: ',limit_600_400_delta_m)
            print('Z_pt: ',limit_600_400_Z_pt)
            print('SVM: ',limit_SVM_4)
            print('SL: ',limit_SL_4)
            print('DL: ',limit_DL_4)
        
            print('\n' + textbar + '\n600_500 Limits\n' + textbar)
            print('delta_m: ',limit_600_500_delta_m)
            print('Z_pt: ',limit_600_500_Z_pt)
            print('SVM: ',limit_SVM_5)
            print('SL: ',limit_SL_5)
            print('DL: ',limit_DL_5)
            
            print('\n' + textbar + '\n500_400_1 Limits\n' + textbar)
            print('delta_m: ',limit_500_400_1_delta_m)
            print('Z_pt: ',limit_500_400_1_Z_pt)
            print('SVM: ',limit_SVM_6)
            print('SL: ',limit_SL_6)
            print('DL: ',limit_DL_6)
        
    #######
    # 360 #
    #######
    
    # Sensitivity arrays
    delta_m_sens_360 = np.array([limit_460_360_delta_m, limit_500_360_delta_m, limit_600_360_delta_m])
    Z_pt_sens_360 = np.array([limit_460_360_Z_pt, limit_500_360_Z_pt, limit_600_360_Z_pt])
    SVM_sens_360 = np.array([limit_SVM_1,limit_SVM_2,limit_SVM_3])
    SL_sens_360 = np.array([limit_SL_1,limit_SL_2,limit_SL_3])
    DL_sens_360 = np.array([limit_DL_1,limit_DL_2,limit_DL_3])
    
    sens_360 = (delta_m_sens_360,Z_pt_sens_360,SVM_sens_360,SL_sens_360,DL_sens_360)
    
    mA_360 = (460, 500, 600)
    label_360 = [r'$\Delta$m',r'$Z_{p_T}$','SVM', 'SL','DL']
    
    f.Line(mA_360, sens_360, label_360, close=True, xtitle=r'$m_A$ (GeV)', 
           ytitle=r'$\sigma$ $\times$ BR(A $\rightarrow$ ZH $\rightarrow$ llt$\bar{t}$)', 
           title='Upper Limit $m_H$=360', saveas="Upper_Limit_mH=360")

    #######
    # 400 #
    #######
    
    # Sensitivity arrays
    delta_m_sens_400 = np.array([limit_500_400_1_delta_m, limit_600_400_delta_m])
    Z_pt_sens_400 = np.array([limit_500_400_1_Z_pt, limit_600_400_Z_pt])
    SVM_sens_400 = np.array([limit_SVM_6,limit_SVM_4])
    SL_sens_400 = np.array([limit_SL_6,limit_SL_4])
    DL_sens_400 = np.array([limit_DL_6,limit_DL_4])
    
    sens_400 = (delta_m_sens_400,Z_pt_sens_400,SVM_sens_400,SL_sens_400,DL_sens_400)
    
    mA_400 = (500, 600)
    label_400 = [r'$\Delta$m',r'$Z_{p_T}$','SVM', 'SL','DL']
    
    f.Line(mA_400, sens_400, label_400, close=True, xtitle=r'$m_A$ (GeV)', 
           ytitle=r'$\sigma$ $\times$ BR(A $\rightarrow$ ZH $\rightarrow$ llt$\bar{t}$)', 
           title='Upper Limit $m_H$=400', saveas="Upper_Limit_mH=400")

###############
### Runtime ###
###############

print('\nSVM 460 Runtime: {:.2f} seconds'.format(SVM_460_time))
print('\nSL 460 Runtime: {:.2f} seconds'.format(SL_460_time))
print('\nDL 460 Runtime: {:.2f} seconds'.format(DL_460_time))

runtime_x = np.array([0,1,2])
runtime_y = np.array([SVM_460_time,SL_460_time,DL_460_time])
runtime_label = 'Time'

# Plot the runtime of SVM, SL and DL
f.Line(runtime_x, runtime_y, runtime_label, close=True, doString=False, RuntimePlot=True, leg_loc='upper left',
       xtitle='Machine learning type', ytitle='Runtime (s)', title='Runtime Comparison', saveas="Runtime")

print('\nRuntime: {:.2f} seconds'.format(time.time() - start_time))


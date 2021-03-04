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
plt.close('all')

#################
#  DATA IMPORT  #
#################

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
    N = N_arr[9]

    lep12_inv_mass = np.load('Arrays/lep12_inv_mass.npy')
    lep13_inv_mass = np.load('Arrays/lep13_inv_mass.npy')
    lep23_inv_mass = np.load('Arrays/lep23_inv_mass.npy')
    
    lep1_pt = np.load('Arrays/lep1_pt.npy', allow_pickle=True)
    lep2_pt = np.load('Arrays/lep2_pt.npy', allow_pickle=True)
    lep3_pt = np.load('Arrays/lep3_pt.npy', allow_pickle=True)

    jet1_pt = np.load('Arrays/jet1_pt.npy', allow_pickle=True)
    jet2_pt = np.load('Arrays/jet2_pt.npy', allow_pickle=True)
    
    bjet1_pt = np.load('Arrays/bjet1_pt.npy', allow_pickle=True)
    bjet2_pt = np.load('Arrays/bjet2_pt.npy', allow_pickle=True)
    
    Z_pt = np.load('Arrays/Z_pt.npy')
    delta_m = np.load('Arrays/delta_m.npy')
    
    tops_angle = np.load('Arrays/tops_angle.npy')
    met_pt = np.load('Arrays/met_pt.npy', allow_pickle=True)
    Wm_mass = np.load('Arrays/Wm_mass.npy')
    Wp_mass = np.load('Arrays/Wp_mass.npy')
    bjet12_angle = np.load('Arrays/bjet12_angle.npy')
    jet12_angle = np.load('Arrays/jet12_angle.npy')
    lep12_angle = np.load('Arrays/lep12_angle.npy')
    lep3_neu_angle = np.load('Arrays/lep3_neu_angle.npy')
    neu_four_mom = np.load('Arrays/neu_four_mom.npy')
    ztt_m = np.load('Arrays/ztt_m.npy')
    M_ttZ = np.load('Arrays/M_ttZ.npy')
    
    weight = np.load('Arrays/weight.npy', allow_pickle=True) 
    
    print('Arrays succesfully loaded')

# If loading fails run the calculations:
except:

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
    
    'ttZ'
    ztt_M = d.ExtractVariable(dataset,'ztt', 1, 'm')
    tt_M = d.ExtractVariable(dataset,'tt', 1, 'm')
    
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
    tops_angle = np.zeros(N)
    lep12_angle = np.zeros(N)
    lep3_neu_angle = np.zeros(N)
    bjet12_angle = np.zeros(N)
    jet12_angle = np.zeros(N)
    
    delta_m_actual = np.zeros(N)
    ztt_m = np.zeros(N)
    
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
        
        # An arbritrary loop to make use of the break function
        for j in range(1):
            
            # if all lep combinations are viable test to see which is best
            if lep12_viable and lep13_viable and lep23_viable:
                if Zm_diff_12[i] < Zm_diff_13[i] and Zm_diff_12[i] < Zm_diff_23[i]:
                    #print(i, ': 12 is the answer')
                    Z_pt [i] = np.sqrt(((lep12_four_mom [i,1])**2)+((lep12_four_mom [i,2])**2))
                    break
                elif Zm_diff_13[i] < Zm_diff_23[i]:
                    #print(i, ': 13 is the answer')
                    Z_pt [i] = np.sqrt(((lep13_four_mom [i,1])**2)+((lep13_four_mom [i,2])**2))
                    break
                else:
                    #print(i, ': 23 is the answer')
                    Z_pt [i] = np.sqrt(((lep23_four_mom [i,1])**2)+((lep23_four_mom [i,2])**2))
                    break
            
            # if only 2 are viable check which one is best
            elif lep12_viable and lep13_viable:
                if Zm_diff_12[i] < Zm_diff_13[i]:
                    Z_pt [i] = np.sqrt(((lep12_four_mom [i,1])**2)+((lep12_four_mom [i,2])**2))
                    #print(i, ': 12 is the answer')
                    break
                else:
                    Z_pt [i] = np.sqrt(((lep13_four_mom [i,1])**2)+((lep13_four_mom [i,2])**2))
                    #print(i, ': 13 is the answer')
                    break
                
            elif lep12_viable and lep23_viable:
                if Zm_diff_12[i] < Zm_diff_23[i]:
                    Z_pt [i] = np.sqrt(((lep12_four_mom [i,1])**2)+((lep12_four_mom [i,2])**2))
                    #print(i, ': 12 is the answer')
                    break
                else:
                    Z_pt [i] = np.sqrt(((lep23_four_mom [i,1])**2)+((lep23_four_mom [i,2])**2))
                    #print(i, ': 23 is the answer')
                    break
                
            elif lep13_viable and lep23_viable:
                if Zm_diff_13[i] < Zm_diff_23[i]:
                    Z_pt [i] = np.sqrt(((lep13_four_mom [i,1])**2)+((lep13_four_mom [i,2])**2))
                    #print(i, ': 13 is the answer')
                    break
                else:
                    Z_pt [i] = np.sqrt(((lep23_four_mom [i,1])**2)+((lep23_four_mom [i,2])**2))
                    #print(i, ': 23 is the answer')
                    break
                
            # if only one path is viable choose it and print Z_diff if it is large
            elif lep12_viable:
                #print(i, ': 12 is the answer')
                if Zm_diff_12[i]>15:   # Max is just above 10 GeV
                    print('Z diff is large: ', Zm_diff_12[i])
                else:
                    Z_pt [i] = np.sqrt(((lep12_four_mom [i,1])**2)+((lep12_four_mom [i,2])**2))
                break
            
            elif lep13_viable:
                #print(i, ': 13 is the answer')
                if Zm_diff_13[i]>15:
                    print('Z diff is large: ', Zm_diff_13[i])
                else:
                    Z_pt [i] = np.sqrt(((lep13_four_mom [i,1])**2)+((lep13_four_mom [i,2])**2))
                break
            
            elif lep23_viable:
                #print(i, ': 23 is the answer')
                if Zm_diff_23[i]>15:
                    print('Z diff is large: ', Zm_diff_23[i])
                else:
                    Z_pt [i] = np.sqrt(((lep23_four_mom [i,1])**2)+((lep23_four_mom [i,2])**2))
                break
            
            else:
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
                    
                # Adjust the main N value
                N = N-1
        
        #!!! Add description
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
        tops_angle[i] = f.angle(top1_four_mom[i],top2_four_mom[i])
        lep12_angle[i] = f.angle(lep1_four_mom[i],lep2_four_mom[i])
        neu_four_mom[i,0] = (f.mom(neu_four_mom[i,1],neu_four_mom[i,2],neu_four_mom[i,3]))**2
        lep3_neu_angle[i] = f.angle(lep3_four_mom[i],neu_four_mom[i])
        bjet12_angle[i] = f.angle(bjet1_four_mom[i],bjet2_four_mom[i])
        jet12_angle[i] = f.angle(jet1_four_mom[i],jet2_four_mom[i])
        ztt_m[i] = ztt_M[i]
    
    # Remove non-viable decays
    lep1_pt = lep1_pt[Z_delete==0]
    lep2_pt = lep2_pt[Z_delete==0]
    lep3_pt = lep3_pt[Z_delete==0]
    jet1_pt = jet1_pt[Z_delete==0]
    jet2_pt = jet2_pt[Z_delete==0]
    bjet1_pt = bjet1_pt[Z_delete==0]
    bjet2_pt = bjet2_pt[Z_delete==0]
    Z_pt = Z_pt[Z_delete==0]
    delta_m = delta_m[Z_delete==0]
    tops_angle = tops_angle[Z_delete==0]
    met_pt = met_pt[Z_delete==0]
    Wm_mass = Wm_mass[Z_delete==0]
    Wp_mass = Wp_mass[Z_delete==0]
    bjet12_angle = bjet12_angle[Z_delete==0]
    jet12_angle = jet12_angle[Z_delete==0]
    lep12_angle = lep12_angle[Z_delete==0]
    lep3_neu_angle = lep3_neu_angle[Z_delete==0]
    neu_four_mom = neu_four_mom[Z_delete==0]
    ztt_m = ztt_m[Z_delete==0]
    M_ttZ = M_ttZ[Z_delete==0]
    weight = weight[Z_delete==0]
    
    # Make an array of the new N values (excluding the ignored values)
    N_arr = np.array([N_ttZ,N_ttWm,N_ttWp,N_ggA_460_360,N_ggA_500_360,N_ggA_600_360,
                      N_ggA_600_400,N_ggA_600_500,N_ggA_500_400,N])
    
    # Save the arrays for efficiency
    np.save('Arrays/N_arr.npy', N_arr)
    np.save('Arrays/lep12_inv_mass.npy', lep12_inv_mass)
    np.save('Arrays/lep13_inv_mass.npy', lep13_inv_mass) 
    np.save('Arrays/lep23_inv_mass.npy', lep23_inv_mass) 
    np.save('Arrays/lep1_pt.npy', lep1_pt)     
    np.save('Arrays/lep2_pt.npy', lep2_pt)     
    np.save('Arrays/lep3_pt.npy', lep3_pt)     
    np.save('Arrays/jet1_pt.npy', jet1_pt)     
    np.save('Arrays/jet2_pt.npy', jet2_pt)     
    np.save('Arrays/bjet1_pt.npy', bjet1_pt)     
    np.save('Arrays/bjet2_pt.npy', bjet2_pt)     
    np.save('Arrays/Z_pt.npy', Z_pt)     
    np.save('Arrays/delta_m.npy', delta_m)     
    np.save('Arrays/tops_angle.npy', tops_angle)     
    np.save('Arrays/met_pt.npy', met_pt)     
    np.save('Arrays/Wm_mass.npy', Wm_mass)     
    np.save('Arrays/Wp_mass.npy', Wp_mass)     
    np.save('Arrays/bjet12_angle.npy', bjet12_angle)     
    np.save('Arrays/jet12_angle.npy', jet12_angle)     
    np.save('Arrays/lep12_angle.npy', lep12_angle)     
    np.save('Arrays/lep3_neu_angle.npy', lep3_neu_angle)     
    np.save('Arrays/neu_four_mom.npy', neu_four_mom)     
    np.save('Arrays/ztt_m.npy', ztt_m)     
    np.save('Arrays/M_ttZ.npy', M_ttZ)     
    np.save('Arrays/weight.npy', weight)     

###########
#  PLOTS  #
###########

'Singular histograms'

f.Hist(lep12_inv_mass, "Selected di-lepton invariant mass", 21, close=True, label='lep1-lep2',
     xtitle="$m_{ll}$ (GeV)", ytitle="Events", title="Di-lepton (1-2) invariant mass")

f.Hist(lep13_inv_mass, "Di-lepton (1-3) invariant mass", 20, close=True,  label='lep13',
     xtitle="m (GeV)", ytitle="Events", title="Di-lepton (1-3) invariant mass")

f.Hist(lep23_inv_mass, "Di-lepton (2-3) invariant mass", 20, close=True,  label='lep23',
     xtitle="m (GeV)", ytitle="Events", title="Di-lepton (2-3) invariant mass")

# f.Hist(jet12_inv_mass, "Di-jet (1-2) invariant mass", 20, close=True,  label='jet12',
#      xtitle="m (GeV)", ytitle="Events", title="Di-jet (1-2) invariant mass", xmax=160, xmin=0)


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

# Z boson plot
f.SignalHist(Z_pt, weight, 25, N_arr, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="$Z_{pT}$",
             saveas="Z_pT", signals = select_signals, line = select_line)

# delta m plot
delta_m_bkg_count, delta_m_sig_count = f.SignalHist(delta_m, weight, 26, N_arr, close=True, xtitle="$\Delta$m (GeV)", ytitle="Events", title="$\Delta$m",
             saveas="delta_m", signals = all_signals, line = all_line, scale='log', xlim=[0, 2000])

'Testing discriminating variables plots'

# delta m discrim
f.SignalHist(delta_m, weight, 26, N_arr, close=True, xtitle="$\Delta$m (GeV)", ytitle="Events", title="$\Delta$m",
             saveas="disc_delta_m", signals = select_signals, 
             normed=True, line = select_line, xlim=[0, 1000])

# tops angle plot
f.SignalHist(tops_angle, weight, 25, N_arr, close=True, xtitle=r't$\bar{t}$ angle (rad)', ytitle="rad", title=r't$\bar{t}$ angle',
             saveas="disc_tt_angle", signals = select_signals, 
             normed=True, line = select_line)

# met pt plot
f.SignalHist(met_pt, weight, 25, N_arr, close=True, xtitle=r'$E^{T}_{miss}$ (GeV)', ytitle="Events", title=r'$E^{T}_{miss}$',
             saveas="disc_met_pt", signals = select_signals, 
             normed = True, line = select_line, xlim=[0, 600])

# lep12 angle plot
f.SignalHist(lep12_angle, weight, 25, N_arr, close=True, xtitle=r'lep1-lep2 angle (rad)', ytitle="rad", title=r'lep1-lep2 angle',
             saveas="disc_lep12_angle", signals = select_signals, 
             normed = True, line = select_line)

# bjet12 angle plot
f.SignalHist(bjet12_angle, weight, 25, N_arr, close=True, xtitle=r'bjet1-bjet2 angle (rad)', ytitle="rad", 
             title=r'bjet1-bjet2 angle',
             saveas="disc_bjet12_angle", signals = select_signals, 
             normed = True, line = select_line)

# Wp mass plot
f.SignalHist(Wp_mass, weight, 26, N_arr, close=True, xtitle=r'$W^+$ mass (GeV)', ytitle="Events", title=r'$W^+$ mass',
             saveas="disc_Wp_mass", signals = select_signals, 
             normed = True, line = select_line, xlim=[0, 1000])

# Wm mass plot
f.SignalHist(Wm_mass, weight, 25, N_arr, close=True, xtitle=r'$W^-$ mass (GeV)', ytitle="Events", title=r'$W^-$ mass',
             saveas="disc_Wm_mass", signals = select_signals, 
             normed = True, line = select_line, xlim=[0, 600])

# Neutrino pz plot
f.SignalHist(neu_four_mom[:,3], weight, 26, N_arr, close=True, xtitle=r'Neutrino $p_Z$ (GeV)', ytitle="Events", title=r'Neutrino $p_Z$',
             saveas="disc_Neutrino_pz", signals = select_signals, 
             normed = True, line = select_line, xlim=[-500, 500])

# Z pt plot
f.SignalHist(Z_pt, weight, 25, N_arr, close=True, xtitle=r'$p_T$ (GeV)', ytitle="Events", title=r'$Z_{pT}$',
             saveas="disc_Z_pT", signals = select_signals, 
             normed = True, line = select_line, xlim=[0, 600])

# lep3-neu angle plot
f.SignalHist(lep3_neu_angle, weight, 25, N_arr, close=True, xtitle=r'lep3-neutrino angle (rad)', ytitle="rad", title=r'lep3-neutrino angle',
             saveas="disc_lep3_neu_angle", signals = select_signals, 
             normed = True, line = select_line, xlim=[0.7, 0.9])

# jet12 angle plot
f.SignalHist(jet12_angle, weight, 25, N_arr, close=True, xtitle=r'jet1-jet2 angle (rad)', ytitle="rad", title=r'jet1-jet2 angle',
             saveas="disc_jet12_angle", signals = select_signals, 
             normed = True, line = select_line)

# ztt_m plot
f.SignalHist(ztt_m, weight, 26, N_arr, close=True, xtitle=r'ztt_m (GeV)', ytitle="Events", title=r'ztt_m',
             saveas="disc_ztt_m", signals = select_signals, 
             normed = True, line = select_line, xlim=[0, 4000])

# M_ttZ plot
f.SignalHist(M_ttZ, weight, 26, N_arr, close=True, xtitle=r'ttZ mass (GeV)', ytitle="Events", title=r'ttZ mass',
             saveas="disc_ttZ_m", signals = select_signals, 
             normed = True, line = select_line)


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

# COULD ROC CURVES OF SIGNAL VS BKG HELP?

    ###
    # C = 15 gamma = 20 tol = 1

if doSVM:
    
    ###########
    # 600_500 #
    ###########
    
    ### Model 1 ###
    
    # Define the model
    model_1 = (delta_m, Z_pt, tops_angle, Wp_mass, bjet12_angle, jet12_angle, lep12_angle) # Best 0.7 AUC
    
    # Attempt 2
    #model_1 = (delta_m, Z_pt, tops_angle, lep12_angle, Wm_mass, Wp_mass, bjet12_angle)
    
    # Just m and Z
    #model_1 = (delta_m, Z_pt)
    
    # Without angles
    #model_1 = (delta_m, Z_pt, Wm_mass, Wp_mass, M_ttZ, ztt_m)
    
    # ALL
    #model_1 = (delta_m, Z_pt, tops_angle, lep12_angle, Wm_mass, Wp_mass, bjet12_angle, M_ttZ, 
    #           jet12_angle, lep3_neu_angle, met_pt)
    
    # Prepare for SVM usage
    model_1_data, model_1_data_norm, X_train_1, y_train_1, X_test_1, y_test_1, y_binary_1 = f.data_prep(model_1, N_ttZ, N, N_ggA_600_400, N_ggA_600_500)
    
    # Build the SVM
    model_1_prob_train, model_1_prob_test = s.SVM(X_train_1, y_train_1, X_test_1, 
                                                  C=200, gamma=1E-7, tol=1, tag='1', ForceModel=True)
    
    # Combine the output probabilities
    model_1_all_prob = np.concatenate((model_1_prob_train,model_1_prob_test))
    
    ### PLOTS 1 ###
    
    # ROC Curve
    f.ROC_Curve(y_binary_1, model_1_all_prob, close=False, tag='1')
    
    # All data
    svm_bkg_count, svm_sig_count = f.SVMHist(y_binary_1, model_1_all_prob, 21, weight, N_arr, close=False,
                    label=['ttZ','ggA_600_500'], xtitle="Probability of signal", ytitle="Events", 
                        title="Model_1_600_500_all", saveas='Model_1_600_500_all')
    
    # Train data
    f.SVMHist(y_train_1, model_1_prob_train, 21, weight, N_arr, close=False,
              label=['ttZ','ggA_600_500'], xtitle="Probability of signal", ytitle="Events", 
            title="Model_1_600_500_train", saveas='Model_1_600_500_train')
    
    # Test data
    f.SVMHist(y_test_1, model_1_prob_test, 21, weight, N_arr, close=False, 
              label=['ttZ','ggA_600_500'], xtitle="Probability of signal", ytitle="Events", 
              title="Model_1_600_500_test", saveas='Model_1_600_500_test')
    
    ###########
    # 600_360 #
    ###########
    
    ### Model 2 ###
    
    model_2 = (delta_m,bjet12_angle,met_pt,lep12_angle, Z_pt, tops_angle)
    
    model_2_data, model_2_data_norm, X_train_2, y_train_2, X_test_2, y_test_2, y_binary_2 = f.data_prep(model_2, N_ttZ, N, N_ggA_500_360, N_ggA_600_360)
    
    model_2_prob_train, model_2_prob_test = s.SVM(X_train_2, y_train_2, X_test_2, 
                                                  C=15, gamma=15, tol=1E-3, tag='2', ForceModel=False)

    model_2_prob = np.concatenate((model_2_prob_train,model_2_prob_test))
    
    ### Plots 2 ###
    f.ROC_Curve(y_binary_2, model_2_prob, close=True, tag='2')
    
    f.SVMHist(y_binary_2, model_2_prob, 21, weight, N_arr, close=True, label=['ttZ','ggA_600_360'], 
            xtitle="Probability of signal", ytitle="Events", title="Model_2_600_360", saveas='Model_2_600_360_all')
    
    f.SVMHist(y_train_2, model_2_prob_train, 21, weight, N_arr, close=True,
              label=['ttZ','ggA_600_360'], xtitle="Probability of signal", ytitle="Events", 
            title="Model_2_600_360_train", saveas='Model_2_600_360_train')
    
    f.SVMHist(y_test_2, model_2_prob_test, 21, weight, N_arr, close=True, 
              label=['ttZ','ggA_600_360'], xtitle="Probability of signal", ytitle="Events", 
              title="Model_2_600_360_test", saveas='Model_2_600_360_test')
    
    ###########
    # 500_360 #
    ###########
    
    ### Model 3 ###
    
    model_3 = (delta_m,bjet12_angle,met_pt,lep12_angle, Z_pt, tops_angle)
    
    model_3_data, model_3_data_norm, X_train_3, y_train_3, X_test_3, y_test_3, y_binary_3 = f.data_prep(model_3, N_ttZ, N, N_ggA_460_360, N_ggA_500_360)
    
    model_3_prob_train, model_3_prob_test = s.SVM(X_train_3, y_train_3, X_test_3, 
                                                  C=15, gamma=15, tol=1E-3, tag='3', ForceModel=False)
    
    model_3_prob = np.concatenate((model_3_prob_train,model_3_prob_test))
    
    ### Plots 3 ###
    f.ROC_Curve(y_binary_3, model_3_prob, close=True, tag='3')
    
    f.SVMHist(y_binary_3, model_3_prob, 21, weight, N_arr, close=True, label=['ttZ','ggA_500_360'], 
            xtitle="Probability of signal", ytitle="Events", title="Model_3_500_360", saveas='Model_3_500_360_all')
    
    f.SVMHist(y_train_3, model_3_prob_train, 21, weight, N_arr, close=True,
              label=['ttZ','ggA_500_360'], xtitle="Probability of signal", ytitle="Events", 
            title="Model_3_500_360_train", saveas='Model_3_500_360_train')
    
    f.SVMHist(y_test_3, model_3_prob_test, 21, weight, N_arr, close=True, 
              label=['ttZ','ggA_500_360'], xtitle="Probability of signal", ytitle="Events", 
              title="Model_3_500_360_test", saveas='Model_3_500_360_test')

    ###########
    # 500_400 #
    ###########
    
    ### Model 4 ###
        
    model_4 = (delta_m, Z_pt, tops_angle, Wm_mass, Wp_mass, bjet12_angle, jet12_angle, M_ttZ, ztt_m)
    
    model_4_data, model_4_data_norm, X_train_4, y_train_4, X_test_4, y_test_4, y_binary_4 = f.data_prep(model_4, N_ttZ, N, N_ggA_600_500, N_ggA_500_400)
    
    model_4_prob_train, model_4_prob_test = s.SVM(X_train_4, y_train_4, X_test_4, 
                                                  C=100, gamma=6, tol=1, tag='4', ForceModel=False)
    
    model_4_prob = np.concatenate((model_4_prob_train,model_4_prob_test))

    ### Plots 4 ###

    f.ROC_Curve(y_binary_4, model_4_prob, close=True, tag='4')
    
    f.SVMHist(y_binary_4, model_4_prob, 21, weight, N_arr, close=True, label=['ttZ','ggA_500_400'], 
            xtitle="Probability of signal", ytitle="Events", title="Model_4_500_400", saveas='Model_4_500_400')
    
    f.SVMHist(y_train_4, model_4_prob_train, 21, weight, N_arr, close=True,
              label=['ttZ','ggA_600_500'], xtitle="Probability of signal", ytitle="Events", 
            title="Model_4_500_400_train", saveas='Model_4_500_400_train')
    
    f.SVMHist(y_test_4, model_4_prob_test, 21, weight, N_arr, close=True, 
              label=['ttZ','ggA_500_400'], xtitle="Probability of signal", ytitle="Events", 
              title="Model_4_500_400_test", saveas='Model_4_500_400_test')


#####################
#  SHALLOW NETWORK  #
#####################


doML = False

if doML:
    model_length = len(model_1)
    model = m.ML(X_train_1, y_train_1, X_test_1, y_test_1, model_length, doFit=True)
    
    # !!! This is also defined in f.dataprep, needs sorting out
    N = model_1_data.shape[0]
    N_train = int(2*N/3)
    
    # Take test data, split into background and signal
    data_test = model_1_data_norm[N_train:,:]
    x_bkg = data_test[  data_test[:,model_length] == 0  ][:,0:model_length] 
    x_sig = data_test[  data_test[:,model_length] > 0 ][:,0:model_length]
    x_all = model_1_data_norm[:][:,0:model_length]
    
    all_pred = model.predict(x_all)
    res_sig = model.predict(x_sig)
    res_bkg = model.predict(x_bkg)
    
    f.makePlot(res_bkg.flatten(),res_sig.flatten(), "nn_pred", 20, False, xtitle="NN output", title="All sample")

######################
#  LIMIT ESTIMATION  #
######################

limit_svm = f.getLimit(svm_sig_count, svm_bkg_count, confidenceLevel=0.95, method=0, err=0.05)
limit_delta_m = f.getLimit(delta_m_sig_count, delta_m_bkg_count, confidenceLevel=0.95, method=0, err=0.05)

print('SVM limit is: ', limit_svm)
print('delta m limit is: ', limit_delta_m)

#limit_m = f.approxLimit(sig_count,bkg_count)

###############
### Runtime ###
###############

print('Runtime: {:.2f} seconds'.format(time.time() - start_time))

# weight_other = weight[N_ttZ:N_ttWp][0]
# weight_ttZ = weight[0:N_ttZ][0]
# weight_460_360 = weight[N_ttWp:N_ggA_460_360][0]
# weight_500_360 = weight[N_ggA_460_360:N_ggA_500_360][0]
# weight_600_360 = weight[N_ggA_500_360:N_ggA_600_360][0]
# weight_600_400 = weight[N_ggA_600_360:N_ggA_600_400][0]
# weight_600_500 = weight[N_ggA_600_400:N_ggA_600_500][0]
# weight_500_400 = weight[N_ggA_600_500:N_ggA_500_400][0]

# weight1 = weight_other*len(weight_other)


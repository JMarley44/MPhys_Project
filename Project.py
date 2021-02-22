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

'''
DATA IMPORT
'''
# Import and manipulation
path = "C:/Users/James/Desktop/Liverpool/Year 4/PHYS498 - Project/Python files/MPhys_Project/dataset_semileptonic.csv"
dataimport = pd.read_csv(path, header=None)
dataset_full = pd.DataFrame(dataimport).to_numpy()
dataset = np.delete(dataset_full,0,1)

# Find the length of the states
N_ttZ,N_ttWm,N_ttWp,N_ggA_460_360,N_ggA_500_360,N_ggA_600_360,N_ggA_600_400,N_ggA_600_500,N_ggA_500_400  = d.DataSplit(dataset_full)

# Length of the dataset
N = len(dataset)

'''
CONSTANTS & COUNTS
'''

W_mass = 80.38 # GeV
top_mass = 172.76 # GeV

'''VARIABLE EXTRACTION'''

'Weights'
weight = d.ExtractVariable(dataset,'weight',1,1)

'Leptons'
# Whole data set
lep1_pt = d.ExtractVariable(dataset,'lep', 1, 'pt')
lep1_eta = d.ExtractVariable(dataset,'lep', 1, 'eta')
lep1_phi = d.ExtractVariable(dataset,'lep', 1, 'phi')

lep2_pt = d.ExtractVariable(dataset,'lep', 2, 'pt')
lep2_eta = d.ExtractVariable(dataset,'lep', 2, 'eta')
lep2_phi = d.ExtractVariable(dataset,'lep', 2, 'phi')

lep3_pt = d.ExtractVariable(dataset,'lep', 3, 'pt')
lep3_eta = d.ExtractVariable(dataset,'lep', 3, 'eta')
lep3_phi = d.ExtractVariable(dataset,'lep', 3, 'phi')

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

'''
EMPTY ARRAYS
'''
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

delta_m_actual = np.zeros(N)
ztt_m = np.zeros(N)

'''MAIN LOOP'''

for i in range(N):
    
    'Leptons'
    # Four momenta
    lep1_four_mom [i,:] = f.four_mom(lep1_pt[i], lep1_eta[i], lep1_phi[i])
    lep2_four_mom [i,:] = f.four_mom(lep2_pt[i], lep2_eta[i], lep2_phi[i])
    lep3_four_mom [i,:] = f.four_mom(lep3_pt[i], lep3_eta[i], lep3_phi[i])
    
    # Addition of four momenta
    lep12_four_mom [i,:] = lep1_four_mom[i,:]+lep2_four_mom[i,:]        # This is the Z boson four vector
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

    '''
    #RECONSTRUCTIONS
    '''    

    'Z boson'
    # lep12 is identical to the Z boson four momenta
    Z_pt [i] = np.sqrt(((lep12_four_mom [i,1])**2)+((lep12_four_mom [i,2])**2))
    
    #!!! Add description
    'Wm boson'
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
    ztt_m[i] = ztt_M[i]
    

'''
PLOTS
'''

'Singular histograms'

f.Hist(lep12_inv_mass, "Di-lepton (1-2) invariant mass", 20, close=True, label='lep12',
     xtitle="m (GeV)", ytitle="Events", title="Di-lepton (1-2) invariant mass")

f.Hist(lep13_inv_mass, "Di-lepton (1-3) invariant mass", 20, close=True,  label='lep13',
     xtitle="m (GeV)", ytitle="Events", title="Di-lepton (1-3) invariant mass")

f.Hist(lep23_inv_mass, "Di-lepton (2-3) invariant mass", 20, close=True,  label='lep23',
     xtitle="m (GeV)", ytitle="Events", title="Di-lepton (2-3) invariant mass")

f.Hist(jet12_inv_mass, "Di-jet (1-2) invariant mass", 20, close=True,  label='jet12',
     xtitle="m (GeV)", ytitle="Events", title="Di-jet (1-2) invariant mass", xmax=160, xmin=0)


'Stacked signal histograms'

###########     ttZ         other
# Signals #     460_360    500_360    600_360
###########     600_400    600_500    500_400

all_signals = ['ttZ', 'other','460_360', '500_360', '600_360', '600_400', '600_500', '500_400']
select_signals = ['ttZ', 'other','500_360', '600_360', '600_500']

all_line = ['-',':','--','--','-',':']
select_line = ['--',':','--','--','-','--']

# Plots with all signals
f.SignalHist(lep1_pt, weight, 25, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="Lepton 1 $p_T$",
             saveas="Lepton1_pt", signals = select_signals, line = select_line, scale='log')

f.SignalHist(lep2_pt, weight, 25, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="Lepton 2 $p_T$",
             saveas="Lepton2_pt", signals = select_signals, line = select_line, scale='log')

f.SignalHist(lep3_pt, weight, 25, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="Lepton 3 $p_T$",
             saveas="Lepton3_pt", signals = select_signals, line = select_line, scale='log')

f.SignalHist(jet1_pt, weight, 25, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="Jet1 $p_T$",
             saveas="Jet1_pt", signals = select_signals, line = select_line, scale='log')

f.SignalHist(jet2_pt, weight, 25, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="Jet2 $p_T$",
             saveas="Jet2_pt", signals = select_signals, line = select_line, scale='log')

f.SignalHist(bjet1_pt, weight, 25, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="bJet1 $p_T$",
             saveas="bJet1_pt", signals = select_signals, line = select_line, scale='log')

f.SignalHist(bjet2_pt, weight, 25, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="bJet2 $p_T$",
             saveas="bJet2_pt", signals = select_signals, line = select_line, scale='log')

# Z boson plot
f.SignalHist(Z_pt, weight, 25, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="$Z_{pT}$",
             saveas="Z_pT", signals = select_signals, line = select_line, scale='log')

# delta m plot
delta_m_bkg_count, delta_m_sig_count = f.SignalHist(delta_m, weight, 25, close=True, xtitle="$\Delta$m (GeV)", ytitle="Events", title="$\Delta$m",
             saveas="delta_m", signals = all_signals, line = all_line, scale='log')

'Testing discriminating variables plots'

# delta m discrim
f.SignalHist(delta_m, weight, 25, close=True, xtitle="$\Delta$m (GeV)", ytitle="Events", title="$\Delta$m",
             saveas="disc_delta_m", signals = select_signals, 
             normed=True, line = select_line, xlim=[0, 1000])

# tops angle plot
f.SignalHist(tops_angle, weight, 25, close=True, xtitle=r't$\bar{t}$ angle (rad)', ytitle="rad", title=r't$\bar{t}$ angle',
             saveas="disc_tt_angle", signals = select_signals, 
             normed=True, line = select_line)

# met pt plot
f.SignalHist(met_pt, weight, 25, close=True, xtitle=r'$E^{T}_{miss}$ (GeV)', ytitle="Events", title=r'$E^{T}_{miss}$',
             saveas="disc_met_pt", signals = select_signals, 
             normed = True, line = select_line, xlim=[0, 600])

# lep12 angle plot
f.SignalHist(lep12_angle, weight, 25, close=True, xtitle=r'lep1-lep2 angle (rad)', ytitle="rad", title=r'lep1-lep2 angle',
             saveas="disc_lep12_angle", signals = select_signals, 
             normed = True, line = select_line)

# bjet12 angle plot
f.SignalHist(bjet12_angle, weight, 25, close=True, xtitle=r'bjet1-bjet2 angle (rad)', ytitle="rad", 
             title=r'bjet1-bjet2 angle',
             saveas="disc_bjet12_angle", signals = select_signals, 
             normed = True, line = select_line)

# Wp mass plot
f.SignalHist(Wp_mass, weight, 25, close=True, xtitle=r'$W^+$ mass (GeV)', ytitle="Events", title=r'$W^+$ mass',
             saveas="disc_Wp_mass", signals = select_signals, 
             normed = True, line = select_line, xlim=[0, 1000])

# Wm mass plot - doesnt show much
f.SignalHist(Wm_mass, weight, 25, close=True, xtitle=r'$W^-$ mass (GeV)', ytitle="Events", title=r'$W^-$ mass',
             saveas="disc_Wm_mass", signals = select_signals, 
             normed = True, line = select_line, xlim=[0, 600])

# Neutrino pz plot
f.SignalHist(neu_four_mom[:,3], weight, 25, close=True, xtitle=r'Neutrino $p_Z$ (GeV)', ytitle="Events", title=r'Neutrino $p_Z$',
             saveas="disc_Neutrino_pz", signals = select_signals, 
             normed = True, line = select_line, xlim=[-500, 500])

# Z pt plot
f.SignalHist(Z_pt, weight, 25, close=True, xtitle=r'$p_T$ (GeV)', ytitle="Events", title=r'$Z_{pT}$',
             saveas="disc_Z_pT", signals = select_signals, 
             normed = True, line = select_line, xlim=[0, 600])

# lep3-neu angle plot
f.SignalHist(lep3_neu_angle, weight, 25, close=True, xtitle=r'lep3-neutrino angle (rad)', ytitle="rad", title=r'lep3-neutrino angle',
             saveas="disc_lep3_neu_angle", signals = select_signals, 
             normed = True, line = select_line)

# lep3-neu angle plot
f.SignalHist(ztt_m, weight, 25, close=True, xtitle=r'ztt_m (GeV)', ytitle="Events", title=r'ztt_m',
             saveas="disc_ztt_m", signals = select_signals, 
             normed = True, line = select_line)



#############################
'''Machine learning aspect'''
#############################

'DATA PREP'
# ggA_500_360 - [N_ggA_460_360:N_ggA_500_360]
# ggA_600_360 - [N_ggA_500_360:N_ggA_600_360]       length = 1092
# ggA_600_400 - [N_ggA_600_360:N_ggA_600_400]
# ggA_600_500 - [N_ggA_600_400:N_ggA_600_500]       length = 1372
# ggA_500_400 - [N_ggA_600_500:N_ggA_500_400]

# Ordering of the signal types
# N_ttZ,N_ttWm,N_ttWp,N_ggA_460_360,N_ggA_500_360,N_ggA_600_360,N_ggA_600_400,N_ggA_600_500,N_ggA_500_400

# First model - delta_m, Z_pt, tops_angle, lep12_angle, Wm_mass, Wp_mass, bjet12_angle
model_1 = (delta_m, Z_pt, tops_angle, Wm_mass, Wp_mass, bjet12_angle)

model_2 = (delta_m,bjet12_angle,met_pt,lep12_angle, Z_pt, tops_angle)

# 600 500
model_1_data, model_1_data_norm, X_train_1, y_train_1, X_test_1, y_test_1, y_binary_1 = f.data_prep(model_1, N_ttZ, N_ggA_600_400, N_ggA_600_500)

# 600 360
model_2_data, model_2_data_norm, X_train_2, y_train_2, X_test_2, y_test_2, y_binary_2 = f.data_prep(model_2, N_ttZ, N_ggA_500_360, N_ggA_600_360)

'''SUPPORT VECTOR MACHINE'''
SVM_optim = False

#!!! MAKE ROC CURVES OF SIGNAL VS BKG

# Hyper-parameter optimisation
if SVM_optim:
    s.SVM_optim(X_train_1, X_test_1, y_train_1, y_test_1)

'600 500 signal'
# First model
model_1_prob_train, model_1_prob_test = s.SVM(X_train_1, y_train_1, X_test_1, 
                                              C=0.1, gamma= 20, tag='1', ForceModel=True)

# TRAINING IS STILL HIGH, SHOULD I LOWER C? IT MOVES THE CENTRE OF THE SIGNAL BUT SOMETHING ELSE COULD BE AT PLAY
# C=0.001, gamma = 25 AUC is 0.68
# C=0.001, gamma = 22 AUC is 0.71
# GAMMA UP
# C=0.001, gamma = 15 AUC is 0.68
# C=0.001, gamma = 17 AUC is 0.67
# C=0.001, gamma = 18 AUC is 0.66
# C=0.001, gamma = 19 AUC is 0.69
# START WITH GAMMA DOWN
# TRY CHANGING GAMMA FOR 0.7 centre
# C=0.0001, gamma = 20 similar to previous
# C=0.0005, gamma = 20 AUC is 0.67, signal centre has moved ~0.5, perhaps change gamma, lets go one step further
# C=0.001, gamma = 20 AUC is 0.7, BUT train is down to 0.99 and better separation BUT lower signal centered ~ 0.7
# C=0.01, gamma = 20 AUC is 0.81
# C=0.1, gamma = 20 AUC is 0.82
# C=0.1, gamma = 20 AUC is 0.81
# C=1, gamma = 20 AUC is 0.81
# COULD WE BE OVERFITTING??
# C=10, gamma = 20 AUC is 0.82
# C=15, gamma = 20 AUC is 0.8
# C=15, gamma = 15 AUC is 0.8 still a large peak near 0.2 but larger centered on 0.8/0.9
# C=15, gamma = 10 AUC is 0.78 much better signal peak but it is central around 0.7/0.8
# C=15, gamma = 0.01 AUC is 0.661 larger signal peak near background
# C=10, gamma = 0.01 AUC is 0.69 larger signal peak near background

# Combine the probabilities
model_1_prob = np.concatenate((model_1_prob_train,model_1_prob_test))

# Round to obtain the predicted value
model_1_pred = np.around(model_1_prob)

# Plots
f.ROC_Curve(y_binary_1, model_1_prob, '1')

svm_bkg_count, svm_sig_count = f.SVMHist(y_binary_1, model_1_prob, 20, close=False, label=['ttZ','ggA_600_500'], 
        xtitle="Probability of signal", ytitle="Events", title="Model_1_600_500", saveas='SVM_Hist_600_500')

'''
'600 360 signal'
# Second model
model_2_prob_train, model_2_prob_test = s.SVM(X_train_2, y_train_2, X_test_2, 
                                              C=1, gamma=20, tag='2', ForceModel=True)

# C=30, gamma = 20 AUC is 0.82
# C=30, gamma = 25 AUC is 0.82
# C=25, gamma = 20 AUC is 0.83
# C=6, gamma = 20 AUC is 0.825
# C=10, gamma = 20 higher AUC 0.835 could be a fluct
# C=17, gamma = 20 higher AUC 0.85 could be a fluct
# C=20, gamma = 50 lower AUC
# C=17, gamma = 50 better ROC AUC, same as 15
# C=16, gamma = 50 lower ROC AUC
# C=15, gamma = 50 no diff on last
# C=15, gamma = 30 no diff on last
# C=15, gamma = 25 "
# C=15, gamma = 20 maybe some better peaks at 0 and 1
# C=15, gamma = 15 no considerable difference
# C=15, gamma = 13 no considerable difference
# C=15, gamma = 12 no considerable difference
# C=15, gamma = 10 better,
# C=15, gamma = 7 even better!
# C=15, gamma = 5 test is getting better
# C=15, gamma=2  -  High train , low test
# C=15, gamma=1  -  High train, decent test
# C=15, gamma=0.5  -  Decent train and test
# started analysing C, found its best to be about 15 (sequence of trials goes up)

# Combine the probabilities
model_2_prob = np.concatenate((model_2_prob_train,model_2_prob_test))

# Round to obtain the predicted value
model_2_pred = np.around(model_2_prob)

# Plots
f.ROC_Curve(y_binary_2, model_2_prob, '2')

f.SVMHist(y_binary_2, model_2_prob, 20, close=False, label=['ttZ','ggA_600_360'], 
        xtitle="Probability of signal", ytitle="Events", title="Model_2_600_360", saveas='SVM_Hist_600_360')

'''

'''SHALLOW NETWORK'''

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

'''LIMIT ESTIMATION'''

limit_svm = f.getLimit(svm_sig_count, svm_bkg_count, confidenceLevel=0.95, method=0, err=0.05)
limit_delta_m = f.getLimit(delta_m_sig_count, delta_m_bkg_count, confidenceLevel=0.95, method=0, err=0.05)



#limit_m = f.approxLimit(sig_count,bkg_count)

###############
### Runtime ###
###############

print('Runtime: {:.2f} seconds'.format(time.time() - start_time))


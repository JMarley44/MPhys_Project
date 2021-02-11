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

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.close('all')

'''
#Data management
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
#Constants
'''

W_mass = 80.38 # GeV
top_mass = 172.76 # GeV

'''Variable extraction'''

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

'''
#Empty array definition
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

'''
#Calculations
'''

'Main loop'
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
    #Reconstructions
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
    

'''
#Plots
'''

'Singular histograms'

f.Hist(lep12_inv_mass, "Di-lepton (1-2) invariant mass", 20, close=True, label='lep12',
     xtitle="$p_T$ (GeV)", ytitle="Events", title="Di-lepton (1-2) invariant mass")

f.Hist(lep13_inv_mass, "Di-lepton (1-3) invariant mass", 20, close=True,  label='lep13',
     xtitle="$p_T$ (GeV)", ytitle="Events", title="Di-lepton (1-3) invariant mass")

f.Hist(lep23_inv_mass, "Di-lepton (2-3) invariant mass", 20, close=True,  label='lep23',
     xtitle="$p_T$ (GeV)", ytitle="Events", title="Di-lepton (2-3) invariant mass")

f.Hist(jet12_inv_mass, "Di-jet (1-2) invariant mass", 20, close=True,  label='jet12',
     xtitle="$p_T$ (GeV)", ytitle="Events", title="Di-jet (1-2) invariant mass", xmax=160, xmin=0)


'Stacked signal histograms'

###########     ttZ         other
# Signals #     460_360    500_360    600_360
###########     600_400    600_500    500_400

# Plots with all signals
f.SignalHist(lep1_pt, weight, 25, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="Lepton 1 $p_T$",
             saveas="Lepton1_pt", signals = ['ttZ', 'other','460_360', '500_360', '600_360', '600_400', '600_500', '500_400'])

f.SignalHist(lep2_pt, weight, 25, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="Lepton 2 $p_T$",
             saveas="Lepton2_pt", signals = ['ttZ', 'other','460_360', '500_360', '600_360', '600_400', '600_500', '500_400'])

f.SignalHist(lep3_pt, weight, 25, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="Lepton 3 $p_T$",
             saveas="Lepton3_pt", signals = ['ttZ', 'other','460_360', '500_360', '600_360', '600_400', '600_500', '500_400'])

f.SignalHist(jet1_pt, weight, 25, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="Jet1 $p_T$",
             saveas="Jet1_pt", signals = ['ttZ', 'other','460_360', '500_360', '600_360', '600_400', '600_500', '500_400'])

f.SignalHist(jet2_pt, weight, 25, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="Jet2 $p_T$",
             saveas="Jet2_pt", signals = ['ttZ', 'other','460_360', '500_360', '600_360', '600_400', '600_500', '500_400'])

f.SignalHist(bjet1_pt, weight, 25, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="bJet1 $p_T$",
             saveas="bJet1_pt", signals = ['ttZ', 'other','460_360', '500_360', '600_360', '600_400', '600_500', '500_400'])

f.SignalHist(bjet2_pt, weight, 25, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="bJet2 $p_T$",
             saveas="bJet2_pt", signals = ['ttZ', 'other','460_360', '500_360', '600_360', '600_400', '600_500', '500_400'])

# Z boson plot
f.SignalHist(Z_pt, weight, 25, close=True, xtitle="$p_T$ (GeV)", ytitle="Events", title="$Z_{pT}$",
             saveas="Z_pT", signals = ['ttZ', 'other','460_360', '500_360', '600_360', '600_400', '600_500', '500_400'])

# delta m plot
f.SignalHist(delta_m, weight, 25, close=True, xtitle="$\Delta$m (GeV)", ytitle="Events", title="$\Delta$m",
             saveas="delta_m", signals = ['ttZ', 'other','460_360', '500_360', '600_360', '600_400', '600_500', '500_400'])

'Testing discriminating variables plots'

#!!! Need to change ytitle on radian plots (Its not in: Events/GeV)

# delta m discrim
f.SignalHist(delta_m, weight, 25, close=False, xtitle="$\Delta$m (GeV)", ytitle="Events", title="$\Delta$m",
             saveas="disc_delta_m", signals = ['ttZ', 'other','500_360', '600_360', '600_500'])

# tops angle plot
f.SignalHist(tops_angle, weight, 25, close=False, xtitle=r't$\bar{t}$ angle (rad)', ytitle="Events", title=r't$\bar{t}$ angle',
             saveas="disc_tt_angle", signals = ['ttZ', 'other','500_360', '600_360', '600_500'])

# met pt plot
f.SignalHist(met_pt, weight, 25, close=False, xtitle=r'$E^{T}_{miss}$ (GeV)', ytitle="Events", title=r'$E^{T}_{miss}$',
             saveas="disc_met_pt", signals = ['ttZ', 'other','500_360', '600_360', '600_500'])

# lep12 angle plot
f.SignalHist(lep12_angle, weight, 25, close=False, xtitle=r'lep1-lep2 angle (rad)', ytitle="Events", title=r'lep1-lep2 angle',
             saveas="disc_lep12_angle", signals = ['ttZ', 'other','500_360', '600_360', '600_500'])

# Wp mass plot
f.SignalHist(Wp_mass, weight, 25, close=False, xtitle=r'$W^+$ mass (GeV)', ytitle="Events", title=r'$W^+$ mass',
             saveas="disc_Wp_mass", signals = ['ttZ', 'other','500_360', '600_360', '600_500'])

# Wm mass plot
f.SignalHist(Wm_mass, weight, 25, close=False, xtitle=r'$W^-$ mass (GeV)', ytitle="Events", title=r'$W^-$ mass',
             saveas="disc_Wm_mass", signals = ['ttZ', 'other','500_360', '600_360', '600_500'])

# Neutrino pz plot
f.SignalHist(neu_four_mom[:,3], weight, 25, close=False, xtitle=r'Neutrino $p_Z$ (GeV)', ytitle="Events", title=r'Neutrino $p_Z$',
             saveas="disc_Neutrino_pz", signals = ['ttZ', 'other','500_360', '600_360', '600_500'])

'''Support vector machine'''


'Data prep and models'

# Ordering of the signal types
# N_ttZ,N_ttWm,N_ttWp,N_ggA_460_360,N_ggA_500_360,N_ggA_600_360,N_ggA_600_400,N_ggA_600_500,N_ggA_500_400

# First model
model_1 = (delta_m,tops_angle,Z_pt,Wp_mass,lep12_angle)

model_1_data = f.data_prep(model_1, N_ttZ, N_ttWp, N_ggA_460_360)

model_1_prob_train,model_1_prob_test = s.SVM(model_1_data, '1')

model_1_prob = np.concatenate((model_1_prob_train,model_1_prob_test))


# Round to obtain the predicted value
model_1_pred = np.around(model_1_prob)

# f.ROC_Curve(model_1_data, model_1_pred, N_ttZ, '1')

# f.Hist(model_1_prob, "Output", 20, close=False, label='Prob',
#      xtitle="Probability of signal", ytitle="Events", title="Model_1", xmin=0, xmax=1, scale='log')



###############
### Runtime ###
###############

print('Runtime: {:.2f} seconds'.format(time.time() - start_time))


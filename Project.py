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

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.close('all')

'''
#Data management
'''

# Import and manipulation
path = "C:/Users/James/Documents/Liverpool/Year 4/PHYS498 - Project/Python files/MPhys_Project/dataset_semileptonic.csv"
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
met_pz = np.zeros(N)

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
    #!!! Add description
    tt_four_mom[i] = top1_four_mom[i] + top2_four_mom[i]
    ttZ_four_mom[i] = tt_four_mom[i] + lep12_four_mom[i]
    
    M_tt [i] = f.inv_mass(tt_four_mom[i])
    M_ttZ[i] = f.inv_mass(ttZ_four_mom[i])
    delta_m[i] = M_ttZ[i] - M_tt[i]
'''
#Pre-plot modifications
'''

'State separations'
# weights
ttZ_weight = weight[0:N_ttZ]
ttWm_weight = weight[N_ttZ:N_ttWm]
ttWp_weight = weight[N_ttWm:N_ttWp]
other_weight = weight[N_ttZ:N_ttWp]
ggA_460_360_weight = weight[N_ttWp:N_ggA_460_360]

# ttZ - [0:N_ttZ]
lep1_ttZ_pt = lep1_pt[0:N_ttZ]
lep2_ttZ_pt = lep2_pt[0:N_ttZ]
lep3_ttZ_pt = lep3_pt[0:N_ttZ]

jet1_ttZ_pt = jet1_pt[0:N_ttZ]
jet2_ttZ_pt = jet2_pt[0:N_ttZ]

bjet1_ttZ_pt = bjet1_pt[0:N_ttZ]
bjet2_ttZ_pt = bjet2_pt[0:N_ttZ]

Z_ttZ_pt = Z_pt[0:N_ttZ]

delta_m_ttZ = delta_m[0:N_ttZ]

# other - [N_ttZ:N_ttWp]
lep1_other_pt = lep1_pt[N_ttZ:N_ttWp]
lep2_other_pt = lep2_pt[N_ttZ:N_ttWp]
lep3_other_pt = lep3_pt[N_ttZ:N_ttWp]

jet1_other_pt = jet1_pt[N_ttZ:N_ttWp]
jet2_other_pt = jet2_pt[N_ttZ:N_ttWp]

bjet1_other_pt = bjet1_pt[N_ttZ:N_ttWp]
bjet2_other_pt = bjet2_pt[N_ttZ:N_ttWp]

Z_other_pt = Z_pt[N_ttZ:N_ttWp]

delta_m_other = delta_m[N_ttZ:N_ttWp]

# ggA_460_360 - [N_ttWp:N_ggA_460_360]
lep1_ggA_460_360_pt = lep1_pt[N_ttWp:N_ggA_460_360]
lep2_ggA_460_360_pt = lep2_pt[N_ttWp:N_ggA_460_360]
lep3_ggA_460_360_pt = lep3_pt[N_ttWp:N_ggA_460_360]

jet1_ggA_460_360_pt = jet1_pt[N_ttWp:N_ggA_460_360]
jet2_ggA_460_360_pt = jet2_pt[N_ttWp:N_ggA_460_360]

bjet1_ggA_460_360_pt = bjet1_pt[N_ttWp:N_ggA_460_360]
bjet2_ggA_460_360_pt = bjet2_pt[N_ttWp:N_ggA_460_360]

Z_ggA_460_360_pt = Z_pt[N_ttWp:N_ggA_460_360]

delta_m_ggA_460_360 = delta_m[N_ttWp:N_ggA_460_360]

'Array combinations'
# Lepton pt's
lep1_pt_plot = ([lep1_other_pt,lep1_ttZ_pt])
lep2_pt_plot = ([lep2_other_pt,lep2_ttZ_pt])
lep3_pt_plot = ([lep3_other_pt,lep3_ttZ_pt])

# jet pt's
jet1_pt_plot = ([jet1_other_pt,jet1_ttZ_pt])
jet2_pt_plot = ([jet2_other_pt,jet2_ttZ_pt])

# bjet pt's
bjet1_pt_plot = ([bjet1_other_pt,bjet1_ttZ_pt])
bjet2_pt_plot = ([bjet2_other_pt,bjet2_ttZ_pt])

# Z
Z_pt_arr = ([Z_other_pt,Z_ttZ_pt])

# delta m
delta_m_arr = ([delta_m_other,delta_m_ttZ])

'Labels, colours and weights'
label = [r'other',r't$\bar{t}$Z','ggA ($m_A$=460, $m_H$=360)']
color = ['lightgreen','cornflowerblue','red']
weights = ([other_weight, ttZ_weight, ggA_460_360_weight])

'''
#Plots
'''

'Singular histograms'

f.Hist(lep12_inv_mass, "Di-lepton (1-2) invariant mass", 20, close=True, label='lep12',
     xtitle="$p_T$ (GeV)", ytitle="Events (#)", title="Di-lepton (1-2) invariant mass")

f.Hist(lep13_inv_mass, "Di-lepton (1-3) invariant mass", 20, close=True,  label='lep13',
     xtitle="$p_T$ (GeV)", ytitle="Events (#)", title="Di-lepton (1-3) invariant mass")

f.Hist(lep23_inv_mass, "Di-lepton (2-3) invariant mass", 20, close=True,  label='lep23',
     xtitle="$p_T$ (GeV)", ytitle="Events (#)", title="Di-lepton (2-3) invariant mass")

f.Hist(jet12_inv_mass, "Di-jet (1-2) invariant mass", 20, close=True,  label='jet12',
     xtitle="$p_T$ (GeV)", ytitle="Events (#)", title="Di-jet (1-2) invariant mass", xmax=160, xmin=0)


'Stacked signal histograms'

# Plots for input variables
f.SignalHist(lep1_pt_plot, lep1_ggA_460_360_pt, weights, 25, close=True,  
             label=label, color=color, xtitle="$p_T$ (GeV)", ytitle="Events", title="Lepton 1 $p_T$", 
             saveas="Lepton1_pt")

f.SignalHist(lep2_pt_plot, lep2_ggA_460_360_pt, weights, 25, close=True,  
             label=label, color=color, xtitle="$p_T$ (GeV)", ytitle="Events", title="Lepton 2 $p_T$", 
             saveas="Lepton2_pt")

f.SignalHist(lep3_pt_plot, lep3_ggA_460_360_pt, weights, 25, close=True,  
             label=label, color=color, xtitle="$p_T$ (GeV)", ytitle="Events", title="Lepton 3 $p_T$", 
             saveas="Lepton3_pt")

f.SignalHist(jet1_pt_plot, jet1_ggA_460_360_pt, weights, 25, close=True,  
             label=label, color=color, xtitle="$p_T$ (GeV)", ytitle="Events", title="Jet1 $p_T$", 
             saveas="Jet1_pt")

f.SignalHist(jet2_pt_plot, jet2_ggA_460_360_pt, weights, 25, close=True,  
             label=label, color=color, xtitle="$p_T$ (GeV)", ytitle="Events", title="Jet2 $p_T$", 
             saveas="Jet2_pt")

f.SignalHist(bjet1_pt_plot, bjet1_ggA_460_360_pt, weights, 25, close=True,  
             label=label, color=color, xtitle="$p_T$ (GeV)", ytitle="Events", title="bJet1 $p_T$", 
             saveas="bJet1_pt")

f.SignalHist(bjet2_pt_plot, bjet2_ggA_460_360_pt, weights, 25, close=True,  
             label=label, color=color, xtitle="$p_T$ (GeV)", ytitle="Events", title="bJet2 $p_T$", 
             saveas="bJet2_pt")

# Z boson plot
f.SignalHist(Z_pt_arr, Z_ggA_460_360_pt, weights, 25, close=True,  
             label=label, color=color, xtitle="$p_T$ (GeV)", ytitle="Events (#)", title="$Z_{pT}$",
             saveas="Z_pT")

# delta m plot
f.SignalHist(delta_m_arr, delta_m_ggA_460_360, weights, 25, close=False,  
             label=label, color=color, xtitle="$\Delta$m (GeV)", ytitle="Events (#)", title="$\Delta$m",
             saveas="delta_m")

print('Runtime: {:.2f} seconds'.format(time.time() - start_time))
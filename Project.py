# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 19:55:22 2020

@author: James
"""
import pandas as pd
import numpy as np
import DataExtract as d
import Functions as f

'''Data management'''

# Import and manipulation
path = "C:/Users/James/Documents/Liverpool/Year 4/PHYS498 - Project/Python files/MPhys_Project/dataset_semileptonic.csv"
dataimport = pd.read_csv(path, header=None)
dataset_full = pd.DataFrame(dataimport).to_numpy()
dataset = np.delete(dataset_full,0,1)

# Find the length of the states
N_ttZ,N_ttWm,N_ttWp,N_ggA_460_360,N_ggA_500_360,N_ggA_600_360,N_ggA_600_400,N_ggA_600_500,N_ggA_500_400  = d.DataSplit()
# Length of the dataset
N = len(dataset)

'''Constants'''

W_mass = 80.38 # GeV
count = 0

'''Variable extraction'''

'Weights'
weight = d.ExtractVariable('weight',1,1)

'Leptons'
# Whole data set
lep1_pt = d.ExtractVariable('lep', 1, 'pt')
lep1_eta = d.ExtractVariable('lep', 1, 'eta')
lep1_phi = d.ExtractVariable('lep', 1, 'phi')

lep2_pt = d.ExtractVariable('lep', 2, 'pt')
lep2_eta = d.ExtractVariable('lep', 2, 'eta')
lep2_phi = d.ExtractVariable('lep', 2, 'phi')

lep3_pt = d.ExtractVariable('lep', 3, 'pt')
lep3_eta = d.ExtractVariable('lep', 3, 'eta')
lep3_phi = d.ExtractVariable('lep', 3, 'phi')

'Jets'
jet1_pt = d.ExtractVariable('jet', 1, 'pt')
jet1_eta = d.ExtractVariable('jet', 1, 'eta')
jet1_phi = d.ExtractVariable('jet', 1, 'phi')

jet2_pt = d.ExtractVariable('jet', 2, 'pt')
jet2_eta = d.ExtractVariable('jet', 2, 'eta')
jet2_phi = d.ExtractVariable('jet', 2, 'phi')

'b-Jets'
bjet1_pt = d.ExtractVariable('bjet', 1, 'pt')
bjet1_eta = d.ExtractVariable('bjet', 1, 'eta')
bjet1_phi = d.ExtractVariable('bjet', 1, 'phi')

bjet2_pt = d.ExtractVariable('bjet', 2, 'pt')
bjet2_eta = d.ExtractVariable('bjet', 2, 'eta')
bjet2_phi = d.ExtractVariable('bjet', 2, 'phi')

'MET'
met_pt = d.ExtractVariable('met', 1, 'pt')
met_phi = d.ExtractVariable('met', 1, 'phi')

'Tops'
top1_pt = d.ExtractVariable('top', 1, 'pt')
top1_eta = d.ExtractVariable('top', 1, 'eta')
top1_phi = d.ExtractVariable('top', 1, 'phi')
top1_m = d.ExtractVariable('top', 1, 'm')

top2_pt = d.ExtractVariable('top', 2, 'pt')
top2_eta = d.ExtractVariable('top', 2, 'eta')
top2_phi = d.ExtractVariable('top', 2, 'phi')
top2_m = d.ExtractVariable('top', 2, 'm')

'''Empty array definition'''
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
pre_met_pz = np.zeros(N)

'tops'

#top1_ttZ_four_mom = np.zeros((N_ttZ,4))
#top1_ttZ_inv_mass = np.zeros(N_ttZ)

'W'
W_energy = np.zeros(N)
W_four_mom = np.zeros((N,4))

'Z'
Z_pt = np.zeros(N)


'Unimplemented'
#ttZ_four_mom = np.zeros((N,4))
#tt_four_mom = np.zeros((N,4))
#tt_inv_mass = np.zeros(N)


'''Calculations'''

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

 
    '''Reconstructions'''
    
    'Z boson'
    
    Z_pt [i] = np.sqrt(((lep12_four_mom [i,1])**2)+((lep12_four_mom [i,2])**2))
    
    'W boson'
    met_px [i] = met_pt[i]*np.cos(met_phi[i])
    met_py [i] = met_pt[i]*np.sin(met_phi[i])
    # Break the pz calculation into two, for simplicity
    pre_met_pz [i] = ((W_mass)**2)+((lep3_four_mom[i,1]+met_px [i])**2)+((lep3_four_mom[i,2]+met_py[i])**2)
    # Put in an extra square and square root for taking real part of pz
    met_pz [i] = np.sqrt(
        np.sqrt(
            ((((pre_met_pz [i]-((lep3_four_mom[i,0])**2)-((met_pt[i])**2)-((lep3_four_mom[i,3])**2))/(2*lep3_four_mom[i,0]))**2)-((met_pt[i])**2))
            
            **2))
    
    W_energy[i] = np.sqrt(((W_mass)**2)+f.mom(met_px[i], met_py[i], met_pz[i]))
    W_four_mom [i,0] = lep3_four_mom[i,0]+f.mom(met_px[i], met_py[i], (met_pz[i]))

    # Giving values less than rest mass, impossible
    if W_four_mom [i,0] < W_mass:
        count = count+1

'''Pre-plot modifications'''

'State separations'
# weights
ttZ_weight = weight[0:N_ttZ]
ttWm_weight = weight[N_ttZ:N_ttWm]
ttWp_weight = weight[N_ttWm:N_ttWp]
other_weight = weight[N_ttZ:N_ttWp]
ggA_460_360_weight = weight[N_ttWp:N_ggA_460_360]

# ttZ
lep1_ttZ_pt = lep1_pt[0:N_ttZ]
lep2_ttZ_pt = lep2_pt[0:N_ttZ]
lep3_ttZ_pt = lep3_pt[0:N_ttZ]

# other
lep1_other_pt = lep1_pt[N_ttZ:N_ttWp]
lep2_other_pt = lep2_pt[N_ttZ:N_ttWp]
lep3_other_pt = lep3_pt[N_ttZ:N_ttWp]

# ggA_460_360
lep1_ggA_460_360_pt = lep1_pt[N_ttWp:N_ggA_460_360]
lep2_ggA_460_360_pt = lep2_pt[N_ttWp:N_ggA_460_360]
lep3_ggA_460_360_pt = lep3_pt[N_ttWp:N_ggA_460_360]

# Z
Z_ttZ_pt = Z_pt[0:N_ttZ]
Z_other_pt = Z_pt[N_ttZ:N_ttWp]
Z_ggA_460_360_pt = Z_pt[N_ttWp:N_ggA_460_360]

'Array combinations'
# Lepton pt's
lep1_pt_plot = ([lep1_other_pt,lep1_ttZ_pt])
lep2_pt_plot = ([lep2_other_pt,lep2_ttZ_pt])
lep3_pt_plot = ([lep3_other_pt,lep3_ttZ_pt])

# Z
Z_pt_arr = ([Z_other_pt,Z_ttZ_pt])

'Labels, colours and weights'
label = [r'other',r't$\bar{t}$Z','ggA ($m_A$=460, $m_H$=360)']
color = ['lightgreen','cornflowerblue','red']
weights = ([other_weight, ttZ_weight, ggA_460_360_weight])

'''Plots'''

'Singular histograms'

f.Hist(lep12_inv_mass, "Di-lepton (1-2) invariant mass", 20, close=True, label='lep12',
     xtitle="$p_T$ (GeV)", ytitle="Events (#)", title="Di-lepton (1-2) invariant mass")

f.Hist(lep13_inv_mass, "Di-lepton (1-3) invariant mass", 20, close=True,  label='lep13',
     xtitle="$p_T$ (GeV)", ytitle="Events (#)", title="Di-lepton (1-3) invariant mass")

f.Hist(lep23_inv_mass, "Di-lepton (2-3) invariant mass", 20, close=True,  label='lep23',
     xtitle="$p_T$ (GeV)", ytitle="Events (#)", title="Di-lepton (2-3) invariant mass")

f.Hist(jet12_inv_mass, "Di-jet (1-2) invariant mass", 20, close=True,  label='jet12',
     xtitle="$p_T$ (GeV)", ytitle="Events (#)", title="Di-jet (1-2) invariant mass", xmax=160, xmin=0)

#!!! Fix double plot on stacks

'Stacked signal histograms'


# Plot for input variables
f.SignalHist(lep1_pt_plot, lep1_ggA_460_360_pt, weights, 25, close=False,  
             label=label, color=color, xtitle="$p_T$ (GeV)", ytitle="Events", title="Lepton 1 $p_T$", 
             saveas="Lepton1_pt")

f.SignalHist(lep2_pt_plot, lep2_ggA_460_360_pt, weights, 25, close=False,  
             label=label, color=color, xtitle="$p_T$ (GeV)", ytitle="Events", title="Lepton 2 $p_T$", 
             saveas="Lepton2_pt")

f.SignalHist(lep3_pt_plot, lep3_ggA_460_360_pt, weights, 25, close=False,  
             label=label, color=color, xtitle="$p_T$ (GeV)", ytitle="Events", title="Lepton 3 $p_T$", 
             saveas="Lepton3_pt")

# Z boson plot
f.SignalHist(Z_pt_arr, Z_ggA_460_360_pt, weights, 25, close=True,  
             label=label, color=color, xtitle="$p_T$ (GeV)", ytitle="Events (#)", title="$Z_{pT}$",
             saveas="Z_pT")

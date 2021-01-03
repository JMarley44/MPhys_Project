# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 19:55:22 2020

@author: James
"""
import pandas as pd
import numpy as np
import DataExtract as d
import Functions as f

'''Data import'''

path = "C:/Users/James/Documents/Liverpool/Year 4/PHYS498 - Project/Python files/MPhys_Project/dataset_semileptonic.csv"
dataimport = pd.read_csv(path, header=None)
dataset_full = pd.DataFrame(dataimport).to_numpy()
dataset = np.delete(dataset_full,0,1)
N = len(dataset)

'''Data extraction'''

# Use DataExtract module
data_ttZ, data_ttWm, data_ttWp, data_ggA_460_360, data_ggA_500_360, data_ggA_600_360,data_ggA_600_400,data_ggA_600_500, data_ggA_500_400 = d.ExtractSignals(dataset_full, N)

# Signal dataset lengths
N_ttZ = len(data_ttZ)

# Weights
ggA_460_360_weight = d.ExtractVariable(data_ggA_460_360, 'weight',1,1)
ttWm__weight = d.ExtractVariable(data_ttWm, 'weight',1,1)
ttWp__weight = d.ExtractVariable(data_ttWp, 'weight',1,1)
ttZ__weight = d.ExtractVariable(data_ttZ, 'weight',1,1)

'Leptons'
lep1_pt = d.ExtractVariable(dataset, 'lep', 1, 'pt')
lep1_eta = d.ExtractVariable(dataset, 'lep', 1, 'eta')
lep1_phi = d.ExtractVariable(dataset, 'lep', 1, 'phi')

lep2_pt = d.ExtractVariable(dataset, 'lep', 2, 'pt')
lep2_eta = d.ExtractVariable(dataset, 'lep', 2, 'eta')
lep2_phi = d.ExtractVariable(dataset, 'lep', 2, 'phi')

lep3_pt = d.ExtractVariable(dataset, 'lep', 3, 'pt')
lep3_eta = d.ExtractVariable(dataset, 'lep', 3, 'eta')
lep3_phi = d.ExtractVariable(dataset, 'lep', 3, 'phi')

lep1_ttZ_pt = d.ExtractVariable(data_ttZ, 'lep', 1, 'pt')
lep1_ttWp_pt = d.ExtractVariable(data_ttWp, 'lep', 1, 'pt')
lep1_ttWm_pt = d.ExtractVariable(data_ttWm, 'lep', 1, 'pt')
lep1_ggA_460_360_pt = d.ExtractVariable(data_ggA_460_360, 'lep', 1, 'pt')

lep2_ttZ_pt = d.ExtractVariable(data_ttZ, 'lep', 2, 'pt')
lep2_ttWp_pt = d.ExtractVariable(data_ttWp, 'lep', 2, 'pt')
lep2_ttWm_pt = d.ExtractVariable(data_ttWm, 'lep', 2, 'pt')
lep2_ggA_460_360_pt = d.ExtractVariable(data_ggA_460_360, 'lep', 2, 'pt')

'Jets'
jet1_pt = d.ExtractVariable(dataset, 'jet', 1, 'pt')
jet1_eta = d.ExtractVariable(dataset, 'jet', 1, 'eta')
jet1_phi = d.ExtractVariable(dataset, 'jet', 1, 'phi')

jet2_pt = d.ExtractVariable(dataset, 'jet', 2, 'pt')
jet2_eta = d.ExtractVariable(dataset, 'jet', 2, 'eta')
jet2_phi = d.ExtractVariable(dataset, 'jet', 2, 'phi')

'Tops'
top1_ttZ_pt = d.ExtractVariable(data_ttZ, 'top', 1, 'pt')
top1_ttZ_eta = d.ExtractVariable(data_ttZ, 'top', 1, 'eta')
top1_ttZ_phi = d.ExtractVariable(data_ttZ, 'top', 1, 'phi')
top1_ttZ_m = d.ExtractVariable(data_ttZ, 'top', 1, 'm')

'''Empty array definition'''

lep1_four_mom = np.zeros((N,4))
lep2_four_mom = np.zeros((N,4))
lep3_four_mom = np.zeros((N,4))

lep12_four_mom = np.zeros((N,4))
lep13_four_mom = np.zeros((N,4))
lep23_four_mom = np.zeros((N,4))

lep12_inv_mass = np.zeros(N)
lep13_inv_mass = np.zeros(N)
lep23_inv_mass = np.zeros(N)

jet1_four_mom = np.zeros((N,4))
jet2_four_mom = np.zeros((N,4))

jet12_four_mom = np.zeros((N,4))
jet12_inv_mass = np.zeros(N)

top1_ttZ_four_mom = np.zeros((N_ttZ,4))
top1_ttZ_inv_mass = np.zeros(N_ttZ)

'''Calculations'''

'All data set loop'
for i in range(N):
    
    'Lepton'
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
    
    'Jet'
    # Four momenta
    jet1_four_mom [i,:] = f.four_mom(jet1_pt[i], jet1_eta[i], jet1_phi[i])
    jet2_four_mom [i,:] = f.four_mom(jet2_pt[i], jet2_eta[i], jet2_phi[i])
    
    # Addition of four momenta
    jet12_four_mom [i,:] = jet1_four_mom[i,:]+jet2_four_mom[i,:]
    
    #Invariant mass
    jet12_inv_mass [i] = f.inv_mass(jet12_four_mom[i])



'''Plots'''

'Singular histograms'

# Invariants
f.Hist(lep12_inv_mass, "Di-lepton (1-2) invariant mass", 20, close=True, label='lep12',
     xtitle="$p_T$ (GeV)", ytitle="Events (#)", title="Di-lepton (1-2) invariant mass")

f.Hist(lep13_inv_mass, "Di-lepton (1-3) invariant mass", 20, close=True,  label='lep13',
     xtitle="$p_T$ (GeV)", ytitle="Events (#)", title="Di-lepton (1-3) invariant mass")

f.Hist(lep23_inv_mass, "Di-lepton (2-3) invariant mass", 20, close=True,  label='lep23',
     xtitle="$p_T$ (GeV)", ytitle="Events (#)", title="Di-lepton (2-3) invariant mass")

f.Hist(jet12_inv_mass, "Di-jet (1-2) invariant mass", 20, close=True,  label='jet12',
     xtitle="$p_T$ (GeV)", ytitle="Events (#)", title="Di-jet (1-2) invariant mass", xmax=160, xmin=0)


'Stacked histograms'

# Define labels, colors and weights for histogram stacks
label = [r't$\bar{t}$Z',r't$\bar{t}$$W^+$',r't$\bar{t}$$W^-$','ggA ($m_A$=460, $m_H$=360)']
color = ['cornflowerblue','lightgreen','bisque','red']
weights = ([ttZ__weight, ttWp__weight, ttWm__weight,ggA_460_360_weight])

# Combine the backgrounds into arrays
lep1_pt_plot = ([lep1_ttZ_pt,lep1_ttWp_pt,lep1_ttWm_pt])
lep2_pt_plot = ([lep2_ttZ_pt,lep2_ttWp_pt,lep2_ttWm_pt])

# Plot signal histograms for input variables
f.SignalHist(lep1_pt_plot, lep1_ggA_460_360_pt, weights, "1st lepton p\u209C", 25, close=False,  
             label=label, color=color, xtitle="$p_T$ (GeV)", ytitle="Events", title="Lepton 1 $p_T$", 
             saveas="Lepton1_pt")

f.SignalHist(lep2_pt_plot, lep2_ggA_460_360_pt, weights, "2nd lepton p\u209C", 25, close=False,  
             label=label, color=color, xtitle="$p_T$ (GeV)", ytitle="Events", title="Lepton 2 $p_T$", 
             saveas="Lepton2_pt")

######OLD
'''
Initial program

def doit(tree, tag, scale=1.):
    for e in tree:
        print ("\""+tag+"\",",
        print (e.eventWeight/scale , ',',
        print (e.lep1_pt      , ',',
        print (e.lep1_eta     , ',',
        print (e.lep1_phi     , ',',
        print (e.lep1_q     , ',',
        print (e.lep1_fl    , ',',
        print (e.lep2_pt      , ',',
        print (e.lep2_eta     , ',',
        print (e.lep2_phi     , ',',
        print (e.lep2_q     , ',',
        print (e.lep2_fl    , ',',
        print (e.lep3_pt      , ',',
        print (e.lep3_eta     , ',',
        print (e.lep3_phi     , ',',
        print (e.lep3_q     , ',',
        print (e.lep3_fl    , ',',
        print (e.bjet1_pt     , ',',
        print (e.bjet1_eta    , ',',
        print (e.bjet1_phi    , ',',
        print (e.bjet2_pt     , ',',
        print (e.bjet2_eta    , ',',
        print (e.bjet2_phi    , ',',
        print (e.jet1_pt      , ',',
        print (e.jet1_eta     , ',',
        print (e.jet1_phi     , ',',
        print (e.jet2_pt      , ',',
        print (e.jet2_eta     , ',',
        print (e.jet2_phi     , ',',
               
        print (e.met_pt       , ',',
        print (e.met_phi      , ',',
               
        print (e.ttbar_eta    , ',',
        print (e.ttbar_phi    , ',',
        print (e.ttbar_pt     , ',',
        print (e.ttbar_m      , ',',
        print (e.zttbar_eta   , ',',
        print (e.zttbar_phi   , ',',
        print (e.zttbar_pt    , ',',
        print (e.zttbar_m     , ',',
        print (e.top1_eta     , ',',
        print (e.top1_phi     , ',',
        print (e.top1_pt      , ',',
        print (e.top1_m       , ',',
        print (e.top2_eta     , ',',
        print (e.top2_phi     , ',',
        print (e.top2_pt      , ',',
        print (e.top2_m 

v_files = ["skimmed_slep_ttZ.root",
           "skimmed_slep_ttWm.root",
           "skimmed_slep_ttWp.root",
           "skimmed_slep_ggA_460_360.root",
           "skimmed_slep_ggA_500_360.root",
           "skimmed_slep_ggA_600_360.root",
           "skimmed_slep_ggA_600_400.root",
           "skimmed_slep_ggA_600_500.root",
           "skimmed_slep_ggA_500_400.root",]

v = R.TLorentzVector()
v.SetPtEtaPhiM(pt, eta, phi, 0)
print( v.Pt() )
print(  v.M() )
'''
'''
for fname in v_files:
    f = R.TFile(path+fname,"read")
    tree = f.Get("newtree")
    scale = 1.
    #if not "ggA" in fname:
    #    scale=1000.
    tag = fname[13:-5]

    #print tag, scale
    doit(tree, tag, scale)
'''
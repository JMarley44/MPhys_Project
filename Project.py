# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 19:55:22 2020

@author: James
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''Data import'''

path = "C:/Users/James/Documents/Liverpool/Year 4/PHYS498 - Project/Python files/MPhys_Project/dataset_semileptonic.csv"
dataimport = pd.read_csv(path, header=None)
dataset = pd.DataFrame(dataimport).to_numpy()

N = len(dataset)

'''Data extraction'''

lep1_pt = dataset[:,2]
lep1_eta = dataset[:,3]
lep1_phi = dataset[:,4]
lep1_q = dataset[:,5]
lep1_fla = dataset[:,6]

lep2_pt = dataset[:,7]
lep2_eta = dataset[:,8]
lep2_phi = dataset[:,9]
lep2_q = dataset[:,10]
lep2_fla = dataset[:,11]

lep3_pt = dataset[:,12]
lep3_eta = dataset[:,13]
lep3_phi = dataset[:,14]
lep3_q = dataset[:,15]
lep3_fla = dataset[:,16]

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

    #####

'''Functions'''

def Plot(X, tag, Nb, close, label, **kwargs):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12.0,10.0))
    
    xtitle = tag
    ytitle = tag
    title = tag
    for key, value in kwargs.items():
        if key == "xtitle":
            xtitle = value
        elif key=="ytitle":
            ytitle = value
        elif key=="title":
            title = value
        elif key=="label":
            label = value
            
    #Definition of variables
    themin = np.amin(X)-1
    themax = np.amax(X)+1
    bins = np.linspace(themin, themax, Nb)
    
    if X.ndim > 1:
        for i in range(len(X[0])):
            plt.hist(X[:,i], bins=bins, label=label[i])
            plt.show()
    elif X.ndim <= 1:
       plt.hist(X, bins=bins, label=label)
        
    plt.title(title, fontsize=40)
    plt.xlabel(xtitle, fontsize=25)
    plt.ylabel(ytitle, fontsize=25)
    plt.legend(loc='upper right')
    plt.xlim(themin, themax)
    ax.set_xticks(bins, minor=True)
    ax.grid(which='minor', axis='x', alpha = 0.5)
    ax.grid(which='major', axis='y', alpha = 0.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig("Plots/"+title+".png")
    if close: plt.close('all')

def four_mom(pt, eta, phi):
    p = np.zeros((4))
    p[0] = pt*np.cosh(eta)
    p[1] = pt*np.cos(phi)
    p[2] = pt*np.sin(phi)
    p[3] = pt*np.sinh(eta)
    return p

def inv_mass(four_mom):
    p = np.sqrt((four_mom[1]**2)+(four_mom[2]**2)+(four_mom[3]**2))
    m0 = np.sqrt(four_mom[0]**2-p**2)
    return m0

    #####

for i in range(N):
    '''Four momentum calculation'''
    lep1_four_mom [i,:] = four_mom(lep1_pt[i], lep1_eta[i], lep1_phi[i])
    lep2_four_mom [i,:] = four_mom(lep2_pt[i], lep2_eta[i], lep2_phi[i])
    lep3_four_mom [i,:] = four_mom(lep3_pt[i], lep3_eta[i], lep3_phi[i])
    
    '''Addition of four momenta'''
    lep12_four_mom [i,:] = lep1_four_mom[i,:]+lep2_four_mom[i,:]
    lep13_four_mom [i,:] = lep1_four_mom[i,:]+lep3_four_mom[i,:]
    lep23_four_mom [i,:] = lep2_four_mom[i,:]+lep3_four_mom[i,:]
    
    '''Invariant mass total calculation'''
    lep12_inv_mass [i] = inv_mass(lep12_four_mom[i])
    lep13_inv_mass [i] = inv_mass(lep13_four_mom[i])
    lep23_inv_mass [i] = inv_mass(lep23_four_mom[i])

'''Plots'''
Plot(lep12_inv_mass, "Di-lepton (1-2) invariant mass", 20, False, label='lep12',
     xtitle="M\u2080 (GeV)", ytitle="Counts (#)", title="Di-lepton (1-2) invariant mass")

Plot(lep13_inv_mass, "Di-lepton (1-3) invariant mass", 20, False,  label='lep13',
     xtitle="M\u2080 (GeV)", ytitle="Counts (#)", title="Di-lepton (1-3) invariant mass")

Plot(lep23_inv_mass, "Di-lepton (2-3) invariant mass", 20, False,  label='lep23',
     xtitle="M\u2080 (GeV)", ytitle="Counts (#)", title="Di-lepton (2-3) invariant mass")

di_lep = np.stack((lep12_inv_mass,lep13_inv_mass,lep23_inv_mass),axis=1)

Plot(di_lep, "Di-lepton combined", 50, False,  label=['lep12','lep13','lep23'],
     xtitle="M\u2080 (GeV)", ytitle="Counts (#)", title="Di-lepton invariant masses")


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
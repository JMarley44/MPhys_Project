# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 19:55:22 2020

@author: James
"""

import ROOT as R

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
)))))))))))))))))))))))))))))))))))))))))))))))




path = "C:/Users/James/Documents/Liverpool/Year 4/PHYS498 - Project/Python files/MPhys_Project/dataset_semileptonic.csv"


v_files = ["skimmed_slep_ttZ.root",
           "skimmed_slep_ttWm.root",
           "skimmed_slep_ttWp.root",
           "skimmed_slep_ggA_460_360.root",
           "skimmed_slep_ggA_500_360.root",
           "skimmed_slep_ggA_600_360.root",
           "skimmed_slep_ggA_600_400.root",
           "skimmed_slep_ggA_600_500.root",
           "skimmed_slep_ggA_500_400.root",]


for fname in v_files:
    f = R.TFile(path+fname,"read")
    tree = f.Get("newtree")
    scale = 1.
    #if not "ggA" in fname:
    #    scale=1000.
    tag = fname[13:-5]

    #print tag, scale
    doit(tree, tag, scale)
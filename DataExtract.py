# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:18:27 2020

@author: James
"""

def ExtractVariable(dataset, part, partno, variable):
    #from Project import dataset
    partno = partno-1
    
    if part=="weight":
        i=0
    elif part=="met":
        if variable == "pt": i=28
        elif variable == "phi": i=29 
        else: print("incorrect variable")
    
    elif part=="lep" or part=="bjet" or part=="jet":
        d = {'pt':0, 'eta':1, 'phi':2, 'q':3, 'fl':4}
        if part=="lep": i=1+(partno*5)+d[variable]
        elif part=="bjet": i=16+(partno*3)+d[variable]
        elif part == "jet": i=22+(partno*3)+d[variable]
        else: print("incorrect variable")
        
    elif part=="tt" or part=="ztt" or part=="top":
        d = {'eta':0, 'phi':1, 'pt':2, 'm':3}
        if part == "tt": i=30+d[variable]
        elif part == "ztt": i=34+d[variable]
        elif part == "top": i=38+(partno*4)+d[variable]
        else: print("incorrect variable")
    else:
        print("incorrect particle type")
        
    variable = dataset[:,i]
    return variable

def DataSplit(dataset_full):
    N_ttZ  = 0
    N_ttWm  = 0
    N_ttWp  = 0
    N_ggA_460_360 = 0
    N_ggA_500_360 = 0
    N_ggA_600_360 = 0
    N_ggA_600_400 = 0
    N_ggA_600_500 = 0
    N_ggA_500_400 = 0
    
    for i in range(len(dataset_full)):
        
        if (dataset_full[i,0]=='ttZ'):
            N_ttZ = N_ttZ + 1
            
        elif (dataset_full[i,0]=='ttWm'):
            N_ttWm = N_ttWm + 1
            
        elif (dataset_full[i,0]=='ttWp'):
            N_ttWp = N_ttWp + 1
            
        elif (dataset_full[i,0]=='ggA_460_360'):
            N_ggA_460_360 = N_ggA_460_360 + 1
            
        elif (dataset_full[i,0]=='ggA_500_360'):
            N_ggA_500_360 = N_ggA_500_360 + 1
            
        elif (dataset_full[i,0]=='ggA_600_360'):
            N_ggA_600_360 = N_ggA_600_360 + 1
            
        elif (dataset_full[i,0]=='ggA_600_400'):
            N_ggA_600_400 = N_ggA_600_400 + 1
            
        elif (dataset_full[i,0]=='ggA_600_500'):
            N_ggA_600_500 = N_ggA_600_500 + 1
            
        elif (dataset_full[i,0]=='ggA_500_400'):
            N_ggA_500_400 = N_ggA_500_400 + 1
            
        
    N_ttWm = N_ttWm + N_ttZ
    N_ttWp = N_ttWp + N_ttWm
    N_ggA_460_360 = N_ggA_460_360 + N_ttWp
    N_ggA_500_360 = N_ggA_500_360 + N_ggA_460_360
    N_ggA_600_360 = N_ggA_600_360 + N_ggA_500_360
    N_ggA_600_400 = N_ggA_600_400 + N_ggA_600_360
    N_ggA_600_500 = N_ggA_600_500 + N_ggA_600_400
    N_ggA_500_400 = N_ggA_500_400 + N_ggA_600_500
    
    
    return N_ttZ,N_ttWm,N_ttWp,N_ggA_460_360,N_ggA_500_360,N_ggA_600_360,N_ggA_600_400,N_ggA_600_500,N_ggA_500_400
    
'''
Data numbering

   0     Weight
   1     lep1_pt
   2     lep1_eta
   3     lep1_phi
   4     lep1_q
   5     lep1_fl
   6     lep2_pt
   7     lep2_eta
   8     lep2_phi
   9     lep2_q
   10     lep2_fl
   11     lep3_pt
   12     lep3_eta
   13     lep3_phi
   14     lep3_q
   15     lep3_fl
   16     bjet1_pt
   17     bjet1_eta
   18     bjet1_phi
   19     bjet2_pt
   20     bjet2_eta
   21     bjet2_phi
   22     jet1_pt
   23     jet1_eta
   24     jet1_phi
   25     jet2_pt
   26     jet2_eta
   27     jet2_phi
               
   28     met_pt
   29     met_phi
              
   30     ttbar_eta
   31     ttbar_phi
   32     ttbar_pt
   33     ttbar_m
   34     zttbar_eta
   35     zttbar_phi
   36     zttbar_pt
   37     zttbar_m
                 
   38     top1_eta
   39     top1_phi
   40     top1_pt
   41     top1_m
   42     top2_eta
   43     top2_phi
   44     top2_pt
   45     top2_m 

'''

    
    
    
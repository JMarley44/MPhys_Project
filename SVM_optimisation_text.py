# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 21:19:46 2021

@author: James
"""

#C = , gamma =  AUC is 


#C = 100, gamma =  AUC is 
    
#C = 10, gamma =  AUC is 

#C = 1, gamma =  AUC is 

#C = 0.1, gamma =  AUC is 

#C = 0.01, gamma =  AUC is 


sig_600_500 = False
sig_500_400 = False

while sig_600_500:
    #C = 100, gamma = 100 AUC is 0.7 TRAIN IS 1!!!
    
    #C = 10, gamma = 100 AUC is 0.693 OVERTRAIN
    
    #C = 1, gamma = 100 AUC is 0.697 OVERTRAIN
    
    #C = 0.1, gamma = 100 AUC is 0.67 OVERTRAIN
    
    #C = 0.01, gamma = 100 AUC is 0.677 OVERTRAIN
    
    
    #C = 100, gamma = 10 AUC is 0.637, train is high
    
    #C = 10, gamma = 10 AUC is 0.652, train is going up
    
    #C = 1, gamma = 10 AUC is 0.649 0.05 - 1.0
    
    #C = 0.1, gamma = 10 AUC is 0.61, 0.15 - 0.85
    
    #C = 0.01, gamma = 10 AUC is 0.6, 0.1 - 0.6
    
    
    #C = 100, gamma = 1 AUC is 0.67 
    
    #C = 10, gamma = 1 AUC is 0.652 0.05 - 1
    
    #C = 1, gamma = 1 AUC is 0.657 0.05 - 1
    
    #C = 0.1, gamma = 1 AUC is 0.63 0.1 - 0.9
    
    #C = 0.01, gamma = 1 AUC is 0.62 0.15 - 0.95
    
    
    #C = 100, gamma = 0.1 AUC is 0.655
    
    #C = 10, gamma = 0.1 AUC is 0.659 0.1 - 1
    
    #C = 1, gamma = 0.1 AUC is 0.634 0.05 - 1
    
    #C = 0.1, gamma = 0.1 AUC is 0.658, 0.1 - 0.9
    
    #C = 0.01, gamma = 0.1 AUC is 0.639, 0.15 - 1
    
    
    #C = 100, gamma = 0.01 AUC is 0.643
    
    #C = 10, gamma = 0.01 AUC is 0.639
    
    #C = 1, gamma = 0.01 AUC is 0.629, range 0.1-1
    
    #C = 0.1, gamma = 0.01 AUC is 0.606, range 0.1-0.9
    
    #C = 0.01, gamma = 0.01 AUC is 0.627, no noticeable features
    
    
    #C = 100, gamma = 0.001 AUC is 0.636 0.1-1
    
    #C = 10, gamma = 0.001 AUC is 0.641
    
    #C = 1, gamma = 0.001 AUC is 0.67 - seems to be an anomaly not that great
    
    #C = 0.1, gamma = 0.001 AUC is 0.659, plot range is 0.2-0.8
    
    #C = 0.01, gamma = 0.001 AUC is 0.635, plot range .2-1
    
    
    #C = 100, gamma = 0.0001 AUC is 0.63, range 0.1- 0.8
    
    #C = 10, gamma = 0.0001 AUC is 0.66, range 0.1-0.8
    
    #C = 1, gamma = 0.0001 AUC is 0.644 ramge 0.1-0.95
    
    #C = 0.1, gamma = 0.0001 AUC is 0.631 range 0.2-0.8
    
    #C = 0.01, gamma = 0.0001 AUC is 0.595 0.2 - 0.9
    
    # LETS TRY C AND GAMMA IN POWERS, FOR TOL=1
    
    ### (delta_m, Z_pt, tops_angle, Wm_mass, Wp_mass, bjet12_angle, jet12_angle, M_ttZ, ztt_m)
    
    print()
    
while sig_500_400:
    
    # C = 1, gamma = 10 sees the best test accuracy
    # Gamma > 10 leads to high training 
    # High gamma and high C sees overtraining
    # Low gamma and low C sees poor classification

    #C = 100, gamma = 100 AUC is 0.766  OVERTRAIN
    
    #C = 10, gamma = 100 AUC is 0.758   OVERTRAIN
    
    #C = 1, gamma = 100 AUC is 0.729 train VERY high
    
    #C = 0.1, gamma = 100 AUC is 0.7 train VERY high
    
    #C = 0.01, gamma = 100 AUC is 0.72 train VERY high
    
    
    #C = 100, gamma = 10 AUC is 0.686, train very high
    
    #C = 10, gamma = 10 AUC is 0.7 , high train ~ 0.82
    
    #C = 1, gamma = 10 AUC is 0.714 test and train similar
    
    #C = 0.1, gamma = 10 AUC is 0.567 bad classification
    
    #C = 0.01, gamma = 10 AUC is 0.58 bad classification
    
    
    #C = 100, gamma = 1 AUC is 0.7 test and train similar
    
    #C = 10, gamma = 1 AUC is 0.65, test and train similar
    
    #C = 1, gamma = 1 AUC is 0.684, test and train similar
    
    #C = 0.1, gamma = 1 AUC is 0.645 plots are shifted right
    
    #C = 0.01, gamma = 1 AUC is 0.63 plots span 0.2-0.8 range
    
    
    #C = 100, gamma = 0.1 AUC is 0.64 no standout features
    
    #C = 10, gamma = 0.1 AUC is 0.65 plots span 0.1-0.9 range
    
    #C = 1, gamma = 0.1 AUC is 0.65 plots span 0.1-0.9 range
    
    #C = 0.1, gamma = 0.1 AUC is 0.65 plots span 0.2-0.8 range
    
    #C = 0.01, gamma = 0.1 AUC is 0.665 plots span 0.2-0.8 range
    
    
    #C = 100, gamma = 0.01 AUC is 0.65 similar peaks throughout
    
    #C = 10, gamma = 0.01 AUC is 0.65 similar peaks throughout
    
    #C = 1, gamma = 0.01 AUC is 0.66 plots span 0.1-0.9
    
    #C = 0.1, gamma = 0.01 AUC is 0.65 plots span 0.1-0.9
    
    #C = 0.01, gamma = 0.01 AUC is 0.66 plots span 0.2-0.8
    
    # LETS TRY C AND GAMMA IN POWERS
    
    
    #C = 100, gamma = 1 AUC is 0.61
    #C = 25, gamma = 1 AUC is 0.64
    #C = 25, gamma = .1 AUC is 0.64
    #C = 15, gamma = 5 AUC is 0.65 
    #C = 5, gamma = 5 AUC is 0.68 looking good again
    #C = 0.1, gamma = 5 AUC is 0.574 plots bad
    #C = 0.1, gamma = 7 AUC is 0.62 plots starting to go bad
    #C = 0.5, gamma = 7 AUC is 0.65 but plots looking good again
    #C = 1, gamma = 5 AUC is 0.65 but plots are looking good
    # RERUN #C = 1, gamma = 7 AUC is 0.69 train good
    #C = 1, gamma = 7 AUC is 0.71 train good
    #C = 1, gamma = 10 AUC is 0.67 train good 
    #C = 1, gamma = 15 AUC is 0.7 train good
    #C = 1, gamma = 25 AUC is 0.68 and train up
    #C = 1, gamma = 20 AUC is 0.7 train good
    #C = 5, gamma = 20 AUC is 0.712, train down
    
    ### START - tol = 1
    
    #C = 10, gamma = 20 AUC is 0.7 but train is 0.9 although distribs look okay
    
    print()
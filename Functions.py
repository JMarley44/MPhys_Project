# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:12:22 2020

@author: James
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

# Momentum
def mom(px, py, pz):
    p = np.sqrt(((px)**2)+((py)**2)+((pz)**2))
    return p

# Four momentum of massless particles
def four_mom(pt, eta, phi):
    p = np.zeros((4))
    p[0] = pt*np.cosh(eta)
    p[1] = pt*np.cos(phi)
    p[2] = pt*np.sin(phi)
    p[3] = pt*np.sinh(eta)
    return p

# Four momentum of massive particle
def four_mom_m(pt, eta, phi, m):
    p = np.zeros((4))
    fact = np.sqrt(pt**2+m**2)
    print(fact)
    p[0] = fact*np.cosh(eta)
    p[1] = pt*np.cos(phi)
    p[2] = pt*np.sin(phi)
    p[3] = fact*np.sinh(eta)
    return p

# Invariant mass
def inv_mass(four_mom):
    p_sq_sq = ((four_mom[1]**2)+(four_mom[2]**2)+(four_mom[3]**2))**2
    p = np.sqrt(np.sqrt(p_sq_sq))
    m0 = np.sqrt(four_mom[0]**2-p**2)
    return m0

def angle(f1, f2):
    dot = (f1[0]*f2[0])+(f1[1]*f2[1])+(f1[2]*f2[2])+(f1[3]*f2[3])
    
    mag1 = np.sqrt((f1[0]**2)+(f1[1]**2)+(f1[2]**2)+(f1[3]**2))
    mag2 = np.sqrt((f2[0]**2)+(f2[1]**2)+(f2[2]**2)+(f2[3]**2))
    
    mag1_mag2 = mag1*mag2
    
    angle = np.arccos(dot/mag1_mag2)
    return angle

# Single histogram plot
def Hist(X, tag, Nb, close, label, **kwargs):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(11.0,9.0))
    
    # Definition of variables
    themin = np.amin(X)
    themax = np.amax(X)
    
    for key, value in kwargs.items():
        if key == "xtitle":
            xtitle = value
        elif key=="ytitle":
            ytitle = value
        elif key=="title":
            title = value
        elif key=="label":
            label = value
        elif key=="xmax":
            themax = value
        elif key=="xmin":
            themin = value
        elif key=="scale":
            scale = value
    
    if 'scale' not in locals():
        scale = 'linear'
        
    # Calculate the bins
    bins = np.linspace(themin, themax, Nb)
    
    # Plot the histogram
    plt.hist(X, bins=bins, label=label)
      
    # Plot customisation
    plt.title(title, fontsize=40)
    plt.xlabel(xtitle, fontsize=25)
    plt.ylabel(ytitle, fontsize=25)
    plt.legend(loc='upper right')
    ax.set_xticks(bins, minor=True)
    ax.grid(which='minor', axis='x', alpha = 0.5)
    ax.grid(which='major', axis='y', alpha = 0.5)
    plt.xticks(fontsize=20)
    plt.xlim(themin, themax)
    plt.yticks(fontsize=20)
    plt.yscale(scale)
    plt.savefig("Plots/"+title+".png")
    if close: plt.close()
    

# Stacked histograms with signal
'''
X - Input data array
weight - weights of dataset
Nb - Number of bins for the histogram
close - Whether to close the plot or not

kwargs - xtitle, ytitle, title, saveas , signal

signal is what signal types to plot
'''
        
def SignalHist(X, weight, Nb, close, **kwargs):
    
    N_ttZ = 6979
    N_ttWp = 7455
    N_ttWm = 7223
    N_ggA_460_360 = 8219
    N_ggA_500_360 = 8935
    N_ggA_600_360 = 10027
    N_ggA_600_400 = 11195
    N_ggA_600_500 = 12567
    N_ggA_500_400 = 14627
    
    label = [r'other',r't$\bar{t}$Z','ggA ($m_A$=460, $m_H$=360)', 'ggA ($m_A$=500, $m_H$=360)', 'ggA ($m_A$=600, $m_H$=360)'
         , 'ggA ($m_A$=600, $m_H$=400)', 'ggA ($m_A$=600, $m_H$=500)', 'ggA ($m_A$=500, $m_H$=400)']
    
    color = ['lightgreen','cornflowerblue','black','red','gold','indigo','cyan','maroon']
    
    bkg_test = np.zeros(2) # other, ttZ
    sig_test = np.zeros(6) # 460_360, 500_360, 600_360, 600_400, 600_500, 500_400
    
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(11.0,9.0))
    global save_count
    
    # Definition of variables
    
    X_other = None
    X_ttZ = None
    weight_other = None
    weight_ttZ = None
    
    X_460_360 = None
    X_500_360 = None 
    X_600_360 = None 
    X_600_400 = None 
    X_600_500 = None 
    X_500_400 = None
    weight_460_360 = None
    weight_500_360 = None
    weight_600_360 = None
    weight_600_400 = None
    weight_600_500 = None
    weight_500_400 = None

    # Extract additonal arguments from kwargs
    for i, j in kwargs.items():
        if i == "xtitle":
            xtitle = j
        elif i=="ytitle":
            ytitle = j
        elif i=="title":
            title = j
        elif i=="saveas":
            saveas = j
        elif i == "signals":
            for k in range(len(j)):
                if j[k] == 'ttZ':
                    X_other = X[N_ttZ:N_ttWp]
                    weight_other = weight[N_ttZ:N_ttWp]
                    bkg_test[0] = 1
                if j[k] == 'other':
                    X_ttZ = X[0:N_ttZ]
                    weight_ttZ = weight[0:N_ttZ]
                    bkg_test[1] = 1
                if j[k] == '460_360':
                    X_460_360 = X[N_ttWp:N_ggA_460_360]
                    weight_460_360 = weight[N_ttWp:N_ggA_460_360]
                    sig_test[0] = 1
                if j[k] == '500_360':
                    X_500_360 = X[N_ggA_460_360:N_ggA_500_360]
                    weight_500_360 = weight[N_ggA_460_360:N_ggA_500_360]
                    sig_test[1] = 1
                if j[k] == '600_360':
                    X_600_360 = X[N_ggA_500_360:N_ggA_600_360]
                    weight_600_360 = weight[N_ggA_500_360:N_ggA_600_360]
                    sig_test[2] = 1
                if j[k] == '600_400':
                    X_600_400 = X[N_ggA_600_360:N_ggA_600_400]
                    weight_600_400 = weight[N_ggA_600_360:N_ggA_600_400]
                    sig_test[3] = 1
                if j[k] == '600_500':
                    X_600_500 = X[N_ggA_600_400:N_ggA_600_500]
                    weight_600_500 = weight[N_ggA_600_400:N_ggA_600_500]
                    sig_test[4] = 1
                if j[k] == '500_400':
                    X_500_400 = X[N_ggA_600_500:N_ggA_500_400]
                    weight_500_400 = weight[N_ggA_600_500:N_ggA_500_400]
                    sig_test[5] = 1

# ggA_500_360 - [N_ggA_460_360:N_ggA_500_360]
# ggA_600_360 - [N_ggA_500_360:N_ggA_600_360]
# ggA_600_400 - [N_ggA_600_360:N_ggA_600_400]
# ggA_600_500 - [N_ggA_600_400:N_ggA_600_500]
# ggA_500_400 - [N_ggA_600_500:N_ggA_500_400]

    # Save names don't exist, give them arbitrary values
    if 'saveas' not in locals():
        saveas = ("unnamed_plot_%d" %save_count)
        save_count = save_count+1
        print('Warning, some plots have no save title')
        
            
    bkg = ([X_other, X_ttZ])
    sig = ([X_460_360, X_500_360, X_600_360, X_600_400, X_600_500, X_500_400])
    weight = ([weight_other, weight_ttZ, weight_460_360, weight_500_360 ,weight_600_360 ,weight_600_400 ,weight_600_500 ,weight_500_400])
    
    bkg_length = len(bkg)
    
    # Max and min
    all_bkg_max = np.empty(len(bkg))
    all_bkg_min = np.empty(len(bkg))

    for i in range(len(bkg)):
        all_bkg_max[i] = np.amax(bkg[i])
        all_bkg_min[i] = np.amin(bkg[i])
        
    bkg_maxim = np.nanmax(all_bkg_max)
    bkg_minim = np.nanmin(all_bkg_min)
    
    # Calculate bins and widths
    bins = np.linspace(bkg_minim, bkg_maxim, Nb)
    width = bins[1]-bins[0]

    bins_plot = np.delete(bins,Nb-1)
    stack_count=0
    
    for i in range(len(bkg_test)):
        if bkg_test[i] == 1:
            count,edge = np.histogram(bkg[i], bins=bins, weights=weight[i])
            plt.bar(bins_plot, count, label=label[i], width = width, align='edge', bottom = stack_count, color=color[i])
            plt.legend(loc='upper right')
            stack_count = stack_count + count
            
    for i in range(len(sig_test)):
        if sig_test[i] == 1:
            step_count,edge = np.histogram(sig[i], bins=bins, weights=weight[bkg_length+i])
            step_count_ext = np.append(step_count, step_count[len(step_count)-1])
            plt.step(bins, step_count_ext, label=label[bkg_length+i], color=color[bkg_length+i], where='post', linewidth = 2.0, linestyle='dashed')
   
    # Plot customisation
    ytitle = (ytitle + " / {width:} GeV").format(width = int(width))
    plt.title(title, fontsize=40)
    plt.xlabel(xtitle, fontsize=25)
    plt.ylabel(ytitle, fontsize=25)
    plt.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1, .9), framealpha=1, edgecolor='k')
    plt.xlim(bkg_minim, bkg_maxim)
    ax.set_xticks(bins, minor=True)
    ax.grid(which='minor', axis='x', alpha = 0.5)
    ax.grid(which='major', axis='y', alpha = 0.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.yscale('log')
    
    # String for adding to the text box
    string = ('$\u221As = 13 TeV, 139$ $fb^{-1}$')
    
    # Add a text box
    ax.text(.97, .97, string, transform=ax.transAxes, fontsize=12, fontweight='bold', horizontalalignment='right',
            verticalalignment='top', bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='round,pad=1'))
    
    # Save the figure, close if close is set true
    plt.savefig("Plots/"+saveas+".png")
    if close: 
        plt.close()
    else:
        plt.show()        

'''
Function to prepare x and y data for machine learning
'''
def data_prep(variables, N_ttZ, signal_start, signal_end):
    
    N = 14627
    
    var_length = len(variables)
    sig_length = signal_end - signal_start
    
    all_variables = np.zeros((N, var_length))
    
    for i in range(var_length):
        all_variables[:,i] = variables[i]
    
    background = all_variables[0:N_ttZ]
    signal = all_variables[signal_start:signal_end]
    
    zeros = np.zeros((N_ttZ,1))
    ones = np.ones((sig_length,1))
    
    xdata = np.concatenate((background,signal))
    ydata = np.concatenate((zeros,ones))
    
    data = np.hstack((xdata,ydata))
    
    return data
    
def ROC_Curve(model_data, model_pred, N_ttZ, tag):
    
    y_pos = len(model_data[0])-1
    length = len(model_pred)
    
    y_all = model_data[:,y_pos]
    y_pred_all = model_pred

    y_train = y_all[0:N_ttZ]
    y_pred_train = y_pred_all[0:N_ttZ]
    
    y_test = y_all[N_ttZ:length]
    y_pred_test = y_pred_all[N_ttZ:length]
    
    fpr_all, tpr_all, thresholds = metrics.roc_curve(y_all, y_pred_all)
    fpr_train, tpr_train, thresholds = metrics.roc_curve(y_train, y_pred_train)
    fpr_test, tpr_test, thresholds = metrics.roc_curve(y_test, y_pred_test)
    
    auc_all = metrics.auc(fpr_all, tpr_all)
    auc_train = metrics.auc(fpr_train, tpr_train)
    auc_test = metrics.auc(fpr_test, tpr_test)
    
    plt.figure(figsize=(10.0,8.0))
    plt.plot(fpr_all, tpr_all, label='All (area = {:.3f})'.format(auc_all), color='red')
    plt.plot(fpr_test, tpr_test, label='Test (area = {:.3f})'.format(auc_test), color='deepskyblue')
    plt.plot(fpr_train, tpr_train, label='Train (area = {:.3f})'.format(auc_train), color='limegreen')
    plt.xlabel('False positive rate', fontsize=25)
    plt.ylabel('True positive rate', fontsize=25)
    plt.title('ROC curve', fontsize=40)
    plt.legend(loc='best', fontsize=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    plt.savefig("Plots/ROC_Curve_"+tag+".png")
    

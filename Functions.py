# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:12:22 2020

@author: James
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from scipy.stats import norm

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

kwargs - xtitle, ytitle, title, saveas, normed, signal

normed - whether to normalise - default: False 
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
    
    color = ['lightgreen','cyan','cornflowerblue','red','gold','indigo','black','maroon']
    
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
    
    normed = False
    scale = 'linear'

    # Extract additonal arguments from kwargs
    for i, j in kwargs.items():
        if i == "xtitle":
            xtitle = j
        elif i=="ytitle":
            ytitle_in = j
        elif i=="title":
            title = j
        elif i=="saveas":
            saveas = j
        elif i =="normed":
            normed = j
        elif i =="line":
            linestyle = j
        elif i =="scale":
            scale = j
        elif i =='xlim':
            #!!! This needs sorting out
            bkg_minim = j[0]
            bkg_maxim = j[1]
        elif i == "signals":
            for k in range(len(j)):
                if j[k] == 'other':
                    X_other = X[N_ttZ:N_ttWp]
                    weight_other = weight[N_ttZ:N_ttWp]
                    bkg_test[0] = 1
                elif j[k] == 'ttZ':
                    X_ttZ = X[0:N_ttZ]
                    weight_ttZ = weight[0:N_ttZ]
                    bkg_test[1] = 1
                elif j[k] == '460_360':
                    X_460_360 = X[N_ttWp:N_ggA_460_360]
                    weight_460_360 = weight[N_ttWp:N_ggA_460_360]
                    sig_test[0] = 1
                elif j[k] == '500_360':
                    X_500_360 = X[N_ggA_460_360:N_ggA_500_360]
                    weight_500_360 = weight[N_ggA_460_360:N_ggA_500_360]
                    sig_test[1] = 1
                elif j[k] == '600_360':
                    X_600_360 = X[N_ggA_500_360:N_ggA_600_360]
                    weight_600_360 = weight[N_ggA_500_360:N_ggA_600_360]
                    sig_test[2] = 1
                elif j[k] == '600_400':
                    X_600_400 = X[N_ggA_600_360:N_ggA_600_400]
                    weight_600_400 = weight[N_ggA_600_360:N_ggA_600_400]
                    sig_test[3] = 1
                elif j[k] == '600_500':
                    X_600_500 = X[N_ggA_600_400:N_ggA_600_500]
                    weight_600_500 = weight[N_ggA_600_400:N_ggA_600_500]
                    sig_test[4] = 1
                elif j[k] == '500_400':
                    X_500_400 = X[N_ggA_600_500:N_ggA_500_400]
                    weight_500_400 = weight[N_ggA_600_500:N_ggA_500_400]
                    sig_test[5] = 1
                else:
                    print('Unknown signal in SignalHist')

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
    # Linestyle doesn't exist, make them all dashed
    if 'linestyle' not in locals():
        linestyle = ['--' for i in range(6)]
            
    bkg = ([X_other, X_ttZ])
    sig = ([X_460_360, X_500_360, X_600_360, X_600_400, X_600_500, X_500_400])
    weight = ([weight_other, weight_ttZ, weight_460_360, weight_500_360,weight_600_360,weight_600_400,weight_600_500 ,weight_500_400])
    
    bkg_length = len(bkg)
    
    # Max and min
    all_bkg_max = np.empty(len(bkg))
    all_bkg_min = np.empty(len(bkg))

    for i in range(len(bkg)):
        all_bkg_max[i] = np.amax(bkg[i])
        all_bkg_min[i] = np.amin(bkg[i])
        
    if 'bkg_maxim' not in locals():
        bkg_maxim = np.nanmax(all_bkg_max)
        bkg_minim = np.nanmin(all_bkg_min)
    
    #print(bkg_maxim)
    
    # Calculate bins and widths
    bins = np.linspace(bkg_minim, bkg_maxim, Nb)
    width = bins[1]-bins[0]

    bins_plot = np.delete(bins,Nb-1)
    stack_count=0
    
    for i in range(len(bkg_test)):
        if bkg_test[0] == 1:
            count,edge = np.histogram(bkg[i], bins=bins, weights=weight[i], density=normed)
            if normed:
                count = count/2
            plt.bar(bins_plot, count, label=label[i], width = width, align='edge', bottom = stack_count, color=color[i])
            plt.legend(loc='upper right')
            stack_count = stack_count + count
        if bkg_test[1] == 1:
            ttZcount,edge = np.histogram(bkg[i], bins=bins, weights=weight[i], density=normed)
            if normed:
                count = count/2
            plt.bar(bins_plot, ttZcount, label=label[i], width = width, align='edge', bottom = stack_count, color=color[i])
            plt.legend(loc='upper right')
            stack_count = stack_count + count
            
    for i in range(len(sig_test)):
        if sig_test[i] == 1:
            step_count,edge = np.histogram(sig[i], bins=bins, weights=weight[bkg_length+i], density=normed)
            step_count_ext = np.append(step_count, step_count[len(step_count)-1])
            plt.step(bins, step_count_ext, label=label[bkg_length+i], color=color[bkg_length+i], where='post', linewidth = 2.0, linestyle=linestyle[i])
            if i == 4:
                step_count_save = step_count
    
    # Plot customisation
    if ytitle_in == 'Events':
        ytitle = ('Events' + " / {width:} GeV").format(width = int(width))
    else:
        ytitle = ('Events' + " / {width:.3f} rad").format(width = width)
    
    if normed:
        ytitle = ytitle + ' (Normalised)'
    
    plt.title(title, fontsize=40)
    plt.xlabel(xtitle, fontsize=25)
    plt.ylabel(ytitle, fontsize=25)
    plt.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1, .9), framealpha=1, edgecolor='k')
    ax.set_xticks(bins, minor=True)
    ax.grid(which='minor', axis='x', alpha = 0.5)
    ax.grid(which='major', axis='y', alpha = 0.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.yscale(scale)
    
    plt.xlim(bkg_minim, bkg_maxim)
    
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
        
    return ttZcount, step_count_save
        
        
# Stacked histograms with signal
#!!! Change green description

'''
X - Histrogram array
X1 - Signal array
Nb - Number of bins for the histogram
close - Whether to close the plot or not
label - labels for histogram with signal label on the end

kwargs - xtitle, ytitle, title, saveas
'''
def SVMHist(y_binary, model_prob, Nb, close, label, **kwargs):
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(11.0,9.0))
    global save_count
    
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
    
    color = ['cornflowerblue','red']
    
    # If color and saveas don't exist, give them arbitrary values
    if 'saveas' not in locals():
        saveas = ("unnamed_plot_%d" %save_count)
        save_count = save_count+1
        print('Warning, some plots have no save title')
        
    # Calculate bins and widths
    bins = np.linspace(0, 1, Nb)
    width = bins[1]-bins[0]
    
    bins_plot = np.delete(bins,Nb-1)

    plot = np.zeros(len(y_binary))
    z = 0
    
    for i in range(len(y_binary)-1):
        if y_binary[i] == 0:
            plot[z] = model_prob[i]
            z = z + 1
            
    mid_count = z
            
    for i in range(len(y_binary)-1):
        if y_binary[i] == 1:
            plot[z] = model_prob[i]
            z = z + 1
        
    # The predicted probability of signal for background events
    X = plot[0:mid_count]
    # The predicted probability of signal for signal events
    X1 = plot[mid_count:z]
    
    # Plot background histogram

    count,edge = np.histogram(X, bins=bins)
    plt.bar(bins_plot, count, label=label[0], width = width, align='edge', color=color[0])
    
    # Plot the signal

    step_count,edge = np.histogram(X1, bins=bins)
    step_count_ext = np.append(step_count, step_count[len(step_count)-1])
    plt.step(bins, step_count_ext, label=label[1], color=color[1], where='post', linewidth = 2.0, linestyle='dashed')
    
    # Plot customisation
    plt.title(title, fontsize=40)
    plt.xlabel(xtitle, fontsize=25)
    plt.ylabel(ytitle, fontsize=25)
    plt.legend(loc='upper right', fontsize=15, framealpha=1, edgecolor='k')
    plt.xlim(0, 1)
    ax.set_xticks(bins, minor=True)
    ax.grid(which='minor', axis='x', alpha = 0.5)
    ax.grid(which='major', axis='y', alpha = 0.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.yscale('log')

    # Save the figure, close if close is set true
    plt.savefig("Plots/"+saveas+".png")
    
    if close: 
        plt.close()
    else:
        plt.show()
        
    return count, step_count


'''
Function to prepare x and y data for machine learning
'''
def data_prep(variables, N_ttZ, signal_start, signal_end):
    
    N = 14627
    
    sig_length = signal_end - signal_start
    
    if len(variables) < 100:
        var_length = len(variables)
        all_variables = np.zeros((N, var_length))
    
        for i in range(var_length):
            all_variables[:,i] = variables[i]
    else:
        all_variables = variables
        all_variables  = all_variables.reshape((len(all_variables),1)) 
    
    background = all_variables[0:N_ttZ]
    signal = all_variables[signal_start:signal_end]
    
    zeros = np.zeros((N_ttZ,1))
    ones = np.ones((sig_length,1))
    
    xdata = np.concatenate((background,signal))
    ydata = np.concatenate((zeros,ones))
    
    data = np.hstack((xdata,ydata))
    
    # Shuffle the data
    np.random.shuffle(data)

    # Training and test splitting
    N = data.shape[0]
    N_train = int(2*N/3)

    # Separate into x and y
    y_pos = len(data[0])-1
    X = data[:,0:y_pos]
    y_binary = data[:,y_pos]
    
    ### Normalisation of X ###
    
    scaler = StandardScaler()
    scaler.fit(X)
    X_norm = scaler.transform(X)
    
    #Normalised dataset
    y_binary_data  = y_binary.reshape((len(y_binary),1)) 
    data_norm = np.hstack((X_norm,y_binary_data))
    
    ### No Normalisat of X ###
    # X_norm = X
    
    ###Both above
    
    X_train = X_norm[0:N_train,:]
    y_train = y_binary[0:N_train]
    
    X_test = X_norm[N_train:,:]
    y_test = y_binary[N_train:]
    
    ### Nikos method
    # X_test_sc = X[N_train:,:]
    # X_train_sc = X[0:N_train,:]
    # scaler = MinMaxScaler(feature_range=(0,1))
    
    # X_train = scaler.fit_transform(X_train_sc)
    
    # X_test = scaler.transform(X_test_sc)
    
    # y_train = y_binary[0:N_train]
    
    # y_test = y_binary[N_train:]
    
    # Convert y to integer form
    y_binary_int = y_binary.astype(int)
    
    # Return the data for visualisation, the training and test sample and the true binary
    return data, data_norm, X_train, y_train, X_test, y_test, y_binary_int
    
def ROC_Curve(y_binary, model_pred, tag):
    
    length = len(model_pred)
    N_train = int(2*length/3)
    
    y_pred_all = model_pred

    y_train = y_binary[0:N_train]
    y_pred_train = y_pred_all[0:N_train]
    
    y_test = y_binary[N_train:length]
    y_pred_test = y_pred_all[N_train:length]
    
    fpr_all, tpr_all, thresholds = metrics.roc_curve(y_binary, y_pred_all)
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
    plt.title('ROC_Curve_Model_'+tag, fontsize=40)
    plt.legend(loc='best', fontsize=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    plt.savefig("Plots/ROC_Curve_Model_"+tag+".png")
    
'''LIMIT CALCULATION'''
    
def calcSign(ns, nb, method=0, err=0.05):
    if method==0:
        return ns/np.sqrt(nb)
    elif method == 1:
        n = ns + nb
        if n >= nb:
            #return math.sqrt( 2.*2.* ( n*math.log(n/nb) - n + nb  )   )
            return np.sqrt( 2.* ( n*np.log(n/nb) - n + nb  )   )
        else:
            return 0.
    elif method == 2:
        n = ns + nb
        sig2 = ( nb*err ) **2 + nb
        den = nb*nb + n*sig2
        num = n*(nb + sig2)
        ll = sig2*(n-nb)/(nb*(nb+sig2))
        #print(n, nb, num/den, ll)
        #t1 = 4.*sqrt(2.)*2.* ( n*math.log( num/den ) - nb*nb * math.log( 1 + ll) / sig2  )
        t1 = 2.* ( n*np.log( num/den ) - nb*nb * np.log( 1 + ll) / sig2  )
        #print(t1, n*math.log( num/den ), nb*nb * math.log( 1 + ll) / sig2)
        if n >= nb:
            return np.sqrt(abs(t1)) # for precision
        else:
            0.
    return 99999


def getLimit(hbkg, hsig, confidenceLevel=0.95, method=0, err=0.05):
    N = len(hbkg)
    ns, nb = 0., 0.
    res = 0.
    for j in range(N):
        i = N - j - 1
        ns += hsig[i]
        nb += hbkg[i]
        if nb >= 3:
            #sign = ns/sqrt(nb)
            sign = calcSign(ns, nb, method, err)
            res += sign*sign
            #print('bin ',i, ns, nb, sign, res)
            ns, nb = 0., 0.
        else:
            continue
            
    s = norm.ppf(1-(1-confidenceLevel)*0.5)
    lim = s/np.sqrt(res)
    #if isinf(lim) and doIt:
    #    return getLimit(savgol_filter(hbkg,5,2), savgol_filter(hsig,5,2),
    #                    confidenceLevel, False)
        
    
    return lim
    
def makePlot(X1,X2, tag, Nb, close, **kwargs):

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.0,8.0))

    xtitle=tag
    title = tag
    for key, value in kwargs.items():
        if key == "xtitle":
            xtitle = value
        elif key=="title":
            title = value

    #Definition of variables
    themin = min( [min(X1), min(X2)])
    themax = max( [max(X1), max(X2)])
    bins = np.linspace(themin, themax, Nb)
    width = np.zeros(len(bins)-1)
    
    #Calculate bin centres and widths
    bincentre = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
            bincentre[i] = bins[i]+((bins[i+1]-bins[i])/2)
            width[i] = bins[i+1]-bins[i]

    #Offset the errorbars for background and signal from the centre so they don't overlap
    err_offset = 0.1*width
    
    #Set axis scale
    #ax.set_yscale('log')
    
    #Implement hatches for errors?
    #https://het.as.utexas.edu/HET/Software/Matplotlib/api/patches_api.html#matplotlib.patches.Patch.set_hatch
    
    
    #Background plot
    plt.hist(X1, bins=bins, label=['Background'],density=True)
    n_back_pos, edge = np.histogram(X1, bins=Nb-1, range=(themin,themax),density=True)
    n_back_count, edge1 = np.histogram(X1, bins=Nb-1, range=(themin,themax))
    back_err = np.sqrt(n_back_count)/(np.sum(n_back_count)*width)
    ax.errorbar(bincentre-err_offset, n_back_pos, xerr=None, yerr=back_err, ls='none', ecolor='k', fmt = 'ko')
    
    
    #Signal plot
    plt.hist(X2, bins=bins, label=['Signal'], histtype=u'step',density=True)
    n_sig_pos, edge2 = np.histogram(X2, bins=Nb-1, range=(themin,themax),density=True)
    n_sig_count, edge3 = np.histogram(X2, bins=Nb-1, range=(themin,themax))
    sig_err = np.sqrt(n_sig_count)/(np.sum(n_sig_count)*width)
    ax.errorbar(bincentre+err_offset, n_sig_pos, xerr=None, yerr=sig_err, ls='none', ecolor='r', fmt = 'ro')

    #Calculate maximum value for y
    ymax = max([(max(n_back_pos)+max(back_err)), (max(n_sig_pos)+max(sig_err))])

    plt.title(title, fontsize=40)
    plt.xlabel(xtitle, fontsize=25)
    plt.ylabel("Entries (Norm)", fontsize=25)
    plt.legend(loc='upper right')
    plt.xlim(themin, themax)
    plt.ylim(0, 1.2*ymax)
    ax.set_xticks(bins, minor=True)
    ax.grid(which='minor', axis='x', alpha = 0.5)
    ax.grid(which='major', axis='y', alpha = 0.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig("Plots/"+tag+".png")

    if close: 
        plt.close()
    else:
        plt.show()

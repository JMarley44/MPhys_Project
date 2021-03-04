# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:12:22 2020

@author: James
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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


#!!! tag isnt used
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
    
    # Take every other value of the bin for the major ticks
    major_ticks = bins[::2]

    # Plot the histogram
    plt.hist(X, bins=bins, label=label)

    # Plot customisation
    plt.title(title, fontsize=40)
    plt.xlabel(xtitle, fontsize=25)
    plt.ylabel(ytitle, fontsize=25)
    plt.legend(loc='upper right')
    
    ax.set_xticks(major_ticks)
    ax.set_xticks(bins, minor=True)
    
    ax.grid(which='both', axis='x', alpha = 0.5)
    ax.grid(which='major', axis='y', alpha = 0.5)
    
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.xlim(themin, themax)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.yscale(scale)
    plt.savefig("Plots/"+title+".png")
    if close: plt.close()
    
###############################################################################

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
        
# Note for better bins use (xmax-xmin)/(Nb-1) = Bin width

def SignalHist(X, weight, Nb, N_arr, close, **kwargs):
    
    N_ttZ = N_arr[0]
    N_ttWm = N_arr[1]
    N_ttWp = N_arr[2]
    N_ggA_460_360 = N_arr[3]
    N_ggA_500_360 = N_arr[4]
    N_ggA_600_360 = N_arr[5]
    N_ggA_600_400 = N_arr[6]
    N_ggA_600_500 = N_arr[7]
    N_ggA_500_400 = N_arr[8]
    
    label = [r'other',r't$\bar{t}$Z','ggA ($m_A$=460, $m_H$=360)', 'ggA ($m_A$=500, $m_H$=360)', 'ggA ($m_A$=600, $m_H$=360)'
         , 'ggA ($m_A$=600, $m_H$=400)', 'ggA ($m_A$=600, $m_H$=500)', 'ggA ($m_A$=500, $m_H$=400)']
    
    color = ['lightgreen','cyan','cornflowerblue','red','indigo','gold','black','maroon']
    
    # Test whether a given bkg/sig is present
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
                    print('Unknown bkg/sig in SignalHist')

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

    # Calculate bins and widths
    bins = np.linspace(bkg_minim, bkg_maxim, Nb)
    width = bins[1]-bins[0]
    
    # Take every other value of the bin for the major tick labels
    major_ticks = bins[::2]

    # Bin centres
    bins_plot = np.delete(bins,Nb-1)
    bins_centre = bins_plot + (width/2)
    
    # Define an area array for normalisation:
    area = np.zeros(bkg_length)
    bkg_weight_sum = np.zeros(bkg_length)
    
    # Define stackable counts and errors
    stack_count = 0  
    final_error = 0

    for i in range(bkg_length):
        if bkg_test[i] == 1:
            # Calculation of the max counts
            count,edge = np.histogram(bkg[i], bins=bins, weights=weight[i])
            # Total area for each histogram
            area[i] = np.sum(count * width)
            bkg_weight_sum[i] = np.sum(weight[i])
            
    # Total area for all histograms
    all_area = np.sum(area)
    bkg_weight_tot = np.sum(bkg_weight_sum)
            
    for i in range(bkg_length):
        if bkg_test[i] == 1:
            # Calculation of the counts
            count,edge = np.histogram(bkg[i], bins=bins, weights=weight[i])

            if normed:
                # Normalise all backgrounds to the sum of their weights
                count = (count/all_area)*bkg_weight_tot
                N =  ( np.sum(weight[i]) *(bins[-1]-bins[0]) ) / len(count)
            else:
                N = 1

            # Save the other and ttZ counts for return
            if i == 0:
                other_count = count

            if i == 1:
                ttZcount = count
                
            # Calculate the error
            error = np.sqrt((np.histogram(bkg[i], bins=bins, 
                                              weights=weight[i]*weight[i])[0]).astype(float) / N)
            
            # Addition of all background errors
            final_error = np.sqrt((final_error**2) + (error**2))
    
            # Make the plot
            plt.bar(bins_plot, count, label=label[i], width = width, align='edge', 
                    bottom = stack_count, color=color[i])
            
            # Add errorbars if it's the final background
            if i == (bkg_length-1):
                plt.errorbar(bins_centre, count+stack_count, barsabove=True, ls='none', 
                              yerr=final_error, marker='+',color='red')

            # Add the count to the stack for the next iteration
            stack_count = stack_count + count
            
    for i in range(len(sig_test)):
        if sig_test[i] == 1:
            # Calculation of the counts
            step_count,edge = np.histogram(sig[i], bins=bins, weights=weight[bkg_length+i],
                                           density=normed)
            
            # Normalise to the sum of the weights if plot is normalised
            if normed:
                step_count = step_count*np.sum(weight[bkg_length+i])
                
            # Extend the step in x to reach the edge of the plot
            step_count_ext = np.append(step_count, step_count[len(step_count)-1])
            
            # Make the plot
            plt.step(bins, step_count_ext, label=label[bkg_length+i], color=color[bkg_length+i], 
                     where='post', linewidth = 2.0, linestyle=linestyle[i])
            
            # Save the signal count for specific signal for return
            # Needs to be changed to extract a given signal
            if i == 4:
                sigcount = step_count
    
    
    # Plot customisation
    
    if ytitle_in == 'Events':
        # If a GeV plot use GeV bins and decimal format
        ytitle = ('Events' + " / {width:} GeV").format(width = int(width))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.d'))
    else:
        # If a rad plot use rad bins and float format
        ytitle = ('Events' + " / {width:.3f} rad").format(width = width)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    if normed:
        ytitle = ytitle + ' (Normalised)'
    
    plt.title(title, fontsize=40)
    plt.xlabel(xtitle, fontsize=25)
    plt.ylabel(ytitle, fontsize=25)
    plt.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1, .9), framealpha=1, edgecolor='k')
    
    # Define ticks for numbering (major only) and grids (both)
    ax.set_xticks(major_ticks)
    ax.set_xticks(bins, minor=True)
    
    # Add axes grids to x and y
    ax.grid(which='both', axis='x', alpha = 0.5)
    ax.grid(which='major', axis='y', alpha = 0.5)
        
    # Rotate the x labels 45 degrees
    #fig.autofmt_xdate(rotation=45)
        
    # Set axis font sizes
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Set y scale
    plt.yscale(scale)
    
    # Set x limit 
    plt.xlim(bkg_minim, bkg_maxim)
    # Set y limit, upper limit is automatic, non-zero for a log scale
    if scale == 'log':
        plt.gca().set_ylim(bottom=0.01)
    else:
        plt.gca().set_ylim(bottom=0)
    
    # String for adding to the text box
    string = ('$\u221As = 13 TeV, 139$ $fb^{-1}$')
    
    # Add a text box
    ax.text(.97, .97, string, transform=ax.transAxes, fontsize=10, fontweight='bold', horizontalalignment='right',
            verticalalignment='top', bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='round,pad=1'))
    
    # Save the figure, close if close is set true
    plt.savefig("Plots/"+saveas+".png")
    if close: 
        plt.close()
    else:
        plt.show()    
        
    # Return the counts, if required
    return ttZcount, sigcount
        
# Alternative errorbars wrt axes
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html
# errorevery may be more suitable than erroroffset in future

# ax.errorbar(bins_centre+err_offset[i], (count+stack_count), xerr=None, yerr=error, 
#             ls='none', ecolor='err_color[i], fmt = 'k+')
        
###############################################################################

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
def SVMHist(y_binary, model_prob, Nb, weight, N_arr, close, label, **kwargs):
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(11.0,9.0))
    global save_count
    
    # Only need the ttZ background
    N_ttZ = N_arr[0]
    N_ttWp = N_arr[2]
    N_ggA_460_360 = N_arr[3]
    N_ggA_500_360 = N_arr[4]
    N_ggA_600_360 = N_arr[5]
    N_ggA_600_400 = N_arr[6]
    N_ggA_600_500 = N_arr[7]
    N_ggA_500_400 = N_arr[8]
    
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

    # Extract the ttZ weight
    ttZ_weight = (weight[0:N_ttZ])[0]


    # Extract the signal weights based on the label
    # Take the first value since they're all the same per signal

    if label[1] == 'ggA_460_360':
        sig_weight = (weight[N_ttWp:N_ggA_460_360])[0]

    elif label[1] == 'ggA_500_360':
        sig_weight = (weight[N_ggA_460_360:N_ggA_500_360] )[0]
        
    elif label[1] == 'ggA_600_360':
        sig_weight = (weight[N_ggA_500_360:N_ggA_600_360])[0]

    elif label[1] == 'ggA_600_400':
        sig_weight = (weight[N_ggA_600_360:N_ggA_600_400] )[0]
        
    elif label[1] == 'ggA_600_500':
        sig_weight = (weight[N_ggA_600_400:N_ggA_600_500])[0]
        
    elif label[1] == 'ggA_500_400':
        sig_weight = (weight[N_ggA_600_500:N_ggA_500_400])[0]
        
    


    color = ['cyan','red']
    
    # If saveas doesn't exist, give them arbitrary values
    if 'saveas' not in locals():
        saveas = ("unnamed_plot_%d" %save_count)
        save_count = save_count+1
        print('Warning, some plots have no save title')
        
    # Calculate bins and widths
    bins = np.linspace(0, 1, Nb)
    width = bins[1]-bins[0]
    
    bins_plot = np.delete(bins,Nb-1)

    # Take every other value of the bin for the major ticks
    major_ticks = bins[::2]
        
    # The predicted probability of signal for background events
    X = model_prob[y_binary==0]
    # The predicted probability of signal for signal events
    X1 = model_prob[y_binary==1]
    
    # Plot background histogram

    count,edge = np.histogram(X, bins=bins)
    
    # Apply the weight
    count = count*ttZ_weight
    
    plt.bar(bins_plot, count, label=label[0], width = width, align='edge', color=color[0])
    
    # Plot the signal
    step_count,edge = np.histogram(X1, bins=bins)
    
    # Apply the weight
    step_count = step_count*sig_weight
    
    step_count_ext = np.append(step_count, step_count[len(step_count)-1])
    plt.step(bins, step_count_ext, label=label[1], color=color[1], where='post', 
             linewidth = 2.0, linestyle='dashed')
    
    # Plot customisation
    plt.title(title, fontsize=40)
    plt.xlabel(xtitle, fontsize=25)
    plt.ylabel(ytitle, fontsize=25)
    plt.legend(loc='upper right', fontsize=15, framealpha=1, edgecolor='k')
    plt.xlim(0, 1)
    
    # Define ticks for numbering (major only) and grids (both)
    ax.set_xticks(major_ticks)
    ax.set_xticks(bins, minor=True)
    
    # Add axes grids to x and y
    ax.grid(which='both', axis='x', alpha = 0.5)
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

# IN CASE I WANT TO PLOT THEM ON THE SAME PLOT, FOR MAIN

# model_1_probs = (model_1_prob_train,model_1_prob_test)

# model_test = type(model_1_probs) is tuple

# model_all_test = type(model_1_all_prob) is tuple

# print(model_test)
# print(model_all_test)



# OLD SEPARATION
# plot = np.zeros(len(y_binary))
# z = 0

# for i in range(len(y_binary)-1):
#     if y_binary[i] == 0:
#         plot[z] = model_prob[i]
#         z = z + 1
        
# mid_count = z
        
# for i in range(len(y_binary)-1):
#     if y_binary[i] == 1:
#         plot[z] = model_prob[i]
#         z = z + 1


'''
Function to prepare x and y data for machine learning
'''
def data_prep(variables, N_ttZ, N, signal_start, signal_end):

    #!!! Put N_arr into here    

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
    
    ### Normalisation ###
    ###  Choose one   ###
    
    ### No Normalistation ###
    # X_norm = X
    
    ###   Standard scalar  ###
    # scaler = StandardScaler()
    # scaler.fit(X)
    # X_norm = scaler.transform(X)

    ###   Scalar 0-1   ###
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(X)
    X_norm = scaler.transform(X)

    # Get train and test data
    
    X_train = X_norm[0:N_train,:]
    y_train = y_binary[0:N_train]
    
    X_test = X_norm[N_train:,:]
    y_test = y_binary[N_train:]

    
    # TESTING
    # X_train = X[0:N_train,:]
    # y_train = y_binary[0:N_train]
    
    # X_test = X[N_train:,:]
    # y_test = y_binary[N_train:]

    # scaler = MinMaxScaler(feature_range=(0,1))

    # X_train_sc = scaler.fit_transform(X_train)
    # X_test_sc = scaler.transform(X_test)
    
    # X_norm = np.concatenate((X_train_sc,X_test_sc))

    #Normalised dataset
    y_binary_data  = y_binary.reshape((len(y_binary),1)) 
    data_norm = np.hstack((X_norm,y_binary_data))
    
    # Convert y to integer form
    y_binary_int = y_binary.astype(int)
    
    # Return the data for visualisation, the training and test sample and the true binary
    return data, data_norm, X_train, y_train, X_test, y_test, y_binary_int
    
def ROC_Curve(y_binary, model_pred, close, tag):
    
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
    
    if close: 
        plt.close()
    else:
        plt.show()
    
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
    
'Old make plot function for NN'
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

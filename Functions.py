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

def dphi(phi1, phi2):
    dphi = phi1-phi2
    
    # Try for an array, except float (for lepton calculcations)
    
    try:
        length = len(dphi)
    
        for i in range(length):
            if dphi[i]>np.pi:
                dphi[i] = dphi[i]-np.pi
            elif dphi[i]<(-1*np.pi):
                dphi[i] = dphi[i]+np.pi
                
    except:
        if dphi>np.pi:
                dphi = dphi-np.pi
        elif dphi<(-1*np.pi):
                dphi = dphi+np.pi
            
    return dphi

def drangle(dphi, eta1, eta2):
    deta = eta1-eta2
    
    # Try for an array, except float (for lepton calculcations)
    
    try:
        length = len(deta)

        deta.astype(float)
        dphi.astype(float)
        
        drangle = np.zeros(length)
        
        for i in range(length):
            drangle[i] = np.sqrt((dphi[i]**2)+(deta[i]**2))
              
    except:

        drangle = np.sqrt((dphi**2)+(deta**2))
    
    return drangle

# Single histogram plot for the Z invariant #!!! Needs error bars and luminosity scaling
def Hist(X, Nb, close, **kwargs):
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
        elif key=="label":
            label = value
    
    if 'scale' not in locals():
        scale = 'linear'

    # Calculate the bins and width
    bins = np.linspace(themin, themax, Nb)
    width = bins[1]-bins[0]
    
    # Take every other value of the bin for the major tick labels
    major_ticks = bins[::2]

    # Bin centres
    bins_plot = np.delete(bins,Nb-1)
    bins_centre = bins_plot + (width/2)

    # Make the histogram
    count,edge = np.histogram(X, bins=bins)

    # Plot the histogram
    plt.hist(X, bins=bins)
    
    # Add error bars
    error = np.sqrt(count)
    plt.errorbar(bins_centre, count, barsabove=True, ls='none', 
                              yerr=error, marker='+', color='red')

    # Plot customisation
    plt.title(title, fontsize=40)
    plt.xlabel(xtitle, fontsize=25)
    plt.ylabel(ytitle, fontsize=25)
    
    ax.set_xticks(major_ticks)
    ax.set_xticks(bins, minor=True)
    
    ax.grid(which='both', axis='x', alpha = 0.5)
    ax.grid(which='major', axis='y', alpha = 0.5)
    
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.xlim(themin, themax)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.yscale(scale)
    plt.savefig("Plots/Singular/"+title+".png")
    
    if close: 
        plt.close()
    else:
        plt.show()   
    
###############################################################################

# Stacked histograms with signal
'''
X - Input data array
weight - weights of dataset
Nb - Number of bins for the histogram
N_arr - The array of N for separation
close - Whether to close the plot or not

kwargs - xtitle, ytitle, title, saveas, shaped, signals

shaped - whether to do a shape comparison
signals - what signal types to plot
'''
        
# Note for better bins could use (xmax-xmin)/(Nb-1) = Bin width

def SignalHist(X, weight, Nb, N_arr, close, shaped=False, limit=False, **kwargs):
    
    N_ttZ = N_arr[0]
    N_ttWm = N_arr[1]
    N_ttWp = N_arr[2]
    N_ggA_460_360 = N_arr[3]
    N_ggA_500_360 = N_arr[4]
    N_ggA_600_360 = N_arr[5]
    N_ggA_600_400 = N_arr[6]
    N_ggA_600_500 = N_arr[7]
    N_ggA_500_400 = N_arr[8]
    N_ggA_500_400_1 = N_arr[9]
    
    label = [r'other',r't$\bar{t}$Z','ggA ($m_A$=460, $m_H$=360)', 'ggA ($m_A$=500, $m_H$=360)', 'ggA ($m_A$=600, $m_H$=360)'
         , 'ggA ($m_A$=600, $m_H$=400)', 'ggA ($m_A$=600, $m_H$=500)', 'ggA ($m_A$=500, $m_H$=400)','ggA ($m_A$=500, $m_H$=400) 1']
    
    color = ['lightgreen','cyan','cornflowerblue','red','indigo','gold','black','maroon','darkorange']
    
    # Test whether a given bkg/sig is present
    bkg_test = np.zeros(2) # other, ttZ
    sig_test = np.zeros(7) # 460_360, 500_360, 600_360, 600_400, 600_500, 500_400, 500_400_1
    
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
    X_500_400_1 = None
    weight_460_360 = None
    weight_500_360 = None
    weight_600_360 = None
    weight_600_400 = None
    weight_600_500 = None
    weight_500_400 = None
    weight_500_400_1 = None
    
    addText = False
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
        elif i =="line":
            linestyle = j
        elif i =="scale":
            scale = j
        elif i =='xlim':
            bkg_minim = j[0]
            bkg_maxim = j[1]
        elif i =='addText':
            addText = True
            add_string = j
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
                elif j[k] == '500_400_1':
                    X_500_400_1 = X[N_ggA_500_400:N_ggA_500_400_1]
                    weight_500_400_1 = weight[N_ggA_500_400:N_ggA_500_400_1]
                    sig_test[6] = 1
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
        
    # Form tuples
    bkg = ([X_other, X_ttZ])
    sig = ([X_460_360, X_500_360, X_600_360, X_600_400, X_600_500, X_500_400, X_500_400_1])
    weight = ([weight_other, weight_ttZ, weight_460_360, weight_500_360,weight_600_360,
               weight_600_400, weight_600_500, weight_500_400, weight_500_400_1])
    
    # Lengths of data
    bkg_length = len(bkg)
    sig_length = len(sig_test)
    
    # If max and min aren't defined, find them
    if 'bkg_maxim' not in locals():
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
    
    # Take every other value of the bin for the major tick labels
    major_ticks = bins[::2]

    # Bin centres
    bins_plot = np.delete(bins,Nb-1)
    bins_centre = bins_plot + (width/2)
    
    # Define arrays for shaping:
    bkg_count_sum = np.zeros(bkg_length)
    sig_count_sum = np.zeros(sig_length)
    bkg_weight_sum = np.zeros(bkg_length)
    
    # Define stackable counts and errors for multiple bkg
    stack_count = 0  
    final_error = 0
    
    # Empty array for signal return and empty ttZ for non-shaped plots
    sigcount = np.zeros((7,(Nb-1)))
    ttZcount = None
    
    # If it's a limit calculation scale ifb to 300, otherwise leave as 139
    if limit:
        ifb = 300
    else:
        ifb = 139

    ###################
    # PLOT BACKGROUND #
    ###################

    # Preliminary for shape comparison
    # Outside the main loop to avoid calculating sums multiple times
    if shaped:
        for i in range(bkg_length):
            if bkg_test[i] == 1:
                # Calculation of the counts
                count,edge = np.histogram(bkg[i], bins=bins, weights=weight[i]*0.5*(ifb/139))
                # Total weights for each histogram
                bkg_weight_sum[i] = np.sum(weight[i])
                # Total count for each histogram
                bkg_count_sum[i] = np.sum(count)
                
        # Total count for all bkg histograms
        all_bkg_count = np.sum(bkg_count_sum)
        # Total weight for all bkg histograms
        bkg_weight_tot = np.sum(bkg_weight_sum)
    
    # Main bkg loop
    for i in range(bkg_length):
        if bkg_test[i] == 1:
            
            # Scale the bkg weights to ifb, half them since there's 2*139 ifb
            weight[i] = weight[i]*0.5*(ifb/139)
            
            # Calculation of the counts
            count,edge = np.histogram(bkg[i], bins=bins, weights=weight[i])

            # If it's a limit calculation save the ttZ count for return
            if limit and i==1:
                ttZcount = count
                
            # Calculate the error
            error = np.sqrt((np.histogram(bkg[i], bins=bins, 
                                              weights=weight[i]*weight[i])[0]).astype(float))
            
            # Addition of all background errors in quadrature
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
            
    ###############
    # PLOT SIGNAL #
    ###############
            
    for i in range(sig_length):
        if sig_test[i] == 1:

            # Normalise to the background if shaped
            if shaped:
                step_count,edge = np.histogram(sig[i], bins=bins)
                sig_count = np.sum(step_count)
                step_count = step_count*(all_bkg_count/sig_count)
            # Otherwise plot an event comparison
            else:
                # Scale the sig weights to ifb
                weight[bkg_length+i] = weight[bkg_length+i]*(ifb/139)
                
                # Histogram the signal
                step_count,edge = np.histogram(sig[i], bins=bins, weights=weight[bkg_length+i])
         
            # Extend the step in x to reach the edge of the plot
            step_count_ext = np.append(step_count, step_count[len(step_count)-1])
            
            # Make the plot
            plt.step(bins, step_count_ext, label=label[bkg_length+i], color=color[bkg_length+i], 
                     where='post', linewidth = 2.0, linestyle=linestyle[i])
            
            # Save the signal count for specific signal for return
            #!!! Needs to be changed to extract a given signal
            if not shaped:
                sigcount[i,:] = step_count
                
                # Multiple sig counts need to returned in the future
                #sigcount[i] = step_count
    
    
    # Plot customisation
    if ytitle_in == 'Events':
        # If a GeV plot use GeV bins and decimal format
        ytitle = ('Events' + " / {width:} GeV").format(width = int(width))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.d'))
        plt.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1, .92), framealpha=1, edgecolor='k')
    else:
        # If a rad plot use rad bins and float format
        ytitle = ('Events' + " / {width:.3f} rad").format(width = width)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1, .92), framealpha=1, edgecolor='k')
    
    if shaped:
        plt.title(title + ' - shape comparison', fontsize=40)
    else:
        if limit:
            plt.title(title + ' - limit plot', fontsize=40)
        else:
            plt.title(title + ' - event comparison', fontsize=40)
    
    plt.xlabel(xtitle, fontsize=25)
    plt.ylabel(ytitle, fontsize=25)
    
    # Define ticks for numbering (major only) and grids (both)
    ax.set_xticks(major_ticks)
    ax.set_xticks(bins, minor=True)
    
    # Add axes grids to x and y
    ax.grid(which='both', axis='x', alpha = 0.5)
    ax.grid(which='major', axis='y', alpha = 0.5)
        
    # Rotate the x labels 45 degrees
    # fig.autofmt_xdate(rotation=45)
        
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
    # if shaped:
    #     #string = '\n'.join(('ATLAS','$\u221As = 13 TeV, 36.1$ $fb^{-1}$','$m_H$ $=130 GeV,$ $n_b$ $= 2$'))
    #     string = ('\u221As = 13 TeV, 139 $fb^{{-1}}$ \n Signal normalized to the background yield')
    # else:
        
    # Define a string with the used luminosity scaling
    string = ('$\u221As = 13 TeV, {lumin:}$ $fb^{{-1}}$'.format(lumin=int(ifb)))
    
    # Add a text box
    # fontweight='bold', 
    ax.text(.97, .97, string, transform=ax.transAxes, fontsize=10, horizontalalignment='right',
            verticalalignment='top', bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='round,pad=1'))
    
    # Add an additional text box if in kwargs
    if addText:
        # ax.text(.97, .50, add_string, transform=ax.transAxes, fontsize=10, horizontalalignment='right',
        #     verticalalignment='top', bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='round,pad=1'))
        
        # fig.text(.5, .05, add_string, ha='center')
        # fig.set_size_inches(11, 10, forward=True)
        
        plt.figtext(0.5, 0.01, add_string, wrap=True, horizontalalignment='center', fontsize=8)
    
    # Save the figure to the relevant folder 
    if shaped:
        plt.savefig("Plots/Shapes/"+saveas+".png")
    else:
        if limit:
            plt.savefig("Plots/Limits/"+saveas+".png")
        else:
            plt.savefig("Plots/Events/"+saveas+".png")
        
    # Close if close is set true
    if close: 
        plt.close()
    else:
        plt.show()    
        
    # Return the counts, if required
    return ttZcount, sigcount
        
        
###############################################################################



'''
Function to prepare x and y data for machine learning
'''

def data_prep(variables, N_ttZ, N, weight, signal_start, signal_end, train_percent=0.5, cut_percent=0):
    
    # Cut some of the signal data to show whether or not the # of events is sufficient
    cut = 1-cut_percent

    sig_length = int((signal_end - signal_start)*cut)
    
    # Test if the input discriminant is singular or packed into a tuple
    # Assign it to the 'all_variables' variable
    if isinstance(variables, tuple):
        var_length = len(variables)
        all_variables = np.zeros((N, var_length))
    
        for i in range(var_length):
            all_variables[:,i] = variables[i]
    else:
        all_variables = variables
        all_variables = all_variables.reshape((len(all_variables),1)) 
    
    # Split to bkg and sig
    background = all_variables[0:N_ttZ]
    signal = all_variables[signal_start:signal_end]
    
    bkg_w = weight[0:N_ttZ]
    sig_w = weight[signal_start:signal_end]
    
    # Normalise the weights
    # Just use float64 and round later to manage float point err
    mean_bkg_w = np.mean(bkg_w, dtype=np.float64)
    mean_sig_w = np.mean(sig_w, dtype=np.float64)
    
    bkg_w_norm = (bkg_w/mean_bkg_w).astype('float32')
    sig_w_norm = (sig_w/mean_sig_w).astype('float32')
    
    # Perform the cut if it has value
    signal = signal[0:sig_length]
    
    # Assign the background value 0 and the signal value 1
    zeros = np.zeros((N_ttZ,1))
    ones = np.ones((sig_length,1))
    
    # Combine backgrounds and signals
    xdata = np.concatenate((background,signal))
    ydata = np.concatenate((zeros,ones))
    wdata = np.concatenate((bkg_w,sig_w))
    w_norm_data = np.concatenate((bkg_w_norm,sig_w_norm))
    
    # Reshape the w data for merging
    wdata = wdata.reshape((len(wdata),1)) 
    w_norm_data = w_norm_data.reshape((len(w_norm_data),1)) 
    
    # Add y and weights to the data for shuffling together
    data = np.append(xdata,ydata, axis=1)
    data = np.append(data,wdata, axis=1)
    data = np.append(data,w_norm_data, axis=1)

    # Shuffle the data
    np.random.shuffle(data)

    # Define training and test splitting
    N_data = data.shape[0]
    N_train = int((train_percent)*N_data)

    # Separate the normalised weights out
    w_norm_pos = len(data[0])-1
    w_norm_shuff = data[:,w_norm_pos]
    
    w_norm_train = w_norm_shuff[0:N_train]
    w_norm_test = w_norm_shuff[N_train:]

    # Remove the normalised weights from the data
    data = np.delete(data,w_norm_pos, axis=1)

    # Separate the weights out
    w_pos = len(data[0])-1
    w_shuff = data[:,w_pos]
    
    w_train = w_shuff[0:N_train]
    w_test = w_shuff[N_train:]
    
    # Remove the weights from the data
    data = np.delete(data,w_pos, axis=1)

    # Separate into x and y
    y_pos = len(data[0])-1
    X = data[:,0:y_pos]
    y_binary = data[:,y_pos]
    
    ### Normalisation ###
    ###  Choose one   ###
    
    ### No Normalistation ###
    # X_norm = X
    
    ###   Standard scalar  ###
    scaler = StandardScaler()

    ###   Scalar 0-1   ###
    # Cheating a bit in scaling all of the data,
    # instead of fitting the transformation on the training set and
    # just applying it on the test set.
    
    # scaler = MinMaxScaler(feature_range=(0,1))
    
    scaler.fit(X)
    X_norm = scaler.transform(X)

    # Get train and test data
    
    X_train = X_norm[0:N_train,:]
    y_train = y_binary[0:N_train]
    
    X_test = X_norm[N_train:,:]
    y_test = y_binary[N_train:]
    
    ### Scalar actual ###
    # X_train_sc = X[0:N_train,:]
    # y_train = y_binary[0:N_train]
    
    # X_test_sc = X[N_train:,:]
    # y_test = y_binary[N_train:]

    # scaler = MinMaxScaler(feature_range=(0,1))

    # X_train = scaler.fit_transform(X_train_sc)
    # X_test = scaler.transform(X_test_sc)
    
    # X_norm = np.concatenate((X_train,X_test))

    #Normalised dataset
    y_binary_data  = y_binary.reshape((len(y_binary),1)) 
    data_norm = np.hstack((X_norm,y_binary_data))
    
    # Convert y to integer form
    y_binary_int = y_binary.astype(int)
    
    # Return the data for visualisation, the training and test sample and the true binary
    return data, data_norm, X_train, y_train, X_test, y_test, w_train, w_test, w_norm_train, y_binary_int, N_train

    
def ROC_Curve(pred_train, pred_test, y_train, y_test, close, title, saveas, **kwargs):
    addTextbool = False
    
    for i, j in kwargs.items():
        if i=="addText":
            addTextbool = True
            addText = j
    
    pred_all = np.concatenate([pred_train,pred_test])
    y_all = np.concatenate([y_train,y_test])
    
    y_all = np.asarray(y_all).astype('int')
    y_train = np.asarray(y_train).astype('int')
    y_test = np.asarray(y_test).astype('int')
    
    fpr_all, tpr_all, thresholds = metrics.roc_curve(y_all, pred_all)
    fpr_train, tpr_train, thresholds = metrics.roc_curve(y_train, pred_train)
    fpr_test, tpr_test, thresholds = metrics.roc_curve(y_test, pred_test)
    
    auc_all = metrics.auc(fpr_all, tpr_all)
    auc_train = metrics.auc(fpr_train, tpr_train)
    auc_test = metrics.auc(fpr_test, tpr_test)
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(11.0,9.0))
    plt.plot(fpr_all, tpr_all, label='All (area = {:.3f})'.format(auc_all), color='red')
    plt.plot(fpr_test, tpr_test, label='Test (area = {:.3f})'.format(auc_test), color='deepskyblue')
    plt.plot(fpr_train, tpr_train, label='Train (area = {:.3f})'.format(auc_train), color='limegreen')
    plt.xlabel('False positive rate', fontsize=25)
    plt.ylabel('True positive rate', fontsize=25)
    plt.title(title + ' ROC Curve', fontsize=40)
    plt.legend(loc='best', fontsize=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
     # if there's additional text put it in a text box
    if addTextbool:
        ax.text(.55, .97, addText, transform=ax.transAxes, fontsize=10, horizontalalignment='right',
        verticalalignment='top', bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='round,pad=1'))
    
    plt.show()
    plt.savefig("Plots/"+saveas+"ROC"+".png")
    
    if close: 
        plt.close()
    else:
        plt.show()
        
    return auc_test
    

def ProbHist(pred_train, pred_test, y_train, y_test, w_train, w_test, Nb, close, label, shaped=True, **kwargs):
    
    # All pred, y and weights
    pred_all = np.concatenate([pred_train,pred_test])
    y_all = np.concatenate([y_train,y_test])
    w_all = np.concatenate([w_train,w_test])
    
    ### Separate bkg and sig for each ###
    
    # All bkg and sig
    pred_all_bkg = pred_all[y_all==0]
    pred_all_sig = pred_all[y_all==1]
    
    w_all_bkg = w_all[y_all==0]
    w_all_sig = w_all[y_all==1]

    # Train bkg and sig
    pred_train_bkg = pred_train[y_train==0]
    pred_train_sig = pred_train[y_train==1]
    
    w_train_bkg = w_train[y_train==0]
    w_train_sig = w_train[y_train==1]
    
    # Test bkg and sig
    pred_test_bkg = pred_test[y_test==0]
    pred_test_sig = pred_test[y_test==1]
    
    w_test_bkg = w_test[y_test==0]
    w_test_sig = w_test[y_test==1]

    ### Pack into tuples ###
    bkg = (pred_all_bkg, pred_train_bkg, pred_test_bkg)
    sig = (pred_all_sig, pred_train_sig, pred_test_sig)
    w_bkg = (w_all_bkg, w_train_bkg, w_test_bkg)
    w_sig = (w_all_sig, w_train_sig, w_test_sig)
    
    # Define an additional title and save term for each
    add_term = ['all','train','test']
    
    ##############
    # PLOT SETUP #
    ##############

    global save_count
    
    # Calculate bins and widths
    bins = np.linspace(0, 1, Nb)
    width = bins[1]-bins[0]
    
    bins_plot = np.delete(bins,Nb-1)

    # Take every other value of the bin for the major ticks
    major_ticks = bins[::2]
    
    # Initialise addTextbool
    addTextbool = False
    
    ###################
    # DATA EXTRACTION #
    ###################
    
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
        elif i=="addText":
            addTextbool = True
            addText = j

    color = ['cyan','red']
    
    # If saveas doesn't exist, give them arbitrary values
    if 'saveas' not in locals():
        saveas = ("unnamed_plot_%d" %save_count)
        save_count = save_count+1
        print('Warning, some plots have no save title')
        
    for i in range(3):
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(11.0,9.0))
        
        ###################
        # PLOT BACKGROUND #
        ###################

        # Reshape
        current_bkg_weight = w_bkg[i].reshape((len(w_bkg[i]),1)) 
        current_bkg = bkg[i].reshape((len(bkg[i]),1)) 

        # Calculate counts
        count,edge = np.histogram(current_bkg, bins=bins, weights=current_bkg_weight)
        
        # Calculate the sum for shape comparison with signal
        if shaped:
            bkg_count_sum =np.sum(count)
        
        # Make plot
        plt.bar(bins_plot, count, label=label[0], width = width, align='edge', color=color[0])
    
        ###############
        # PLOT SIGNAL #
        ###############
        
        # Reshape
        current_sig_weight = w_sig[i].reshape((len(w_sig[i]),1))
        current_sig = sig[i].reshape((len(sig[i]),1)) 
        
        # Calculate counts
        step_count,edge = np.histogram(current_sig, bins=bins, weights=current_sig_weight)
        
        # Calculate the sum for shape comparison with signal
        if shaped:
            sig_count_sum =np.sum(step_count)
        
            # Apply shape comparison (equate integrals)
            step_count = step_count*(bkg_count_sum/sig_count_sum)
        
        # Make plot
        step_count_ext = np.append(step_count, step_count[len(step_count)-1])
        plt.step(bins, step_count_ext, label=label[1], color=color[1], where='post', 
                 linewidth = 2.0, linestyle='dashed')
        
        ######################
        # PLOT CUSTOMISATION #
        ######################
        
        # Titles and legend
        plt.title(title+add_term[i], fontsize=40)
        plt.xlabel(xtitle, fontsize=25)
        plt.ylabel(ytitle, fontsize=25)
        plt.legend(loc='upper right', fontsize=15, framealpha=1, edgecolor='k')
        
        # Limit 0-1 for probability
        plt.xlim(0, 1)
        
        # Define ticks for numbering (major only) and grids (both)
        ax.set_xticks(major_ticks)
        ax.set_xticks(bins, minor=True)
        
        # Add axes grids to x and y
        ax.grid(which='both', axis='x', alpha = 0.5)
        ax.grid(which='major', axis='y', alpha = 0.5)
        
        # Set tick fontsize and yscale
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        #plt.yscale('log')
        
        # if there's additional text put it in a text box
        if addTextbool:
            ax.text(.55, .97, addText, transform=ax.transAxes, fontsize=10, horizontalalignment='right',
            verticalalignment='top', bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='round,pad=1'))
    
        # Save the figure
        if shaped:
            plt.savefig("Plots/"+saveas+add_term[i]+".png")
        else:
            plt.savefig("Plots/"+saveas+'Event_'+add_term[i]+".png")
    
        # Close plots if close is set true
        if close: 
            plt.close()
        else:
            plt.show()
        
    return count, step_count
    
      
'''LIMIT CALCULATION'''

def SigLimitCount(X, weight, Nb, N_arr, close, **kwargs):
    
    N_ttZ = N_arr[0]
    N_ttWm = N_arr[1]
    N_ttWp = N_arr[2]
    N_ggA_460_360 = N_arr[3]
    N_ggA_500_360 = N_arr[4]
    N_ggA_600_360 = N_arr[5]
    N_ggA_600_400 = N_arr[6]
    N_ggA_600_500 = N_arr[7]
    N_ggA_500_400 = N_arr[8]
    N_ggA_500_400_1 = N_arr[9]
    
    label = [r'other',r't$\bar{t}$Z','ggA ($m_A$=460, $m_H$=360)', 'ggA ($m_A$=500, $m_H$=360)', 'ggA ($m_A$=600, $m_H$=360)'
         , 'ggA ($m_A$=600, $m_H$=400)', 'ggA ($m_A$=600, $m_H$=500)', 'ggA ($m_A$=500, $m_H$=400)','ggA ($m_A$=500, $m_H$=400) 1']
    
    color = ['lightgreen','cyan','cornflowerblue','red','indigo','gold','black','maroon','darkorange']
    
    # Test whether a given bkg/sig is present
    bkg_test = np.zeros(2) # other, ttZ
    sig_test = np.zeros(7) # 460_360, 500_360, 600_360, 600_400, 600_500, 500_400, 500_400_1
    
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
    X_500_400_1 = None
    weight_460_360 = None
    weight_500_360 = None
    weight_600_360 = None
    weight_600_400 = None
    weight_600_500 = None
    weight_500_400 = None
    weight_500_400_1 = None
    
    shaped = False
    addText = False
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
        elif i =="shaped":
            shaped = j
        elif i =="line":
            linestyle = j
        elif i =="scale":
            scale = j
        elif i =='xlim':
            bkg_minim = j[0]
            bkg_maxim = j[1]
        elif i =='addText':
            addText = True
            add_string = j
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
                elif j[k] == '500_400_1':
                    X_500_400_1 = X[N_ggA_500_400:N_ggA_500_400_1]
                    weight_500_400_1 = weight[N_ggA_500_400:N_ggA_500_400_1]
                    sig_test[6] = 1
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
            
    # Form tuples
    bkg = ([X_other, X_ttZ])
    sig = ([X_460_360, X_500_360, X_600_360, X_600_400, X_600_500, X_500_400, X_500_400_1])
    weight = ([weight_other, weight_ttZ, weight_460_360, weight_500_360,weight_600_360,
               weight_600_400, weight_600_500, weight_500_400, weight_500_400_1])
    
    # Lengths of data
    bkg_length = len(bkg)
    sig_length = len(sig_test)
    
    # If max and min aren't defined, find them
    if 'bkg_maxim' not in locals():
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
    
    # Take every other value of the bin for the major tick labels
    major_ticks = bins[::2]

    # Bin centres
    bins_plot = np.delete(bins,Nb-1)
    bins_centre = bins_plot + (width/2)
    
    # Define arrays for shaping:
    bkg_count_sum = np.zeros(bkg_length)
    sig_count_sum = np.zeros(sig_length)
    bkg_weight_sum = np.zeros(bkg_length)
    
    # Define stackable counts and errors for multiple bkg
    stack_count = 0  
    final_error = 0
    
    # Empty array for signal return and empty ttZ for non-shaped plots
    sigcount = np.zeros((7,(Nb-1)))
    ttZcount = None

    ###################
    # PLOT BACKGROUND #
    ###################

    # Preliminary for shape comparison
    # Outside the main loop to avoid calculating sums multiple times
    if shaped:
        for i in range(bkg_length):
            if bkg_test[i] == 1:
                # Calculation of the counts
                count,edge = np.histogram(bkg[i], bins=bins, weights=weight[i])
                # Total weights for each histogram
                bkg_weight_sum[i] = np.sum(weight[i])
                # Total count for each histogram
                bkg_count_sum[i] = np.sum(count)
                
        # Total count for all bkg histograms
        all_bkg_count = np.sum(bkg_count_sum)
        # Total weight for all bkg histograms
        bkg_weight_tot = np.sum(bkg_weight_sum)
    
    # Main bkg loop
    for i in range(bkg_length):
        if bkg_test[i] == 1:
            # Calculation of the counts
            count,edge = np.histogram(bkg[i], bins=bins, weights=weight[i])

            # Save the ttZ count for return later
            if not shaped and i==1:
                ttZcount = count
                
            # Calculate the error
            error = np.sqrt((np.histogram(bkg[i], bins=bins, 
                                              weights=weight[i]*weight[i])[0]).astype(float))
            
            # Addition of all background errors in quadrature
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
            
    ###############
    # PLOT SIGNAL #
    ###############
            
    for i in range(sig_length):
        if sig_test[i] == 1:

            # Normalise to the background if shaped
            if shaped:
                step_count,edge = np.histogram(sig[i], bins=bins)
                sig_count = np.sum(step_count)
                step_count = step_count*(all_bkg_count/sig_count)
            # Otherwise plot an event comparison
            else:
                step_count,edge = np.histogram(sig[i], bins=bins, weights=weight[bkg_length+i])
         
            # Extend the step in x to reach the edge of the plot
            step_count_ext = np.append(step_count, step_count[len(step_count)-1])
            
            # Make the plot
            plt.step(bins, step_count_ext, label=label[bkg_length+i], color=color[bkg_length+i], 
                     where='post', linewidth = 2.0, linestyle=linestyle[i])
            
            # Save the signal count for specific signal for return
            #!!! Needs to be changed to extract a given signal
            if not shaped:
                sigcount[i,:] = step_count
                
                # Multiple sig counts need to returned in the future
                #sigcount[i] = step_count
    
    
    # Plot customisation
    if ytitle_in == 'Events':
        # If a GeV plot use GeV bins and decimal format
        ytitle = ('Events' + " / {width:} GeV").format(width = int(width))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.d'))
        plt.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1, .92), framealpha=1, edgecolor='k')
    else:
        # If a rad plot use rad bins and float format
        ytitle = ('Events' + " / {width:.3f} rad").format(width = width)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1, .92), framealpha=1, edgecolor='k')
    
    if shaped:
        plt.title(title + ' - shape comparison', fontsize=40)
    else:
        plt.title(title + ' - event comparison', fontsize=40)
    
    plt.xlabel(xtitle, fontsize=25)
    plt.ylabel(ytitle, fontsize=25)
    
    # Define ticks for numbering (major only) and grids (both)
    ax.set_xticks(major_ticks)
    ax.set_xticks(bins, minor=True)
    
    # Add axes grids to x and y
    ax.grid(which='both', axis='x', alpha = 0.5)
    ax.grid(which='major', axis='y', alpha = 0.5)
        
    # Rotate the x labels 45 degrees
    # fig.autofmt_xdate(rotation=45)
        
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
    # if shaped:
    #     #string = '\n'.join(('ATLAS','$\u221As = 13 TeV, 36.1$ $fb^{-1}$','$m_H$ $=130 GeV,$ $n_b$ $= 2$'))
    #     string = ('\u221As = 13 TeV, 139 $fb^{{-1}}$ \n Signal normalized to the background yield')
    # else:
    string = ('$\u221As = 13 TeV, 139$ $fb^{-1}$')
    
    # Add a text box
    # fontweight='bold', 
    ax.text(.97, .97, string, transform=ax.transAxes, fontsize=10, horizontalalignment='right',
            verticalalignment='top', bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='round,pad=1'))
    
    # Add an additional text box if in kwargs
    if addText:
        # ax.text(.97, .50, add_string, transform=ax.transAxes, fontsize=10, horizontalalignment='right',
        #     verticalalignment='top', bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='round,pad=1'))
        
        # fig.text(.5, .05, add_string, ha='center')
        # fig.set_size_inches(11, 10, forward=True)
        
        plt.figtext(0.5, 0.01, add_string, wrap=True, horizontalalignment='center', fontsize=8)
    
    # Save the figure to the relevant folder 
    if shaped:
        plt.savefig("Plots/Shapes/"+saveas+".png")
    else:
        plt.savefig("Plots/Events/"+saveas+".png")
        
    # Close if close is set true
    if close: 
        plt.close()
    else:
        plt.show()    
        
    # Return the counts, if required
    return ttZcount, sigcount  
    return count, step_count

# Similar functionality to ProbHist, could combine in future
def ProbLimitCount(pred_train, pred_test, y_train, y_test, w_train, w_test, Nb, close, label, **kwargs):
    
    # Combine train and test
    pred = np.concatenate((pred_train,pred_test))
    y = np.concatenate((y_train,y_test))
    w = np.concatenate((w_train,w_test))
    
    ### Separate bkg and sig ###
    
    # Test bkg and sig
    pred_bkg = pred[y==0]
    pred_sig = pred[y==1]
    
    w_bkg = w[y==0]
    w_sig = w[y==1]
    
    ##############
    # PLOT SETUP #
    ##############

    global save_count
    
    # Calculate bins and widths
    bins = np.linspace(0, 1, Nb)
    width = bins[1]-bins[0]
    
    bins_plot = np.delete(bins,Nb-1)

    # Take every other value of the bin for the major ticks
    major_ticks = bins[::2]
    
    # Initialise addTextbool
    addTextbool = False
    
    ###################
    # DATA EXTRACTION #
    ###################
    
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
        elif i=="addText":
            addTextbool = True
            addText = j

    color = ['cyan','red']
    
    # If saveas doesn't exist, give them arbitrary values
    if 'saveas' not in locals():
        saveas = ("unnamed_plot_%d" %save_count)
        save_count = save_count+1
        print('Warning, some plots have no save title')

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(11.0,9.0))
    
    ###################
    # PLOT BACKGROUND #
    ###################

    # Scale the weight to 139 ifb (*0.5 since there is 139*2 ifb of data)
    w_bkg = w_bkg*0.5*(300/139)
    
    # Reshape
    w_bkg = w_bkg.reshape((len(w_bkg),1)) 
    pred_bkg = pred_bkg.reshape((len(pred_bkg),1)) 

    # Calculate counts
    count,edge = np.histogram(pred_bkg, bins=bins, weights=w_bkg)

    # Make plot
    plt.bar(bins_plot, count, label=label[0], width = width, align='edge', color=color[0])

    ###############
    # PLOT SIGNAL #
    ###############
    
    # Scale the signal to 139 igb
    w_sig = w_sig*(300/139)
    
    # Reshape the weights
    w_sig = w_sig.reshape((len(w_sig),1)) 
    pred_sig = pred_sig.reshape((len(pred_sig),1)) 
    
    # Calculate counts
    step_count,edge = np.histogram(pred_sig, bins=bins, weights=w_sig)

    # Make plot
    step_count_ext = np.append(step_count, step_count[len(step_count)-1])
    plt.step(bins, step_count_ext, label=label[1], color=color[1], where='post', 
             linewidth = 2.0, linestyle='dashed')
    
    ######################
    # PLOT CUSTOMISATION #
    ######################
    
    # Titles and legend
    plt.title(title+'test', fontsize=40)
    plt.xlabel(xtitle, fontsize=25)
    plt.ylabel(ytitle, fontsize=25)
    plt.legend(loc='upper right', fontsize=15, framealpha=1, edgecolor='k')
    
    # Limit 0-1 for probability
    plt.xlim(0, 1)
    
    # Define ticks for numbering (major only) and grids (both)
    ax.set_xticks(major_ticks)
    ax.set_xticks(bins, minor=True)
    
    # Add axes grids to x and y
    ax.grid(which='both', axis='x', alpha = 0.5)
    ax.grid(which='major', axis='y', alpha = 0.5)
    
    # Set tick fontsize and yscale
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.yscale('log')
    
    # if there's additional text put it in a text box
    if addTextbool:
        ax.text(.55, .97, addText, transform=ax.transAxes, fontsize=10, horizontalalignment='right',
        verticalalignment='top', bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='round,pad=1'))


    plt.savefig("Plots/"+saveas+'Limit'+".png")

    # Close plots if close is set true
    if close: 
        plt.close()
    else:
        plt.show()
        
    return count, step_count
    
    
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
        if nb >= 2:
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
    



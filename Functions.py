# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:12:22 2020

@author: James
"""
import numpy as np
import matplotlib.pyplot as plt

# Close all plots on program start
plt.close('all')
# Global variable for untitled figures
save_count = 1

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
    plt.savefig("Plots/"+title+".png")
    if close: plt.close()
    
# Stacked histograms with signal
'''
X - Histrogram array
X1 - Signal array
weight - weights for histograms with signal weight on the end
Nb - Number of bins for the histogram
close - Whether to close the plot or not
label - labels for histograms with signal label on the end

kwargs - xtitle, ytitle, title, colour/color, saveas

colour for histograms with signal colour on the end
'''
def SignalHist(X, X1, weight, Nb, close, label, **kwargs):
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(11.0,9.0))
    global save_count
    
    # Definition of variables
    length = len(X)
    count = np.zeros((length, Nb))
    width = np.zeros((length, Nb))
    minimum = np.zeros(length)
    maximum = np.zeros(length)

    # Extract additonal arguments from kwargs
    for i, j in kwargs.items():
        if i == "xtitle":
            xtitle = j
        elif i=="ytitle":
            ytitle = j
        elif i=="title":
            title = j
        elif i=="color" or i=="colour":
            color = j
        elif i=="saveas":
            saveas = j

    # If color and saveas don't exist, give them arbitrary values
    if 'color' not in locals():
        color = np.full(length+1, None)
    if 'saveas' not in locals():
        saveas = ("unnamed_plot_%d" %save_count)
        save_count = save_count+1
        print('Warning, some plots have no save title')

    # Calculate min, max and counts for arrays
    for i in range(length):
        minimum[i] = np.amin(X[i])
        maximum[i] = np.amax(X[i])
        count[i][:], edge = np.histogram(X[i], Nb)
    
    # Calculate total min, max, bins and widths
    themin = np.amin(minimum)
    themax = np.amax(maximum)
    bins = np.linspace(themin, themax, Nb)
    width = bins[1]-bins[0]

    bins_plot = np.delete(bins,Nb-1)
    count = np.zeros((length, Nb-1))
    stack_count=0
    
    # Plot and stack histograms
    for i in range(length):
        count[i][:],edge = np.histogram(X[i], bins=bins, weights=weight[i])
        plt.bar(bins_plot, count[i][:], label=label[i], width = width, align='edge', bottom = stack_count, color=color[i])
        plt.legend(loc='upper right')
        stack_count = stack_count + count[:][i]
        
    # Plot the signal
    step_count,edge = np.histogram(X1, bins=bins, weights=weight[length])
    step_count_ext = np.append(step_count, step_count[len(step_count)-1])
    plt.step(bins, step_count_ext, label=label[length], color=color[length], where='post', linewidth = 2.0, linestyle='dashed')
        
    # Plot customisation
    ytitle = (ytitle + " / {width:.2f} GeV").format(width = width)
    plt.title(title, fontsize=40)
    plt.xlabel(xtitle, fontsize=25)
    plt.ylabel(ytitle, fontsize=25)
    plt.legend(loc='right', fontsize=15, bbox_to_anchor=(1, .75), framealpha=1, edgecolor='k')
    plt.xlim(themin, themax)
    ax.set_xticks(bins, minor=True)
    ax.grid(which='minor', axis='x', alpha = 0.5)
    ax.grid(which='major', axis='y', alpha = 0.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.yscale('log')
    
    # String for adding to the text box
    string = ('$\u221As = 13 TeV, 139$ $fb^{-1}$')
    
    # Add a text box
    ax.text(.96, .95, string, transform=ax.transAxes, fontsize=20, fontweight='bold', horizontalalignment='right',
            verticalalignment='top', bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='round,pad=1'))
    
    print("plotted: ", title)
    # Save the figure, close if close is set true
    plt.savefig("Plots/"+saveas+".png")
    if close: 
        plt.close()
        print("closed: ",title)
    
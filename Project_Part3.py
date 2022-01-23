# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:51:29 2021

@author: Stefano
"""

import numpy as np
import matplotlib.pyplot as pl
from project_functions_norm import *

ps = {#"text.usetex": True,
        "font.size" : 16,
        "font.family" : "Times New Roman",
        "axes.labelsize": 15,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.figsize": [7.5, 6],
        "mathtext.default": "regular"
        }
pl.rcParams.update(ps)
del ps


"""
Download Data and set standard parameters
"""
# Download the two sets of data: the observed data and the simulated expected 
# number of events if there were no oscillations (nomix)
data, simul = extract("C:/Users/Stefano/OneDrive - Imperial College London/"+
          "year3 stuff/data.txt")

edges = np.linspace(0, 10, len(data), endpoint = False)
centres = edges + (edges[1] - edges[0])/2
L = 295

#The paramaters for ALL NLL functions
ps = [data, simul, centres, L]
#%%
"""
3.1 THE DATA
"""
pl.figure()
ax = pl.axes()

ax.bar(edges, data, width = edges[1]-edges[0], align = "edge",
       label = "Observed data, with osicllations", color = "Red", 
       alpha = 0.75)
ax.bar(1,0, color = "Blue", label = "Simulated, with no osicllations")

ax2 = ax.twinx()
ax2.bar(edges, simul, width = edges[1]-edges[0], align = "edge",
       label = "Simulated, with no osicllations", color = "Blue", 
       alpha = 0.5)

#ax.set_title("$\u03BD_\u03BC$ Events per Energy")
ax.set_xlabel("Energy [GeV]")
ax.set_ylabel(r"Real $\nu_\mu$ Occurrencies", color = "Red")
ax2.set_ylabel(r"No-oscillations Occurrencies", color = "Blue")

ax.tick_params(axis = "y", colors = "Red")
ax2.tick_params(axis = "y", colors = "Blue")

ax.yaxis.label.set_color('Red')
ax2.yaxis.label.set_color("Blue")

ax.legend()
ax.grid(lw = 0.4)
pl.tight_layout()
pl.show()

del ax, ax2
#%%
"""
3.2 FIT FUNCTION
"""

# Starting values
energy = np.linspace(0.025, 10, 1000) #GeV
L      = 295        #km
theta  = 1.0  #np.pi/4
dm2    = 2.8  #10^-3 eV^2

#the oscillated event rate expectation
l_mix = lambdaMix(centres, simul)


# Plot of probability it has not oscillated
pl.figure(tight_layout = True)
p = pl.plot(energy, pNonMix(energy, L, theta, dm2) )
#pl.title(r"Survival Probability $\nu_\mu\rightarrow\nu_\mu$ " +
         #"vs Energy")
props = dict(boxstyle='round', facecolor='white', alpha=1,
             edgecolor = p[0].get_color())
text = r"$\theta_{23}$ = $\pi/4$"+ "\n"+ \
r"$\Delta m^2_{23}$"+\
rf"$ = {dm2} $" + r" $10^{-3} eV^2$"
pl.annotate (text, (0.7, 0.1), 
         xycoords = "axes fraction", size = 20, bbox = props)
pl.xlabel("Energy [GeV]")
pl.ylabel(r"Survival Probability $\nu_\mu\rightarrow\nu_\mu$")
pl.grid(lw = 0.4)
pl.show()

del energy, theta, dm2, p, props, text

#%%
"""
3.3 LIKELIHOOD FUNCTION
"""

"Plotting the Negative Log Likelihood vs Theta and Deltam^2"
# README "CHECKPOINT": INSERT 1000 IN NP.LINSPACE TO GET EXACTLY SAME PLOT AS 
# IN REPORT, BUT IT WILL TAKE QUITE SOME TIME
ts = np.linspace(0, 2, 100)
dm2 = np.linspace(0, 50, 100)
NLL = nLL([ts, dm2], ps) #takes time

from mpl_toolkits import mplot3d
colors = ["nipy_spectral","gist_ncar", "jet"]

for i in ["gist_ncar"]:
    title = r"Negative Log Likelihood vs $\theta_{23}$" +\
        r" & $\Delta m_{23}^2$"
    
    title2 = "Zoom in Region of Interest"
    
    fig, axes = pl.subplots(2,1, figsize = (7, 9), 
                            gridspec_kw={'height_ratios': [2, 1]})
    ax0 = axes[0]
    ax1 = axes[1]
    
    cntr0 = ax0.contourf(ts, dm2, NLL, 300, cmap = i)
    #ax0.set_title(title)
    ax0.set_xlabel(r"$\theta_{23}$ $[\pi/4]$")
    ax0.set_ylabel(r"$\Delta m_{23}^2$ $[10^{-3} eV^2]$")
    ax0.annotate ("a)", (-0.15, 1.00), xycoords = "axes fraction")
    
    pl.subplot(2,1,2)
    cntr1 = ax1.contourf(ts, dm2, NLL, 300, cmap = i)
    #ax1.set_title(title2, fontsize = 15)
    ax1.set_xlabel(r"$\theta_{23}$ $[\pi/4]$")
    ax1.set_ylabel(r"$\Delta m_{23}^2$ $[10^{-3} eV^2]$")
    ax1.set_ylim(2, 5)
    ax1.set_xlim(0.4, 1)
    ax1.annotate ("b)", (-0.15, 1.00), xycoords = "axes fraction")
    pl.subplots_adjust(hspace = 0.2, top = 0.95, bottom = 0.1)# bottom = 0.1, left = 0.2,
    fig.colorbar(cntr0, ax = axes, label = "Negative Log Likelihood")
    fig.show()

del colors, i, ts, dm2, fig, axes, ax0, ax1, title, title2, cntr0, cntr1
#%%
"""
3.4 PARABOLIC  MINIMISER
"""
print("\n3.4 PARABOLIC MINIMISER\n")

"Apply Parabolic Minimisation to nLL to find theta_min"

#Get a starting estimate for the t0, t1, t2 around the minimum
ts = np.linspace(0, 1, 500)
dm2 = 2.8
NLL = nLL_th(ts, [dm2] + ps)
imin = np.argmin(NLL)
t0 = ts[imin - 1]; t1 = ts[imin]; 
try: t2 = ts[imin + 1]
except: t2 = ts[imin - 2]

# Find Minimum
theta_min, NLL_min, coeff = parabMinimiser(nLL_th, [t0, t1, t2], True, 
                                           params = [dm2] + ps)

# Print Nice Result
theta_min = round(theta_min, 4)
print(f"With \u0394m^2 = {dm2/1000} eV^2, the estimated theta_min is:\n\
               \u03B8_min = {theta_min}\n")
print(f"It leads to minimum in NLL = {NLL_min}")

#Clear memory of useless variables
del t0, t1, t2 

#%%
"""
3.5 ACCURACY OF FIT RESULT
"""
print("\n3.5 ACCURACY OF FIT\n")

"Scanning to find the 1 Standard Deviation Interval"
#Find it
for i in range(len(NLL)):
    if NLL[i] <= NLL_min + 0.5:
        theta_minus = ts[i-1]
        std_below   = theta_min - theta_minus
        break
for i in range(imin, len(NLL)):
    if NLL[i] >= NLL_min + 0.5:
        theta_plus  = ts[i]
        std_above   = theta_plus - theta_min
        break
#Print Nice Result

magn = np.floor(np.log10(std_above))
std_above = round(std_above, int(abs(magn)) + 1)
magn = np.floor(np.log10(std_below))
std_below = round(std_below, int(abs(magn)) + 1)
print(f"Scanning the NLL(\u03B8_23) array, the 1std uncertainty is\
 estimated being +{std_above} pi/4 and - {std_below} pi/4.\n")
 
 
 
"Estimate from the Curvature of the Parabolic Estimate"
# The curvature of the parabolic interpolation about the minimum is 2a.
# Then, comparing it to a Gaussian - like Likelihood (central limit theorem),
# the 1std interval should be 1/sqrt(2a)
sigma = 1 / np.sqrt(2 * coeff[0])
magn = np.floor(np.log10(sigma))
sigma = round(sigma, int(abs(magn)) + 1)
print(f"Considering the Curvature of the Parabolic Interpolation, comparing\
 it to a Gaussian-like Likelihood due to Central Limit Theorem, we get a 1std\
 interval of {sigma} pi/4.\n")


"Using Bisection Method to find Uncertainty"
#Find the uncertainty above
tl  = theta_min
tr = theta_min + 2*sigma
theta_plus = bisection(bisectand, [tl, tr], 
                       params = [nLL_th, NLL_min + 0.5, [dm2]+ps])
std_above  = theta_plus - theta_min

tl  = theta_min - 2*sigma
tr = theta_min
theta_minus = bisection(bisectand, [tl, tr], 
                        params = [nLL_th, NLL_min + 0.5, [dm2]+ps])
std_below  = theta_min - theta_minus

magn = np.floor(np.log10(std_above))
std_above = round(std_above, int(abs(magn)) + 1)
magn = np.floor(np.log10(std_below))
std_below = round(std_below, int(abs(magn)) + 1)
print(f"Using bisection method to solve NLL(theta)-NLL_min-0.5 = 0,\
 the 1std uncertainty is estimated being \
 +{std_above} pi/4 and - {std_below} pi/4.\n")


del std_above, std_below, theta_minus, theta_plus, coeff, sigma, magn
del ts, tl, tr, i, imin



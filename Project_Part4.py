# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:51:31 2021

@author: Stefano
"""

import numpy as np
import matplotlib.pyplot as pl
from project_functions_norm import *

ps = {#"text.usetex": True,
        "font.size" : 18,
        "font.family" : "Times New Roman",
        "axes.labelsize": 17,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.figsize": [7.5, 6],
        "mathtext.default": "regular"
        }
pl.rcParams.update(ps)
del ps


"""Download Data and set standard parameters"""
# Download the two sets of data: the observed data and the simulated expected 
# number of events if there were no oscillations (nomix)
data, simul = extract("C:/Users/Stefano/OneDrive - Imperial College London/"+
          "year3 stuff/data.txt")

edges = np.linspace(0, 10, len(data), endpoint = False)
centres = edges + (edges[1] - edges[0])/2
L = 295

#The paramaters for ALL NLL functions
ps = [data, simul, centres, L]

del edges
#%%
"""4.1 2D UNIVARIATE MINIMISATION"""

print("\n4.1 UNIVARIATE MINIMISATION\n")

thetavec = [0.9, 0.95, 0.99]
dm2vec   = [2.7, 2.9, 3.1]
uvec     = [thetavec, dm2vec]

uvec_min1 = univariate(nLL, uvec, epsilon = 1e-6, params = ps)
NLL_min = nLL(uvec_min1, ps)
print("Univariate with \u03B8 minimised first:\n", 
      f"\u03B8   = {uvec_min1[0]} pi/4, \n\u0394m^2 = {uvec_min1[1]} 10^-3 eV^2", 
      f"\nNLL= {NLL_min}")

uvec_min2 = univariate(nLL, uvec, [1,0], epsilon = 1e-6, params = ps)
NLL_min = nLL(uvec_min2, ps)
print("\nUnivariate with \u0394m^2 minimised first:\n", 
      f"\u03B8   = {uvec_min2[0]} pi/4\
      \n\u0394m^2 = {uvec_min2[1]} 10^-3 eV^2", 
      f"\nNLL= {NLL_min}")

del thetavec, dm2vec, uvec, uvec_min1, uvec_min2, NLL_min

#%%
"""
4.2 NEWTON MINIMISATION
"""
print("\n4.2 NEWTON MINIMISATION\n")
umin = [0,0]
NLL_min = [1000, 1000]

uvec = [0.96, 2.9]
umin[0], curv = newMin(nLL, uvec, 1e-6, params = ps, epsilon = 1e-10)
NLL_min[0] = nLL(umin[0], ps)
print(f"Minimisation of NLL with Newton Method:\n\
      - \u03B8    = {umin[0][0]} pi/4\n\
      - \u0394m^2 = {umin[0][1]} 10^-3 eV\n\
These lead to a minimum NLL = {NLL_min[0]}.\n")


print("\n4.2 NEWTON-GRADIENT MINIMISATION\n")
umin = [0,0]
NLL_min = [1000, 1000]

uvec = [0.96, 2.9]
umin[0], curv = newGradMin(nLL, uvec, 1e-6, alpha = 1e-3,
                           params = ps, epsilon = 1e-10)
NLL_min[0] = nLL(umin[0], ps)
print(f"Minimisation of NLL with Newton Method:\n\
      - \u03B8    = {umin[0][0]} pi/4\n\
      - \u0394m^2 = {umin[0][1]} 10^-3 eV\n\
These lead to a minimum NLL = {NLL_min[0]}.\n")


print("\n4.2 DFP MINIMISATION\n")

uvec = [0.96, 2.9]
umin[1] = quasiNewMin(nLL, uvec, 1e-6, method = "DFP", epsilon = 1e-10, 
                         Nstop = 2e3, params = ps)
NLL_min[1] = nLL(umin[1], ps)
print(f"Minimisation of NLL with Newton Method:\n\
      - \u03B8    = {umin[1][0]} pi/4\n\
      - \u0394m^2 = {umin[1][1]} 10^-3 eV\n\
These lead to a minimum NLL = {NLL_min[1]}.\n")


i = np.argmin(NLL_min)
umin = umin[i]
NLL_min = NLL_min[i]
 
del i, uvec
#%%
"""
4.2b ACCURACY OF FIT
"""

"Take first rough estimates from curvature"
sigma_th = 1/np.sqrt(curv[0])
sigma_dm = 1/np.sqrt(curv[1])


"Use Bisection Method to find Uncertainty"
theta_min = umin[0]
dm2_min   = umin[1]
NLL_min   = nLL(umin, ps)

#Find the uncertainty above
tl  = theta_min
tr  = theta_min + 4*sigma_th
dml  = dm2_min 
dmr  = dm2_min + 4*sigma_dm
u_plus = bisection(bisectand, [[tl, dml], [tr, dmr]], u = [theta_min, dm2_min],
                        params = [nLL, NLL_min + 0.5, ps])
std_above_th  = u_plus[0] - theta_min
std_above_dm  = u_plus[1] - dm2_min

#Find the uncertainty below
tl  = theta_min - 4*sigma_th
tr = theta_min
dml  = dm2_min - 4*sigma_dm
dmr = dm2_min 
u_minus = bisection(bisectand, [[tl, dml], [tr, dmr]], u =[theta_min, dm2_min],
                        params = [nLL, NLL_min + 0.5, ps])
std_below_th  = theta_min - u_minus[0]
std_below_dm  = dm2_min - u_minus[1]

# Make it nice
magna = np.floor(np.log10(std_above_th))
std_above_th = round(std_above_th, int(abs(magna)) + 1)
magna = np.floor(np.log10(std_below_th))
std_below_th = round(std_below_th, int(abs(magna)) + 1)
umin[0]      = round(umin[0], int(abs(magna)) + 1)
magnb = np.floor(np.log10(std_above_dm))
std_above_dm = round(std_above_dm, int(abs(magnb)) + 1)
magnb = np.floor(np.log10(std_below_dm))
std_below_dm = round(std_below_dm, int(abs(magnb)) + 1)
umin[1]      = round(umin[1], int(abs(magnb)) + 1)


print("Using bisection method to solve NLL(theta)-NLL_min-0.5 = 0,to estimate\
 the 1std uncertainty, the final result is:\n",
 f"- \u03B8    = {umin[0]}   + {std_above_th} , - {std_below_th}  pi/4\n",
 f"- \u0394m^2 = {umin[1]} + {std_above_dm}, - {std_below_dm} 10^-3 eV^2\n",
f"These lead to a minimum NLL = {NLL_min}.")
del theta_min, dm2_min, tl, tr, dml, dmr, u_plus, u_minus
del std_above_th, std_above_dm, std_below_dm, std_below_th
del magna, magnb, NLL_min
#%%
"Goodness of Fit"
l_mix = lambdaMix(centres, simul, L = 295, theta = umin[0], dm2 = umin[1])
Nparams = 2
dof = len(data) - Nparams

chi2_2D   = chi2(data, l_mix, Nparams = Nparams)
chi2_2D_w = chi2(data, l_mix, Nparams = Nparams, correction = "Williams")

p_2D   = pvalue(chi2_2D, dof, Nstop = 23)
magn = np.floor(np.log10(p_2D))
try: p_2D = round(p_2D, int(abs(magn)) + 2)
except: pass

p_2D_w = pvalue(chi2_2D_w, dof, Nstop = 23) 
magn = np.floor(np.log10(p_2D_w))
p_2D_w = round(p_2D_w, int(abs(magn)) + 2)

print("\nThe results of the Chi-Squared Calculations yield:")
print(f" Standard reduced \u03C7^2 = {chi2_2D/dof}  -> pvalue = {p_2D}")
print(f" Williams'reduced \u03C7^2 = {chi2_2D_w/dof}-> pvalue = {p_2D_w}" ) 

del Nparams, dof, chi2_2D, chi2_2D_w, p_2D, p_2D_w, magn

#%%
"Plotting Data vs Expectation"
l_mix = lambdaMix(centres, simul, theta = umin[0], dm2 = umin[1] )
pl.figure()
pl.bar(centres, data, width = centres[1]-centres[0],
       label = "Observed data", color = "Red", alpha = 0.75)
pl.bar(centres, l_mix, width = centres[1]-centres[0],
        label = "Expectation \u03BB", color = "Blue", alpha = 0.4)
#pl.title("$\u03BD_\u03BC$ Events per Energy")
pl.xlabel("Energy [GeV]")
pl.ylabel(r"Number of $\nu_\mu$ Events")
x = np.linspace(0, 10, 11)
pl.xticks(x)
pl.legend()
pl.grid(lw = 0.4)
pl.tight_layout()
pl.show()

del l_mix, x
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:51:33 2021

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
"""
5 ENERGY DEPENDENT LAMBDA
"""

"Plot NLL vs alpha, to get an idea"
alpha = np.linspace(0.01, 20, 1000)
th = 1.00
dm2 = 2.90
NLL_alpha = nLLnew([th, dm2, alpha], ps) 
pl.figure()
p = pl.plot(alpha, NLL_alpha)
props = dict(boxstyle='round', facecolor='white', alpha = 1,
                 edgecolor = p[0].get_color()) 
text = r"$\theta_{23}$"+ rf"$ = {th}$ $\pi/4$"+ "\n"+ \
    r"$\Delta m^2_{23}$"+\
    rf"$ = {dm2} $" + r" $10^{-3} eV^2$"
pl.annotate (text, (0.6, 0.1), 
             xycoords = "axes fraction", size = 20, bbox = props)

#pl.title("NLL vs \u03B1")
pl.xlabel(r"$\alpha$ $[GeV^{-1}]$")
pl.ylabel("NLL")
pl.grid(lw = 0.3)
pl.tight_layout()
pl.show()
del alpha, NLL_alpha, th, dm2, text, props, p

#%%
umin = [0,0,0,0]; NLL_min = [0,0,0,0]
print("\nMINIMISATON WITH ANNEALING METROPOLIS:")
# def metropolis (f, interval, iters = 5e3, kT0 = 100, anneal = 0.5, scan = 1e3, 
#                 step = 0.2, close_factor = 2, track = False, halving = 0.5,
#                 params = float("nan"))
interv = np.array([[0, 0, 0.01],  [1, 100, 5]])
kT0  = 10
iters = int(30e3)
anneal = 0.75
scan = 3e3
umin[0], NLL_min[0] = metropolis(nLLnew, interv, iters, kT0, anneal, scan,
                             params = ps )
print("     Metropolis' Range            Result")
print(f"\u03B8:       {interv[:,0]}         ->     {umin[0][0]} pi/4 ")
print(f"\u0394m^2:    {interv[:,1]}     ->     {umin[0][1]} 10^-3 eV^-2 ")
print(f"alpha:   {interv[:,2]}     ->     {umin[0][2]} GeV^-1 ")
print(f"These lead to a minimum NLL {NLL_min[0]}. These values are used in \
following algorithms as starting points.\n")


print("\nMINIMISATION WITH NEWTON METHOD:")
umin[1], curv = newMin(nLLnew, umin[0], [1e-6, 1e-6, 1e-6], epsilon = 1e-8,
                       params = ps)
NLL_min[1] = nLLnew(umin[1], ps)
if umin[1][0] > 1:
    umin[1][0] = 2 - umin[1][0]
print(f"\
      - \u03B8     = {umin[1][0]} pi/4\n\
      - \u0394m^2  = {umin[1][1]} 10^-3 eV^2\n\
      - alpha = {umin[1][2]} GeV^-1\n\
These lead to a minimum NLL = {NLL_min[1]}.\n")


print("\nMINIMISATION WITH NEWTON-GRADIENT METHOD:")
umin[2], curv = newGradMin(nLLnew, umin[0], [1e-6, 1e-6, 1e-6], alpha = 1e-3, 
                      params = ps, epsilon = 1e-8)
NLL_min[2] = nLLnew(umin[2], ps)
if umin[2][0] > 1:
    umin[2][0] = 2 - umin[2][0]
print(f"\
      - \u03B8     = {umin[2][0]} pi/4\n\
      - \u0394m^2  = {umin[2][1]} 10^-3 eV^2\n\
      - alpha = {umin[2][2]} GeV^-1\n\
These lead to a minimum NLL = {NLL_min[2]}.\n")


print("\nMINIMISATION WITH DFP METHOD:")
umin[3] = quasiNewMin(nLLnew, umin[0], [1e-6, 1e-6, 1e-6], method = "DFP", 
                      params = ps, epsilon = 1e-8)
NLL_min[3] = nLLnew(umin[3], ps)
if umin[3][0] > 1:
    umin[3][0] = 2 - umin[3][0]
print(f"\
      - \u03B8     = {umin[3][0]} pi/4\n\
      - \u0394m^2  = {umin[3][1]} 10^-3 eV^2\n\
      - alpha = {umin[3][2]} GeV^-1\n\
These lead to a minimum NLL = {NLL_min[2]}.\n")

del interv, kT0, iters, anneal, scan

i = np.argmin(NLL_min)
umin = umin[i]
NLL_min = min(NLL_min)

#%%
"""
ACCURACY OF FIT
"""
"Take first estimates from curvature"
sigma_th = 1/np.sqrt(curv[0])
sigma_dm = 1/np.sqrt(curv[1])
sigma_al = 1/np.sqrt(curv[2])


"Use Bisection Method to find Uncertainty"
theta_min = umin[0]
dm2_min   = umin[1]
al_min    = umin[2]
NLL_min   = nLLnew(umin, ps)

#Find the uncertainty above
tl  = theta_min
tr  = theta_min + 4*sigma_th
dml  = dm2_min 
dmr  = dm2_min + 4*sigma_dm
al = al_min
ar = al_min +4*sigma_al 
u_plus = bisection(bisectand, [[tl, dml, al], [tr, dmr, ar]], 
                   u = [theta_min, dm2_min, al_min], 
                   params = [nLLnew, NLL_min + 0.5, ps])
std_above_th  = u_plus[0] - theta_min
std_above_dm  = u_plus[1] - dm2_min
std_above_al  = u_plus[2] - al_min

#Find the uncertainty below
tl  = theta_min - 4*sigma_th
tr = theta_min
dml  = dm2_min - 4*sigma_dm
dmr = dm2_min 
al = al_min - 4*sigma_al
ar = al_min
u_minus = bisection(bisectand, [[tl, dml, al], [tr, dmr, ar]], 
                   u = [theta_min, dm2_min, al_min], 
                   params = [nLLnew, NLL_min + 0.5, ps])
std_below_th  = theta_min - u_minus[0]
std_below_dm  = dm2_min - u_minus[1]
std_below_al  = al_min - u_minus[2]

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

magnc = np.floor(np.log10(std_above_al))
std_above_al = round(std_above_al, int(abs(magnc)) + 1)
magnc = np.floor(np.log10(std_below_al))
std_below_al = round(std_below_al, int(abs(magnc)) + 1)
umin[2]      = round(umin[2], int(abs(magnc)) + 1)


print("Using bisection method to solve NLL(theta)-NLL_min-0.5 = 0,to estimate\
 the 1std uncertainty, the final result is:\n",
 f"- \u03B8    = {umin[0]} + {std_above_th}, - {std_below_th} pi/4\n",
 f"- \u0394m^2 = {umin[1]} + {std_above_dm}, - {std_below_dm} 10^-3 eV\n",
 f"- \u03B1    = {umin[2]} + {std_above_al}, - {std_below_al} GeV^-1\n",
f"These lead to a minimum NLL = {NLL_min}.")

del curv, sigma_al, sigma_dm, sigma_th
del theta_min, dm2_min, tl, tr, dml, dmr, u_plus, u_minus
del std_above_th, std_above_dm, std_below_dm, std_below_th, std_above_al
del std_below_al, magna, magnb, magnc, NLL_min
#%%
print("\n\nCALCULATION OF CHI-SQUARED AND P-VALUE")

l_new   = lambdaNew (centres, simul, umin[0], umin[1], umin[2])
Nparams = 3
dof     = len(data) - Nparams 
chi2_3D   = chi2(data, l_new, Nparams = Nparams)
chi2_3D_w = chi2(data, l_new, Nparams = Nparams, correction = "Williams")

p_3D   = pvalue(chi2_3D, dof, Nstop = 22)
magn = np.floor(np.log10(p_3D))
p_3D = round(p_3D, int(abs(magn)) + 2)

p_3D_w = pvalue(chi2_3D_w, dof, Nstop = 22) 
magn = np.floor(np.log10(p_3D_w))
p_3D_w = round(p_3D_w, int(abs(magn)) + 2)

print("\nThe results of the Chi-Squared Calculations yield:")
print(f" Standard reduced \u03C7^2 = {chi2_3D/dof}-> pvalue = {p_3D}")
print(f" Williams'reduced \u03C7^2 = {chi2_3D_w/dof}-> pvalue = {p_3D_w}" )

del Nparams, dof, chi2_3D, chi2_3D_w, p_3D, p_3D_w, magn

#%%
"Plotting Data vs Expectation"
pl.figure()
pl.bar(centres, data, width = centres[1]-centres[0],
       label = "Observed data", color = "Red", alpha = 0.75)
pl.bar(centres, l_new, width = centres[1]-centres[0],
        label = r"Expectation $\lambda$", color = "Blue", alpha = 0.4)
#pl.title("$\u03BD_\u03BC$ Events per Energy")
pl.xlabel("Energy [GeV]")
pl.ylabel(r"Number of $\mu_\nu$ Events")
pl.legend()
pl.grid(lw = 0.4)
pl.tight_layout()
pl.show()

del l_new
#%%
print("\nThe 1std deviation below for the sin^2 term in Probability is:")
print(np.sin(2 * (umin[0]-0.06) * np.pi/4))
del umin



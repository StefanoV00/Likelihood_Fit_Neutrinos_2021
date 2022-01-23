# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 22:55:58 2021

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


"""
FUNCTIONS USED IN LATER SECTIONS OF THIS MODULE
"""
def parab(x, coeff = [1, -2, 193]):
    a = coeff[0]
    b = coeff[1]
    c = coeff[2]
    return a*x*x + b*x + c 

def test1 (u):
     x = u[0]
     y = u[1]
     return 2*x**3 + 6*x*y**2 - 3*y**3 - 150*x 
f1 = "f = 2x^3 + 6xy^2 - 3y^3 - 150x"
# http://personal.maths.surrey.ac.uk/st/S.Zelik/teach/calculus/max_min_2var.pdf

def test2 (u):
    x = u[0]
    y = u[1]
    return np.cos(193*x) * 4 * y ** 2 - np.exp(-2*x) * np.sin(0.005*y**2)
f2 = "f = cos(193x) + 4y^2 - exp(-2x) * sin(0.005y^2)"

def sinc3 (u):
    x = u[0]
    y = u[1]
    z = u[2]
    return - np.sinc(x) * np.sinc(y-np.pi) * np.sinc(z+np.pi)
f3 = "f = - sinc(x) * sinc(y-\u03c0) * sinc(z+\u03c0)"


#%%
print("\nTESTING THE DERIVATIVES on f1 and f2:\n")

a1 = der1 (test1, [3,11], 1e-4, 0)
a2 = der1 (test1, [3,11], 1e-4, 1)
a3 = der2 (test1, [3,11], 1e-4, 0)
a4 = der2 (test1, [3,11], 1e-4, 1)
a5 = der2 (test1, [3,11], 1e-4, [0,1])
a6 = der2 (test1, [3,11], 1e-4, [1,0])
print(f1)
print("      Expected  vs       Obtained:\n",
      f"d_x  =  630    vs    {a1}\n",
      f"d_y  = -693    vs    {a2}\n",
      f"d_xx =   36    vs    {a3}\n",
      f"d_yy = -162    vs    {a4}\n",
      f"d_xy =  132    vs    {a5}\n",
      f"d_yx =  132    vs    {a6}\n")


a1 = der1 (test2, [0.20, 1.47], 1e-4, 0)
a2 = der1 (test2, [0.20, 1.47], 1e-4, 1)
a3 = der2 (test2, [13,   0.19], 1e-4, 0)
a4 = der2 (test2, [3,      11], 1e-4, 1)
a5 = der2 (test2, [13,   0.19], 1e-4, [0,1])
a6 = der2 (test2, [0.017,3.56], 1e-4, [1,0])
print("\n"+f2)
print("       Expected    vs       Obtained:\n",
      f"d_x  =-1307.66    vs    {a1}\n",
      f"d_y  = 7.2921     vs    {a2}\n",
      f"d_xx = 2282.58    vs    {a3}\n",
      f"d_yy = 4.67328    vs    {a4}\n",
      f"d_xy = -265.634   vs    {a5}\n",
      f"d_yx = 763.861    vs    {a6}\n")

del a1, a2, a3, a4, a5, a6


#%%
print("\nTESTING THE PARABOLIC MINIMISER (1D):")

mx_parab, m = parabMinimiser(parab,  [-88, -31, 1945])
mx_cos1,  m = parabMinimiser(np.cos, [3.2, 3.1, 2.9] )
mx_cos2,  m, coeff = parabMinimiser(np.cos, [9.2, 9.1, 9.6] , True)
print(f" For x^2-2x+193 we expect 1.0.    The obtained result is {mx_parab}")
print(" For 1st np.cos test we expect \u03c0. The obtained result is "+
      f"{round(mx_cos1, 6)}")
print(" For 2nd np.cos we expect 3 * \u03c0.  The obtained result is "+
      f"3*{round(mx_cos2/3, 6)}")

#Clear memory of useless variables
del mx_parab, mx_cos1, mx_cos2, m, coeff
#%%
print("\n\n\nTESTING THE UNIVARIATE MINIMISER on f1 and f3:\n")


print(f1 +f"\nLocal minimum [5, 0], saddles [-3, -4],[3, 4], maximum [-5, 0]")
start = [[2.0, 2.5, 2.25], [0.5, 1.0, 1.5]]
umin, track_uni_1 = univariate(test1, start, track = 1)
print(f"My minimum from {start} is {umin}")
start = [[3.0, 2.5, 2.0], [1.5, 2.0, 1.8]]
umin, track_uni_2 = univariate(test1, start, track = 1)
print(f"My minimum from {start} is {umin}")
start = [[-4.0, -2.0, 1.0], [0.5, 1.0, 2.0]]
umin, track_uni_3 = univariate(test1, start, track = 1)
print(f"My minimum from {start} is {umin}")
start = [[-4.0, -2.0, 1.0], [0.5, -3.0, 2.0]]
umin, track_uni_4 = univariate(test1, start, track = 1)
print(f"My minimum from {start} is {umin}\n")


print(f3 + f"\nReal Global Solution at [0, \u03c0, -\u03c0], many local")
start = [[-2, -1, 1], [2, 3, 4], [-2, -2.5, -3.5]]
umin = univariate(sinc3, start, 1e-8, Nstop = 2e4)
print(f"My minimum from {start} is {umin}")
start = [[-3, -2, 1], [1, 3, 4], [-2, -2.5, -4.0]]
umin = univariate(sinc3, start, 1e-8, Nstop = 2e4)
print(f"My minimum from {start} is {umin}\n")

print("NOTE: it is clear from theory and last result, the convergence of the",
      "univariate method  to the ",
      "correct solution heavily relies on function's shape & initial guesses.")
 
#Clear memory of useless variables
del umin, start




print("\n\n\nTESTING THE NEWTON MINIMISER on f1 and f3:\n")


print(f1 + "\nLocal minimum [5, 0], saddles [-3, -4],[3, 4], maximum [-5, 0]",
      " and keeps going down for negative x.")
start = [2.5, 1.0]
umin, curv, track_new_1 = newMin(test1, start, 1e-6, track = 1)
print(f"My minimum from {start}= {umin}")
start = [2.5, 2.0]
umin, curv, track_new_2 = newMin(test1, start, 1e-6, track = 1)
print(f"My minimum from {start}= {umin}")
start = [-2.0, 1.0]
umin, curv, track_new_3 = newMin(test1, start, 1e-6, track = 1)
print(f"My minimum from {start}= {umin}")
start = [-2.0, -3.0]
umin, curv, track_new_4 = newMin(test1, start, 1e-6, track = 1)
print(f"My minimum from {start}= {umin}\n")


print(f3 + f"\nReal Global Solution at [0, \u03c0, -\u03c0], many local")
start = [ 0.3, 3, -3]     
umin, curv = newMin(sinc3, start, 1e-6)
print(f"My minimum from {start} = {umin}")
start = [ 0.6, 2.5, -2]
umin, curv = newMin(sinc3, start, 1e-6)
print(f"My minimum from {start} = {umin}\n")

print("NOTE: Newtonian Minimiser approaches closer critical point,\
 independently of its nature.")
#Clear memory of useless variables
del umin, curv, start





print("\n\n\nTESTING THE NEWTON_GRADIENT MINIMISER on f1 and f3:\n")


print(f1 + "\nLocal minimum [5, 0], saddles [-3, -4],[3, 4], maximum [-5, 0]",
      " and keeps going down for negative x.")
start = [2.5, 1.0]
umin, curv, track_newg_1 = newGradMin(test1, start, 1e-6, 
                                      alpha = 1e-2, track = 1)
print(f"My minimum from {start}= {umin}")
start = [2.5, 2.0]
umin, curv, track_newg_2 = newGradMin(test1, start, 1e-6, 
                                     alpha = 1e-2, track = 1)
print(f"My minimum from {start}= {umin}")
start = [-2.0, 1.0]
umin, curv, track_newg_3 = newGradMin(test1, start, 1e-6,
                                      alpha = 1e-2, track = 1)
print(f"My minimum from {start}= {umin}")
start = [-2.0, -3.0]
umin, curv, track_newg_4 = newGradMin(test1, start, 1e-6,
                                      alpha = 1e-2, track = 1)
print(f"My minimum from {start}= {umin}\n")


print(f3 + f"\nReal Global Solution at [0, \u03c0, -\u03c0], many local")
start = [ 0.3, 3, -3]     
umin, curv = newGradMin(sinc3, start, 1e-6, epsilon = 1e-10, alpha = 1e-2)
print(f"My minimum from {start} = {umin}")
start = [ 0.6, 2.5, -2]
umin, curv = newGradMin(sinc3, start, 1e-6, epsilon = 1e-10, alpha = 1e-2)
print(f"My minimum from {start} = {umin}\n")


print("NOTE: Newton-Gradient Minimiser only moves to the minimum, either\
 approaching one, or infinitely going down (in case of poor choice of\
 initial guesses), which is exactly what we expect from a minimiser, with\
 reduced risk of getting stucked at a saddle, no risk of getting stucked at \
 maxima. ")
#Clear memory of useless variables
del umin, curv, start




print("\n\n\nTESTING THE QUASI-NEWTON MINIMISER on f1 and f3:\n")


print(f1 + f"\nLocal minimum [5, 0], saddles [-3, -4],[3, 4], maximum [-5, 0]")
start = [2.5, 1]
umin, track_dfp_1 = quasiNewMin(test1, start, 1e-6, alphamax = 2, Nstop = 2e4,
                   method = "DFP", track = 1)
print(f"My minimum from {start} with DFP = {umin}")

start = [2.5, 1]
umin, track_bfgs_1 = quasiNewMin(test1, start, 1e-6, alphamax = 2,
                    track = 1)
print(f"My minimum from {start} with BFGS = {umin}")


start = [2.5, 2]
umin, track_dfp_2 = quasiNewMin(test1, start, 1e-6, alphamax = 2,
                   method = "DFP", track = 1)
print(f"My minimum from {start} with DFP  = {umin}")

start = [2.5, 2]
umin, track_bfgs_2 = quasiNewMin(test1, start, 1e-6, alphamax = 2,
                    track = 1)
print(f"My minimum from {start} with BFGS  = {umin}")

start = [-2, 1]
umin, track_dfp_3 = quasiNewMin(test1, start, 1e-6, alphamax = 2,
                   method = "DFP", track = 1)
print(f"My minimum from {start} with DFP = {umin}")
start = [-2, 1]
umin, track_bfgs_3 = quasiNewMin(test1, start, 1e-6, alphamax = 2,
                   method = "BFGS", track = 1, Nstop = 2e4)
print(f"My minimum from {start} with BFGS = {umin}")

start = [-2, -3]
umin, track_dfp_4 = quasiNewMin(test1, start, 1e-6, alphamax = 2, method = "DFP",
                    track = 1)
print(f"My minimum from {start} with DFP = {umin}")
start = [-2, -3]
umin, track_bfgs_4 = quasiNewMin(test1, start, 1e-6, alphamax = 2, method = "BFGS",
                    track = 1, Nstop = 2e4)
print(f"My minimum from {start} with BFGS = {umin}")


print("\n", f3 + f"\nReal Global Solution at [0, \u03c0, -\u03c0], many local")
start = [ 0.3, 3, -3]     
umin = quasiNewMin(sinc3, start, 1e-6, alphamax = 2, Nstop = 3e3,method ="DFP")
print(f"My minimum from {start} with DFP = {umin}")
start = [ 0.3, 3, -3]     
umin = quasiNewMin(sinc3, start, 1e-6, alphamax = 2, Nstop = 3e3)
print(f"My minimum from {start} with BFGS = {umin}")
start = [ 0.6, 2.5, -2]
umin = quasiNewMin(test2, start, 1e-6, alphamax = 2, Nstop = 3e3, method="DFP")
print(f"My minimum from {start} with DFP = {umin}\n")
start = [ 0.6, 2.5, -2]
umin = quasiNewMin(test2, start, 1e-6, alphamax = 2, Nstop = 3e3)
print(f"My minimum from {start} with BFGS = {umin}\n")


print("NOTE1: Quasi-Newton apparently only moves to minima, with much much\
 lower probability of converging to a saddle!\n")
print("NOTE2: Quasi-Newton obviously is not perfect and it\
 might get lost in a bumpy function, iterating, never converging, until\
 Nstop is reached, which may happen especially for big alpha.\n")
print("NOTE3: DFP appears superior when far from the local minima we are\
 searching for. BFGS instead overshoots enormously as it finds an unbounded \
 descent, then the curvature factor corrects its trajectory and \
 very slowly makes it approach a critical point, but if this isn't the minimum\
 it will overhsoot again and repeat. n")
#Clear memory of useless variables
del umin, start


#%%
"Plotting path of methods in test1"

colors = ["gist_ncar", "seismic"]
colors = [colors[0]]

xa = -10
xb =  10
ya = -5
yb =  7  
x = np.linspace(xa, xb, 1000)
y = np.linspace(ya, yb, 1000)
X, Y = np.meshgrid(x,y)
Z = test1([X,Y])

for i in colors:
    
    fig, axes = pl.subplots(2,1, figsize = (7, 9), sharex = True)
    ax0 = axes[0]
    ax1 = axes[1]
    cntr0 = ax0.contourf(X,Y,Z, 300, cmap = i)
    
    ax0.annotate ("a)", (-0.15, 0.95), xycoords = "axes fraction")
    ax0.plot(track_uni_1[:,0], track_uni_1[:,1], marker = ".", c = "black", 
            label = "Univariate")
    ax0.plot(track_uni_2[:,0], track_uni_2[:,1], marker = ".", c = "black")
    ax0.plot(track_uni_3[:,0], track_uni_3[:,1], marker = ".", c = "black")
    
    ax0.plot(track_new_1[:,0], track_new_1[:,1], marker = ".", c = "b", 
            label = "Newton's", zorder = 2)
    ax0.plot(track_new_2[:,0], track_new_2[:,1], marker = ".", c = "b", 
            zorder = 2)
    ax0.plot(track_new_3[:,0], track_new_3[:,1], marker = ".", c = "b", 
            zorder = 2)
    
    ax0.plot(track_newg_1[:,0], track_newg_1[:,1], marker = ".", c = "g", 
            label = "Newton-Gradient", zorder = 3)
    ax0.plot(track_newg_2[:,0], track_newg_2[:,1], marker = ".", c = "g")
    ax0.plot(track_newg_3[:,0], track_newg_3[:,1], marker = ".", c = "g")

    ax0.plot( 5, 0, marker = "$X$", ls = "None", ms = 10, c = "w", 
            label = "Critical Points", zorder = 3)
    ax0.plot(-5, 0, marker = "$M$", ls = "None", ms = 10, c = "w", zorder = 3)
    ax0.plot([3, -3], [4, -4], ls = "None", marker = "$S$", ms = 10, c = "w", 
            zorder = 3)

    ax0.plot(2.5, 1, marker = "$A$", ls = "None", ms = 10,
            label = "Starting Points", c = "black", zorder = 3)
    ax0.plot(2.5, 2, marker = "$B$", ms = 10, c = "black", zorder = 3)
    ax0.plot(-2, 1, marker = "$C$", ms = 10, c = "black", zorder = 3)
    
    ax0.set_xlim(xa, xb); ax0.set_ylim(ya, yb)
    ax0.set_ylabel("y")
    ax0.legend(fontsize = 11)

    
    cntr1 = ax1.contourf(X,Y,Z, 300, cmap = i)
    ax1.annotate ("b)", (-0.15, 0.95), xycoords = "axes fraction")
    ax1.plot(track_dfp_1[:,0], track_dfp_1[:,1], marker = ".", c = "g", 
            label = "DFP")
    ax1.plot(track_dfp_2[:,0], track_dfp_2[:,1], marker = ".", c = "g")
    ax1.plot(track_dfp_3[:,0], track_dfp_3[:,1], marker = ".", c = "g")
    
    ax1.plot(track_bfgs_1[:,0], track_bfgs_1[:,1], marker = ".", c = "b", 
            label = "BFGS")
    ax1.plot(track_bfgs_2[:,0], track_bfgs_2[:,1], marker = ".", c = "b")
    ax1.plot(track_bfgs_3[:,0], track_bfgs_3[:,1], marker = ".", c = "b")
    

    ax1.plot( 5, 0, marker = "$X$", ls = "None", ms = 10, c = "white", 
            label = "Critical Points")
    ax1.plot(-5, 0, marker = "$M$", ls = "None", ms = 10, c = "white")
    ax1.plot([3, -3], [4, -4], ls = "None", marker = "$S$", ms = 10, c = "white")
    
    ax1.plot(2.5, 1, marker = "$A$", ls = "None", ms = 10,
            label = "Starting Points", c = "black", zorder = 3)
    ax1.plot(2.5, 2, marker = "$B$", ms = 10, c = "black", zorder = 3)
    ax1.plot(-2, 1, marker = "$C$", ms = 10, c = "black", zorder = 3)
    
    ax1.set_xlim(xa, xb); ax1.set_ylim(ya, yb)
    ax1.set_xlabel("x");  ax1.set_ylabel("y")
    ax1.set_xticks([-7.5, -5, -2.5, 0, 2.5, 5, 7.5])
    ax1.legend(fontsize = 11)
    
    pl.subplots_adjust(hspace = 0.1, top = 0.98, bottom = 0.1)
    fig.colorbar(cntr0, ax = axes, label = r"$f_1(x,y)$")
    fig.show() 

del fig, ax0, ax1, cntr0
del X, Y, Z, x, y
del xa, xb, ya, yb
del track_uni_1,  track_uni_2,  track_uni_3,  track_uni_4
del track_new_1,  track_new_2,  track_new_3,  track_new_4
del track_newg_1, track_newg_2, track_newg_3, track_newg_4
del track_dfp_1,  track_dfp_2,  track_dfp_3,  track_dfp_4
del track_bfgs_1, track_bfgs_2, track_bfgs_3, track_bfgs_4 
#%%
print("\nTESTING THE METROPOLIS MINIMISER on f2 and f3:\n")
# def metropolis (f, interval, iters = 1e3, kT0 = 100, anneal = 0.5, scan = 50, 
#                 step = 0.2, close_factor = 2, params = float("nan"))

print("\n" + f3)
print("Real Global Solution is at [0, \u03c1, -\u03c1], but the functions has\
      some other local minima in the interval.")
interv = [[-15, -15, -15], [15, 15, 15]]
print(f"Note: the function goes from -1 to 1 in the chosen {interv} interval.")

#README CHECKPOINT: THIS DOES TAKE A WHILE, THE NUMBER OF ITERATIONS USED FOR
# REPORT OBSERVATIONS WAS iters = 10,000, N = 100
iters = 10000
scan = [200, 500, 3000, 10000]
N = 10

step = [0.05, 0.20, 0.30]
kTs  = [0.10, 1.00, 5.00]
anns = [0.80, 0.50, 0.20]
close= [5.00, 2.00, 1.00]


print(f"\nAverage sum of distance from result per {iters}",
      "iterations with aggressive choice of parameters, hence: small step, \
small temperature, big annealing, big close_factor.")
for sc in scan:
    distance = 0
    for i in range(N):
        umin, minimum = metropolis(sinc3, interv, iters, kTs[0], anns[0], 
                            scan = sc, step = step[0], close_factor = close[0])
        distance += np.sqrt(sum((umin - np.array([0.0, np.pi, -np.pi]))**2))
    distance /= N
    print(f"{sc} scans -> distance average : {distance}")


print(f"\nAverage sum of distance from result per {iters}",
      "iterations with mild choice of parameters, later halving.")
for sc in scan:
    distance = 0
    for i in range(N):
        umin, minimum = metropolis(sinc3, interv, iters, kTs[1], anns[1], 
                            scan = sc, step = step[1], close_factor = close[1],
                            halving = 0.75)
        distance += sum((umin - np.array([0.0, np.pi, -np.pi]))**2)
    distance /= N
    print(f"{sc} scans -> distance average : {distance}")


print(f"\nAverage sum of distance from result per {iters}",
      "iterations with safe choice of parameters, hence: large step, \
large temperature, small annealing, small close_factor, no halving.")
for sc in scan:
    distance = 0
    for i in range(N):
        umin, minimum = metropolis(sinc3, interv, iters, kTs[2], anns[2], 
                                   scan = sc, step = step[2], 
                                   close_factor = close[2], halving = 0)
        distance += sum((umin - np.array([0.0, np.pi, -np.pi]))**2)
    distance /= N
    print(f"{sc} scans -> distance average : {distance}")

del iters, N, anns, kTs, umin, minimum, distance, interv

#%%
print("\nTESTING CHI-SQUARED CALCULATOR:\n")

a1 = chi2 ([149, 151, 246, 78, 91], [136, 120, 280, 79, 115])
a2 = chi2 ([149, 151, 246, 78, 91], [115, 138, 310, 60, 70]) 
a3 = chi2 ([149, 151, 246, 78, 91], [81, 81, 390, 81, 81]) 
print("      Expected     vs    Obtained:\n",
      f"      18.4009     vs   {a1}\n",
      f"      36.1897     vs   {a2}\n",
      f"     172.0951     vs   {a3}\n")


print("\nTESTING P-VALUE CALCULATOR:\n")
#http://courses.atlas.illinois.edu/spring2016/STAT/STAT200/pchisq.html
a1 = pvalue (0.455, 1, Nstop = 21) 
a2 = pvalue (2.770, 2)
a3 = pvalue (6.250, 3)
a4 = pvalue (18.25, 15)
a5 = pvalue (56.33, 50)
a6 = pvalue (43.188, 60)
a7 = pvalue (102, 100)
print("      Expected     vs      Obtained:\n",
      f"       0.5000     vs   {a1}\n",
      f"       0.2503     vs   {a2}\n",
      f"       0.1001     vs   {a3}\n",
      f"       0.2498     vs   {a4}\n",
      f"       0.2501     vs   {a5}\n",
      f"       0.9500     vs   {a6}\n",
      f"       0.4256     vs   {a7}\n")

del a1, a2, a3, a4, a5, a6

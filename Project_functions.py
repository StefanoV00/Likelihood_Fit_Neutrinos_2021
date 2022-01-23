# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 17:30:10 2021

@author: Stefano
"""
import numpy as np
import math

def extract (path):
    filedata = []
    with open("C:/Users/Stefano/OneDrive - Imperial College London/"+
              "year3 stuff/data.txt", "r") as fl: 
            f = fl.readlines()          #f is the list of lines(strings)
            for ln in f:
                try: 
                    filedata.append(float(ln))
                except: 
                    pass

    # The following works in general for any length of data, not just 200
    data = [int(f) for f in filedata[:len(filedata)//2]]
    data2 = filedata[len(filedata)//2:]
    return data, data2


def parab(x, coeff = [1, -2, 193]):
    a = coeff[0]
    b = coeff[1]
    c = coeff[2]
    return a*x*x + b*x + c 



def pNonMix (E, L, theta, dm2):
    """
    The probability muon neutrinos with energy E are still muon after length L 
    as their mixing angle is theta and difference in squared masses is dm2.
    Note:\n
    -energy in GeV \n
    -length in km  \n
    -theta in fraction of np.pi/4  \n
    -dm2 in 10^-3 eV^2   \n
    """
    theta *= np.pi/4
    dm2 /= 1000
    A = np.sin(2*theta) * np.sin(2*theta)
    B = 1.267 * dm2 * L / E
    B = np.sin(B) * np.sin(B)
    return 1 - A * B


def lambdaMix (E, simul_no_mix, L = 295, theta = 1, dm2 = 2.4 ):
    """
    Return the with-oscillations expected bin occupancy as a function of 
    energy values, simulated-no-oscillations expected values simul_no_mix, the 
    travelled length L, the mixing angle theta, the mass^2 difference dm2. 
    Note: \n
    -energy in GeV \n
    -length in km  \n
    -theta in rad  \n
    -dm2 in eV^2 
    """
    l_mix = simul_no_mix * pNonMix(E, L, theta, dm2)
    return l_mix



def nLL(uvec, params):
    """
    The negative log likelihood NLL of the Poisson distribued values of data 
    array with expected values l_mix, depending on mixing angle theta and 
    difference in m^2. Note:\n
    -energy in GeV \n
    -length in km  \n
    -theta in rad  \n
    -dm2 in eV^2   
    """
    theta = uvec[0]
    dm2   = uvec[1]
    
    data  = params[0]
    simul = params[1]
    E     = params[2]
    L     = params[3]
    
    if not hasattr(theta, "__len__"):
        if not hasattr(dm2, "__len__"):
            l_mix = lambdaMix (E, simul, L,  theta, dm2)
            NLL = 0
            for i in range(len(data)):
                di = data[i]
                li = l_mix[i]
                if di:
                    NLL += li - di * np.log(li) 
                    NLL += np.log(float(math.factorial(di)))
                else:
                    NLL += li
            return NLL
        else:
            NLL = []
            for d in dm2:
                l_mix = lambdaMix (E, simul, L,  theta, d)
                NLLi = 0
                for i in range(len(data)):
                    di = data[i]
                    li = l_mix[i]
                    if di:
                        NLLi += li - di * np.log(li) 
                        NLLi += np.log(float(math.factorial(di)))
                    else:
                        NLLi += li
                NLL.append(NLLi)
            NLL = np.array(NLL)
            return NLL
    else:
        if not hasattr(dm2, "__len__"): 
            NLL = []
            for t in theta:
                l_mix = lambdaMix (E, simul, L,  t, dm2)
                NLLi = 0
                for i in range(len(data)):
                    di = data[i]
                    li = l_mix[i]
                    if di:
                        NLLi += li - di * np.log(li) 
                        NLLi += np.log(float(math.factorial(di)))
                    else:
                        NLLi += li
                NLL.append(NLLi)
            NLL = np.array(NLL)
            return NLL
        else:
            NLL = []
            for d in dm2:
                NLLi = []
                for t in theta:
                    l_mix = lambdaMix (E, simul, L,  t, d)
                    NLLii = 0
                    for i in range(len(data)):
                        di = data[i]
                        li = l_mix[i]
                        if di:
                            NLLii += li - di * np.log(li) 
                            NLLii += np.log(float(math.factorial(di)))
                        else:
                            NLLii += li
                    NLLi.append(NLLii)
                NLL.append(NLLi)
            NLL = np.array(NLL)
            return NLL



def nLLBig(uvec, params):
    """
    The negative log likelihood NLL of the Poisson distribued values of data 
    array with expected values l_mix, depending on mixing angle theta and 
    difference in m^2. Note:\n
    -energy in GeV \n
    -length in km  \n
    -theta in rad  \n
    -dm2 in eV^2   
    """
    theta = uvec[0]
    dm2   = uvec[1]
    
    data  = params[0]
    simul = params[1]
    E     = params[2]
    L     = params[3]
    
    if not hasattr(theta, "__len__"):
        if not hasattr(dm2, "__len__"):
            l_mix = lambdaMix (E, simul, L,  theta, dm2)
            NLL = 0
            for i in range(len(data)):
                di = data[i]
                li = l_mix[i]
                if di:
                    NLL += li - di + di * np.log(di/li)
                else:
                    NLL += li
            return NLL
        else:
            NLL = []
            for d in dm2:
                l_mix = lambdaMix (E, simul, L,  theta, d)
                NLLi = 0
                for i in range(len(data)):
                    di = data[i]
                    li = l_mix[i]
                    if di:
                        NLLi += li - di + di * np.log(di/li)
                    else:
                        NLLi += li
                NLL.append(NLLi)
            NLL = np.array(NLL)
            return NLL
    else:
        if not hasattr(dm2, "__len__"): 
            NLL = []
            for t in theta:
                l_mix = lambdaMix (E, simul, L,  t, dm2)
                NLLi = 0
                for i in range(len(data)):
                    di = data[i]
                    li = l_mix[i]
                    if di:
                        NLLi += li - di + di * np.log(di/li)
                    else:
                        NLLi += li
                NLL.append(NLLi)
            NLL = np.array(NLL)
            return NLL
        else:
            NLL = []
            for d in dm2:
                NLLi = []
                for t in theta:
                    l_mix = lambdaMix (E, simul, L,  t, d)
                    NLLii = 0
                    for i in range(len(data)):
                        di = data[i]
                        li = l_mix[i]
                        if di:
                            NLLii += li - di + di * np.log(di/li)
                        else:
                            NLLii += li
                    NLLi.append(NLLii)
                NLL.append(NLLi)
            NLL = np.array(NLL)
            return NLL
        

def nLL_th (theta, params):
    """
    NLL as monodimensional function of theta, with dm2 being the first
    of the parameters.
    """
    
    return nLL([theta, params[0]], params[1:])


    

def lambdaNew (E, simul_no_mix, theta, dm2, alpha, L = 295):
    """
    Return the new lambda, assuming linearly energy-dependent cross section.
    """
    l_new = simul_no_mix * pNonMix(E, L, theta, dm2) * alpha * E
    return l_new



def nLLnew(uvec, params):
    """
    The New Negative Log Likelihood NLL, assuming a cross section linearly 
    dependent on energy. Note:\n
    -energy in GeV \n
    -length in km  \n
    -theta in rad  \n
    -dm2 in eV^2   
    """
    data    = params[0]
    simul   = params[1]
    E       = params[2]
    L       = params[3]
    
    theta = uvec[0]
    dm2   = uvec[1]
    alpha = uvec[2]
    
    if not hasattr(theta, "__len__"):
        if not hasattr(dm2, "__len__"):
            
            if not hasattr(alpha, "__len__"): #none is an array
                l_mix = lambdaNew (E, simul, theta, dm2, alpha, L)
                NLL = 0
                for i in range(len(data)):
                    di = data[i]
                    li = l_mix[i]
                    if di:
                        NLL += li - di * np.log(li) 
                        NLL += np.log(float(math.factorial(di)))
                    else:
                        NLL += li
                return NLL
            else: # only alpha is an array
                NLL = []
                for a in alpha:
                    l_mix = lambdaNew (E, simul, theta, dm2, a, L)
                    NLLi = 0
                    for i in range(len(data)):
                        di = data[i]
                        li = l_mix[i]
                        if di:
                            NLLi += li - di * np.log(li) 
                            NLLi += np.log(float(math.factorial(di)))
                        else:
                            NLLi += li
                    NLL.append(NLLi)
                NLL = np.array(NLL)
                return NLL
            
        else: #dm2 is an array
            if not hasattr(alpha, "__len__"):
                NLL = []
                for d in dm2:
                    l_mix = lambdaNew (E, simul, theta, d, alpha, L)
                    NLLi = 0
                    for i in range(len(data)):
                        di = data[i]
                        li = l_mix[i]
                        if di:
                            NLLi += li - di * np.log(li) 
                            NLLi += np.log(float(math.factorial(di)))
                        else:
                            NLLi += li
                    NLL.append(NLLi)
                NLL = np.array(NLL)
                return NLL
            else: #dm2 and alpha are arrays
                NLL = []
                for d in dm2:
                    NLLi = []
                    for a in alpha:
                        l_mix = lambdaNew (E, simul, theta, d, a, L)
                        NLLii = 0
                        for i in range(len(data)):
                            di = data[i]
                            li = l_mix[i]
                            if di:
                                NLLii += li - di * np.log(li) 
                                NLLii += np.log(float(math.factorial(di)))
                            else:
                                NLLii += li
                        NLLi.append(NLLii)
                    NLL.append(NLLi)
                NLL = np.array(NLL)
                return NLL
            
    else: #theta is an array
        if not hasattr(dm2, "__len__"): 
            
            if not hasattr(alpha, "__len__"): #only theta is an array
                NLL = []
                for t in theta:
                    l_mix = lambdaNew (E, simul, t, dm2, alpha, L)
                    NLLi = 0
                    for i in range(len(data)):
                        di = data[i]
                        li = l_mix[i]
                        if di:
                            NLLi += li - di * np.log(li) 
                            NLLi += np.log(float(math.factorial(di)))
                        else:
                            NLLi += li
                    NLL.append(NLLi)
                NLL = np.array(NLL)
                return NLL
            else: #theta and alpha are arrays
                NLL = []
                for t in theta:
                    NLLi = []
                    for a in alpha:
                        l_mix = lambdaNew (E, simul, t, dm2, a, L)
                        NLLii = 0
                        for i in range(len(data)):
                            di = data[i]
                            li = l_mix[i]
                            if di:
                                NLLii += li - di * np.log(li) 
                                NLLii += np.log(float(math.factorial(di)))
                            else:
                                NLLii += li
                        NLLi.append(NLLii)
                    NLL.append(NLLi)
                NLL = np.array(NLL)
                return NLL
            
        else: #theta and dm2 ara arrays
            if not hasattr(alpha, "__len__"):
                NLL = []
                for t in theta:
                    NLLi = []
                    for d in dm2:
                        l_mix = lambdaNew (E, simul, t, d, alpha, L)
                        NLLii = 0
                        for i in range(len(data)):
                            di = data[i]
                            li = l_mix[i]
                            if di:
                                NLLii += li - di * np.log(li) 
                                NLLii += np.log(float(math.factorial(di)))
                            else:
                                NLLii += li
                        NLLi.append(NLLii)
                    NLL.append(NLLi)
                NLL = np.array(NLL)
                return NLL
            else: #all are arrays
                NLL = []
                for t in theta:
                    NLLi = []
                    for d in dm2:
                        NLLii = 0
                        for a in alpha:
                            l_mix = lambdaNew (E, simul, t, d, a, L)
                            NLLiii = 0
                            for i in range(len(data)):
                                di = data[i]
                                li = l_mix[i]
                                if di:
                                    NLLiii += li - di * np.log(li) 
                                    NLLiii += np.log(float(math.factorial(di)))
                                else:
                                    NLLiii += li
                                NLLii.append(NLLiii)
                            NLLi.append(NLLii)
                        NLL.append(NLLi)
                NLL = np.array(NLL)
                return NLL



def nLLnewBig(uvec, params):
    """
    The New Negative Log Likelihood NLL, assuming a cross section linearly 
    dependent on energy. Note:\n
    -energy in GeV \n
    -length in km  \n
    -theta in rad  \n
    -dm2 in eV^2   
    """
    data    = params[0]
    simul   = params[1]
    E       = params[2]
    L       = params[3]
    
    theta = uvec[0]
    dm2   = uvec[1]
    alpha = uvec[2]
    
    if not hasattr(theta, "__len__"):
        if not hasattr(dm2, "__len__"):
            
            if not hasattr(alpha, "__len__"): #none is an array
                l_mix = lambdaNew (E, simul, theta, dm2, alpha, L)
                NLL = 0
                for i in range(len(data)):
                    di = data[i]
                    li = l_mix[i]
                    if di:
                        NLL += li - di + di * np.log(di/li)
                    else:
                        NLL += li
                return NLL
            else: # only alpha is an array
                NLL = []
                for a in alpha:
                    l_mix = lambdaNew (E, simul, theta, dm2, a, L)
                    NLLi = 0
                    for i in range(len(data)):
                        di = data[i]
                        li = l_mix[i]
                        if di:
                            NLLi += li - di + di * np.log(di/li)
                        else:
                            NLLi += li
                    NLL.append(NLLi)
                NLL = np.array(NLL)
                return NLL
            
        else: #dm2 is an array
            if not hasattr(alpha, "__len__"):
                NLL = []
                for d in dm2:
                    l_mix = lambdaNew (E, simul, theta, d, alpha, L)
                    NLLi = 0
                    for i in range(len(data)):
                        di = data[i]
                        li = l_mix[i]
                        if di:
                            NLLi += li - di + di * np.log(di/li)
                        else:
                            NLLi += li
                    NLL.append(NLLi)
                NLL = np.array(NLL)
                return NLL
            else: #dm2 and alpha are arrays
                NLL = []
                for d in dm2:
                    NLLi = []
                    for a in alpha:
                        l_mix = lambdaNew (E, simul, theta, d, a, L)
                        NLLii = 0
                        for i in range(len(data)):
                            di = data[i]
                            li = l_mix[i]
                            if di:
                                NLLii += li - di + di * np.log(di/li)
                            else:
                                NLLii += li
                        NLLi.append(NLLii)
                    NLL.append(NLLi)
                NLL = np.array(NLL)
                return NLL
            
    else: #theta is an array
        if not hasattr(dm2, "__len__"): 
            
            if not hasattr(alpha, "__len__"): #only theta is an array
                NLL = []
                for t in theta:
                    l_mix = lambdaNew (E, simul, t, dm2, alpha, L)
                    NLLi = 0
                    for i in range(len(data)):
                        di = data[i]
                        li = l_mix[i]
                        if di:
                            NLLi += li - di + di * np.log(di/li)
                        else:
                            NLLi += li
                    NLL.append(NLLi)
                NLL = np.array(NLL)
                return NLL
            else: #theta and alpha are arrays
                NLL = []
                for t in theta:
                    NLLi = []
                    for a in alpha:
                        l_mix = lambdaNew (E, simul, t, dm2, a, L)
                        NLLii = 0
                        for i in range(len(data)):
                            di = data[i]
                            li = l_mix[i]
                            if di:
                                NLLii += li - di + di * np.log(di/li)
                            else:
                                NLLii += li
                        NLLi.append(NLLii)
                    NLL.append(NLLi)
                NLL = np.array(NLL)
                return NLL
            
        else: #theta and dm2 ara arrays
            if not hasattr(alpha, "__len__"):
                NLL = []
                for t in theta:
                    NLLi = []
                    for d in dm2:
                        l_mix = lambdaNew (E, simul, t, d, alpha, L)
                        NLLii = 0
                        for i in range(len(data)):
                            di = data[i]
                            li = l_mix[i]
                            if di:
                                NLLii += li - di + di * np.log(di/li)
                            else:
                                NLLii += li
                        NLLi.append(NLLii)
                    NLL.append(NLLi)
                NLL = np.array(NLL)
                return NLL
            else: #all are arrays
                NLL = []
                for t in theta:
                    NLLi = []
                    for d in dm2:
                        NLLii = 0
                        for a in alpha:
                            l_mix = lambdaNew (E, simul, t, d, a, L)
                            NLLiii = 0
                            for i in range(len(data)):
                                di = data[i]
                                li = l_mix[i]
                                if di:
                                    NLLiii += li - di + di * np.log(di/li)
                                else:
                                    NLLiii += li
                                NLLii.append(NLLiii)
                            NLLi.append(NLLii)
                        NLL.append(NLLi)
                NLL = np.array(NLL)
                return NLL
        








"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
CALCULUS & EQUATIONS        CALCULUS & EQUATIONS       CALCULUS & EQUATIONS
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
def der1 (f, u0, h = 1e-6, which = 0, params = float("nan")):
    """1st derivative of a function f at u0 over its which^th variable, with
    finite difference h and 4th order accuracy.\n
    
    If the function takes 1 extra numerical parameter, assigned by default, 
    you need to reassign it. Sorry :("""  
    
    if hasattr(params, "__len__"): #given input
        boo = True
    elif not np.isnan(params): #number different than "nan"
        boo = True
    else:
        boo = False
        
    if boo:
        if not hasattr(u0, "__len__"): #1D
            A = +1.0 *  f(u0 - 2 * h, params)
            B = -8.0 *  f(u0 - h, params)
            C = +8.0 *  f(u0 + h, params)
            D = -1.0 *  f(u0 + 2 * h, params)
        else: #multidimensional
            u = np.array(u0, float)
            u[which] = u0[which] - 2.0 * h; A = +1.0 *  f(u, params)
            u[which] = u0[which] - 1.0 * h; B = -8.0 *  f(u, params)
            u[which] = u0[which] + 1.0 * h; C = +8.0 *  f(u, params)
            u[which] = u0[which] + 2.0 * h; D = -1.0 *  f(u, params)
    else:
        if not hasattr(u0, "__len__"): #1D
            A = +1.0 *  f(u0 - 2 * h)
            B = -8.0 *  f(u0 - h)
            C = +8.0 *  f(u0 + h)
            D = -1.0 *  f(u0 + 2 * h)
        else: #multidimensional
            u = np.array(u0, float)
            u[which] = u0[which] - 2.0 * h; A = +1.0 *  f(u)
            u[which] = u0[which] - 1.0 * h; B = -8.0 *  f(u)
            u[which] = u0[which] + 1.0 * h; C = +8.0 *  f(u)
            u[which] = u0[which] + 2.0 * h; D = -1.0 *  f(u)   
        
    return (A + B + C + D) / 12.0 / h



def der2 (f, u0, h = 1e-6, which = 0, params = float("nan")):
    """
    Compute 2nd derivative for a N-dimensional function f(u), with u then
    being a N-dimensional vector, at point u0.\n
    
    It derives over the variables specified by the argument which:\n
        - if it is an integer m with 0 <= m < N, derives twice over the mth
        variable.\n
        - if it is a 2D-list with entries 0 <= m,n < N, derives over the mth
        and nth variables.\n
     
    Uses finite difference h = [hx, hy]. If h is a scalar, then uses [h, h], 
    with 4th order accuracy derivatives.\n
    
    If the function takes 1 extra numerical parameter, assigned by default, 
    you need to reassign it. Sorry :(.
    """

    u = np.array(u0, float)
    D = 0
    
    if not hasattr(h, "__len__"):
        h = [h for i in u0]
        
    
    if hasattr(params, "__len__"): # it is a given input
        boo = True
    elif not np.isnan(params): #number different than "nan"
        boo = True
    else:
        boo = False
       
    if not hasattr(which,"__len__"):  #double derivative over "which"th
        h = h[which]
        if boo: 
            u[which] = u0[which] - 2.0 * h; A = - 1.0 *f(u, params)
            u[which] = u0[which] - 1.0 * h; B = 16.0 * f(u, params)
            C = - 30.0 * f(u0, params)
            u[which] = u0[which] + 1.0 * h; D = 16.0 * f(u, params)
            u[which] = u0[which] + 2.0 * h; E = - 1.0 *f(u, params)
        else:
            u[which] = u0[which] - 2.0 * h; A = -1.0 * f(u)
            u[which] = u0[which] - 1.0 * h; B = 16.0 * f(u)
            C = - 30.0 * f(u0) 
            u[which] = u0[which] + 1.0 * h; D = 16.0 * f(u)
            u[which] = u0[which] + 2.0 * h; E = -1.0 * f(u)   
        return (A + B + C + D + E) / 12.0 / h**2
    
    elif which[0] == which[1]:   #double derivative over "which[0]"th
        which = which[0]
        h = h[which]
        if boo: 
            u[which] = u0[which] - 2.0 * h; A = - 1.0 *f(u, params)
            u[which] = u0[which] - 1.0 * h; B = 16.0 * f(u, params)
            C = - 30.0 * f(u0, params)
            u[which] = u0[which] + 1.0 * h; D = 16.0 * f(u, params)
            u[which] = u0[which] + 2.0 * h; E = - 1.0 *f(u, params)
        else:
            u[which] = u0[which] - 2.0 * h; A = -1.0 * f(u)
            u[which] = u0[which] - 1.0 * h; B = 16.0 * f(u)
            C = - 30.0 * f(u0) 
            u[which] = u0[which] + 1.0 * h; D = 16.0 * f(u)
            u[which] = u0[which] + 2.0 * h; E = -1.0 * f(u)  
        return (A + B + C + D + E) / 12.0 / h**2
    
    else: # combined first derivatives over variables
        hcoffs = [-2.0, -1.0, 1.0, 2.0]
        j = [1/12, -2/3, 2/3, -1/12]
        if boo:
            for i in range(len(hcoffs)):
                u[which[0]] = u0[which[0]] + hcoffs[i] * h[0]
                D += j[i] * der1(f, u, h[1], which[1], params)
        else:
            for i in range(len(hcoffs)):
                u[which[0]] = u0[which[0]] + hcoffs[i] * h[0]
                D += j[i] * der1(f, u, h[1], which[1])  
        D/=h[0]
        return D



def trapsimps (f, r, epsilon = 1e-6, Nstop = 20):
    """
    Takes 1D function f and limits of integration r as arguments, using a more
    efficient mix of trapezoid and simpson's methods.\n
    
    If the second extremum of the interval is inf, then will assign r[-1]=R,
    such that f(R) < epsilon.
    """
    
    j = 0
    e = abs(epsilon) + 1 # just an initial value
    
    if np.isinf(r[-1]):
        fR = 1
        R  = 10
        while fR > epsilon:
            fR = f(R)
            R *= 10
        r[-1] = R
        
    while e > epsilon:
        j += 1        #counter
        
        if j == 1:
            hj = r[-1] - r[0]
            Nj = 2
            Tj = hj * (f(r[0]) + f(r[-1])) / 2
            
            hjj = hj / 2
            Tjj = Tj / 2 #other terms to be added 
            Njj = Nj * 2 - 1
            rjj = np.linspace(r[0], r[-1], Njj)
            Tjj += sum(hjj * f(x) for x in rjj[1:][::2])
            Ij = 4 * Tjj / 3 - Tj / 3
            continue
        
        Ipre = Ij * 1 
        hj   = hjj * 1
        Tj   = Tjj * 1
        Nj   = Njj * 1
        
        hjj = hj / 2
        Tjj = Tj / 2 #other terms to be added 
        Njj = Nj * 2 - 1
        rjj = np.linspace(r[0], r[-1], Njj)
        Tjj += sum(hjj * f(x) for x in rjj[1:][::2])
        Ij = 4 * Tjj / 3 - Tj / 3
        
        if Ij != 0:
            e = abs((Ij - Ipre) / Ij )

        if j == Nstop:
            break
    return Ij



def bisectand (uvec, params = float("nan")):
    f = params[0]
    A = params[1]
    ps = params[2]
    try:    B = f(uvec, ps)
    except: B = f(uvec)
    return B - A


def bisection (f, uveclr, u = 0, epsilon = 1e-8, Nstop = 1e3, 
               params = float("nan")):
    """
    Apply bisection method to solve f(u) = 0.\n
    
    IMPORTANT: If function takes one extra numerical parameter, you have to 
    insert it even if the default value is wanted, otherwise returns NaN, 
    sorry :(.\n
    
    Parameters:\n
    ------------- \n   
    - uveclr : [[u0left, u1left, ...], [u0right, u1right, ...]] coordinates of 
    initial points, supposed to be at left and right of solution.\n
    
    - u: [u0,u1,...] if f is multidimensional, coordinates solutions for 
    coordinates will be found individually, with others being set to values 
    in u.\n
    
    - converge when change in coordinates is smaller than epsilon\n
    
    - Nstop: maximum number of iterations PER DIMENSION.\n

    Returns\n
    -------\n
    A number if 1D, an array if multidimensional, with solutions per dimension.
    """

    if hasattr(u, "__len__"):
        uleft  = np.array(uveclr[0], float)
        uright = np.array(uveclr[1], float)
        usol = []
        for i in range(len(u)):
            uvec = np.array(u, float)
            xl = uleft[i]; uvec[i] = xl
            try:   fl = f(uvec, params)
            except:fl = f(uvec)
            xr = uright[i]; uvec[i] = xr
            try:   fr = f(uvec, params)
            except:fr = f(uvec)
            
            counter = 0
            e = 1
            while e > epsilon and counter < Nstop:
                xnew = (xl + xr) / 2; uvec[i] = xnew
                # print(xl, xr, xnew)
                # print(uvec)
                try:   fnew = f(uvec, params)
                except:fnew = f(uvec)
                if fnew * fl >= 0:
                    e = abs(xl - xnew)
                    xl = xnew
                else:
                    e = abs(xr - xnew)
                    xr = xnew
                counter += 1

            usol.append(xnew)
            
        return np.array(usol)
    
    else:
        uleft  = uveclr[0]
        uright = uveclr[1]
        xl = uleft
        try:   fl = f(xl, params)
        except:fl = f(xl)
        xr = uright
        try:   fr = f(xr, params)
        except:fr = f(xr)
        
        counter = 0
        e = 1
        while e > epsilon and counter < Nstop:
            xnew = (xl + xr) / 2
            try:   fnew = f(xnew, params)
            except:fnew = f(xnew)
            if fnew * fl >= 0:
                e = abs(xl-xnew)
                xl = xnew
            else:
                e = abs(xr-xnew)
                xr = xnew
            counter += 1
        
        usol = xnew
        
        return usol
            
        
        
    
                

"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MINIMISERS          MINIMISERS          MINIMISERS          MINIMISERS
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

def parabMinimiser (f, xvec, want_coeffs = False, epsilon = 1e-10, 
                    params = float("nan")):
    """
    Apply parabolic minimiser to fnction f, starting from three points around
    the minimum with coordinates xvec = [x0, x1, x2]. \n
    
    IMPORTANT: If function takes one extra numerical parameter, you have to 
    insert it even if the default value is wanted, otherwise returns NaN, 
    sorry :(.\n
    
    Returns tuple xmin, ymin. \n
    
    
    If want_coeffs == True, then also return the three coefficients [a,b,c] of 
    the last parabolic interpolation.
    """
    try: yvec = [f(x, params) for x in xvec] 
    except: yvec = [f(x) for x in xvec]  
    e = 1
    while e > epsilon:
        x0 = xvec[0]; x1 = xvec[1]; x2 = xvec[2]
        y0 = yvec[0]; y1 = yvec[1]; y2 = yvec[2]
        N = (x2*x2 - x1*x1) * y0 + (x0*x0 - x2*x2) * y1 + (x1*x1 - x0*x0) * y2
        D = (x2 - x1) * y0 + (x0 - x2) * y1 + (x1 - x0) * y2
        x3 = N/D/2;
        e = abs(x3 - xvec[-1])
        xvec.append(x3)
        try: y3 = f(x3, params)
        except: y3 = f(x3)
        yvec.append(y3)
        i = np.argmax(yvec)
        xvec.pop(i); yvec.pop(i)
        if i == 3:
            break
        
    if want_coeffs:
        x0 = xvec[0]; x1 = xvec[1]; x2 = xvec[2]
        y0 = yvec[0]; y1 = yvec[1]; y2 = yvec[2]
        
        d0 = (x0 - x1) * (x0 - x2)
        d1 = (x1 - x0) * (x1 - x2)
        d2 = (x2 - x1) * (x2 - x0)
        
        a0 = y0 / d0
        a1 = y1 / d1
        a2 = y2 / d2
        a = a0 + a1 + a2
        
        b01 = y0 * x1 / d0
        b02 = y0 * x2 / d0
        b10 = y1 * x0 / d1
        b12 = y1 * x2 / d1
        b20 = y2 * x0 / d2
        b21 = y2 * x1 / d2
        b = - (b01 + b02 + b10 + b12 + b20 + b21)
        
        c0 = y0 * x1 * x2 / d0
        c1 = x0 * y1 * x2 / d1
        c2 = x0 * x1 * y2 / d2
        c = c0 + c1 + c2
        
        i = np.argmin(yvec)
        return xvec[i], yvec[i], [a,b,c]   
    else:
        i = np.argmin(yvec)
        return xvec[i], yvec[i]



def univariate (f, uvec, order = 1, epsilon = 1e-9, Nstop = 1e3, track = 0,
                params = float("nan")):
    """
    Application of the univariate method for the minimisation of ND function
    f (x0, x1, ..., x(N-1)), given three initial coordinates per dimension in
    uvec = [[x00, x01, x02], [x10, x11, x12], ...]. \n
    
    IMPORTANT: If function takes one extra numerical parameter, you have to 
    insert it even if the default value is wanted, otherwise returns NaN, 
    sorry :(.\n
    
    The optional order parameter must be a list representing the order we
    want the parameters to be minimised as a permutation (FROM 0) of the args 
    of f. If not provided, it will be the obvious [0,1,2, ..., N-1]. 
    Otherwise, it might be, for example, [3, 0, 2, 1], for a 4D function.\n
    
    Solution convrges when the biggest change in any coordinate is smaller
    than epsilon.\n
    
    Returns tuple [x1min, x2min, ...]. If track, also return list of points
    it went through.
    """
    
    uvec00 = (np.array(uvec, float)).tolist() #starting point
    uvec0 = [] #best estimates
    ulist = [] #list of best estimates
    err = []
    fvec = []
    for uvec_i in uvec:          #Create arrays of:
        err.append(1)            #differences between best estimates
        uvec0.append(uvec_i[1])  #best estimates 
        fvec.append([1,1,1])     #arrays of values of f
    
    ulist = []
    ulist.append(uvec0 * 1)
    
    if not hasattr(order, "__len__"):
        order = [i for i in range(len(uvec))] #list of indices
    
    counter = 0 
    nochange = 0
    while max(err) > epsilon:
        if nochange == len(uvec0): #no variable has change its best estimate
            break
        nochange = 0
        for i in order: #for each dimension
            x0 = uvec00[i][0]; x1 = uvec00[i][1]; x2 = uvec00[i][2]

            try:
                uvec0[i] = x0; f0 = f(uvec0, params);
                fvec[i][0] = f0
                uvec0[i] = x1; f1 = f(uvec0, params);
                fvec[i][1] = f1
                uvec0[i] = x2; f2 = f(uvec0, params);
                fvec[i][2] = f2
            except:
                uvec0[i] = x0; f0 = f(uvec0);
                fvec[i][0] = f0
                uvec0[i] = x1; f1 = f(uvec0);
                fvec[i][1] = f1
                uvec0[i] = x2; f2 = f(uvec0);
                fvec[i][2] = f2
                
            N = (x2*x2 - x1*x1)*f0 + (x0*x0 - x2*x2)*f1 + (x1*x1 - x0*x0)*f2
            D = (x2 - x1) * f0 + (x0 - x2) * f1 + (x1 - x0) * f2
            if D != 0.:
                x3 = N/D/2
                err[i] = abs(x3 - ulist[-1][i])
            else:
                x3 = ulist[-1][i] * 1
                err[i] = 0
            
            # Append the found point x3 and compute f(x3, others)
            uvec00[i].append(x3)
            uvec0[i] = x3 
            try: f3 = f(uvec0, params)
            except: f3 = f(uvec0)
            fvec[i].append(f3)
            
            # Discard the one with greatest f
            j = np.argmax(fvec[i])
            if x3 == uvec00[i][j]:
                nochange += 1
            fvec[i].pop(j)
            uvec00[i].pop(j)
            
                
            # Take the best point
            k = np.argmin(fvec[i])
            uvec0[i] = uvec00[i][k] * 1.
            ulist.append(uvec0 * 1)
            
            #update counter
            counter += 1
            
            if counter == Nstop:
                print(f"The Nstop = {Nstop} iteration has been reached.")
                break #for loop
        
        if counter == Nstop:
                break #while loop
                
    if track:
        return uvec0, np.array(ulist)
    else:
        return uvec0



def newMin (f, uvec0, h = 1e-6, epsilon = 1e-9, Nstop = 1e4, track = 0, 
            params = float("nan")):
    """
    Apply Newton's Minimisation Method in N-dimensions for a function f(u) of
    N coordiantes with starting guess uvec0.\n
    
    IMPORTANT: If function takes one extra numerical parameter, you have to 
    insert it even if the default value is wanted, otherwise returns NaN, 
    sorry :(.\n
    
    The finite difference derivatives are computed with finite difference 
    h = [hx, hy, ...]. If h is a scalar, then we have h = [h, h, ..., h].\n
    
    Solution convrges when the biggest change in any coordinate is smaller
    than epsilon.\n
    
    Returns the coordinates of the minimum and the 2nd derivatives. If track, 
    also return the list of points. 
    """
    counter = 0
    e = 1
    Dim = len(uvec0)
    ulist = []
    ulist.append(uvec0 * 1)
    if not hasattr(h, "__len__"):
        h = [h for i in uvec0]
    
    H = np.zeros((Dim, Dim)) #prepare the Hessian Matrix
    grad = np.zeros(Dim) #prepare the gradient vector
    
    counter = 0
    while e > epsilon:
        counter += 1
        
        #Create the Hessian
        derivs = {} #keep track of derivatives to avoid computing them twice
        for i in range(Dim):
            for j in range(Dim):
                try:
                    H[i,j] = derivs[str([j, i])]
                except:
                    dij = der2(f, uvec0, h, [i,j], params)
                    H[i,j] = dij
                    derivs[str([i,j])] = dij 
        
        #Get the inverse
        Hinv = np.linalg.inv(H)
        
        #Create the Gradient
        for i in range(Dim):
            grad[i] = der1(f, uvec0, h[i], i, params)
        
        #Find new position
        delta = - np.matmul(Hinv, grad)
        uvec = uvec0 + delta
        
        #Compute error
        e = max(abs(delta))
        # if e == e1:
        #     break
        # else:
        #     e = e1 * 1.0
        
        # Prepare for next iteration:
        uvec0 = np.array(uvec, float)
        ulist.append(uvec0)
         
        if counter == Nstop:
            print(f"The Nstop = {Nstop} iteration has been reached.")
            break
        
    curvature = []
    for i in range(Dim):
        dii = der2(f, uvec0, h, i, params)
        curvature.append(H[i,i])
        
    if track:
        return uvec, curvature, np.array(ulist)
    else:
        return uvec, curvature




def newGradMin (f, uvec0, h = 1e-6, epsilon = 1e-9, Nstop = 2e3, alpha = 1e-2,
            track = 0, params = float("nan")):
    """
    Apply a mix of Newton's and Gradient Minimisation Methods in N-dimensions 
    for a function f(u) of N coordiantes with starting guess uvec0.\n
    
    The mix works as following:
        - compute the change in uvec accoridng to Newton's method.\n
        - if the change of all coordinates is in the opposite direction to the 
        gradient's' we are moving downwards, as required, hence take it.\n
        - otherwise (or if H is not invertible), compute the change according 
        to the gradient method.\n
    
    IMPORTANT: If function takes one extra numerical parameter, you have to 
    insert it even if the default value is wanted, otherwise returns NaN, 
    sorry :(.\n
    
    The finite difference derivatives are computed with finite difference 
    h = [hx, hy, ...]. If h is a scalar, then we have h = [h, h, ..., h].\n
    
    Solution convrges when the biggest change in any coordinate is smaller
    than epsilon.\n
    
    Returns the coordinates of the minimum and the 2nd derivatives. If track, 
    also return the list of points. 
    """
    counter = 0
    e = 1
    Dim = len(uvec0)
    if not hasattr(h, "__len__"):
        h = [h for i in uvec0]
    uvec = np.array(uvec0, float)
    ulist = [uvec0 * 1]
    
    H = np.zeros((Dim, Dim)) #prepare the Hessian Matrix
    grad = np.zeros(Dim) #prepare the gradient vector
    
    counter = 0
    while e > epsilon:
        derivs = {} #keep track of derivatives to avoid computing them twice
        counter += 1
        
        #Create the Gradient
        for i in range(Dim):
            grad[i] = der1(f, uvec0, h[i], i, params)
        
        #Create the Hessian
        for i in range(Dim):
            for j in range(Dim):
                try:
                    H[i,j] = derivs[str([j, i])]
                except:
                    dij = der2(f, uvec0, h, [i,j], params)
                    H[i,j] = dij
                    derivs[str([i,j])] = dij 
        
        #Get the inverse and delta
        try:
            Hinv = np.linalg.inv(H)
            delta = - np.matmul(Hinv, grad)
            uvec = uvec0 + delta
            if f(uvec) > f(uvec0): # if it didn't go down, change method
                delta = - alpha * grad
                uvec = uvec0 + delta
    
        except: #H non invertible
            delta = - alpha * grad
            uvec = uvec0 + delta
         
        
        #Compute error
        e1 = max(abs(delta))
        if e == e1: #same overshooting back and forth
            break
        else:
            e = e1 * 1.0
        
        # Prepare for next iteration:
        uvec0 = np.array(uvec, float)
        ulist.append(uvec0 * 1)
         
        if counter == Nstop:
            print(f"The Nstop = {Nstop} iteration has been reached.")
            break
        
    curvature = []
    for i in range(Dim):
        curvature.append(H[i,i])
    if track:
        return uvec, curvature, np.array(ulist)
    else:
        return uvec, curvature




def quasiNewMin (f, uvec, h = 1e-6, alphamax = 2, alphatry = 50, 
                 method = "BFGS", epsilon = 1e-9, Nstop = 1e3, track = 0,
                      params = float("nan")):
    """
    Quasi-Newton Method of minimisation in N-dimensions for a function f(u) of
    N coordiantes with starting guess uvec0.\n
    
    IMPORTANT: If function takes one extra numerical parameter, you have to 
    insert it even if the default value is wanted, otherwise returns NaN, 
    sorry :(.\n
    
    The finite difference derivatives are computed with finite difference 
    h = [hx, hy, ...]. If h is a scalar, then we have h = [h, h, ..., h].\n
    
    - alphamax and alphatry: coefficient alpha can have at every step maximum
    value alphamax, being it reduced up to alphatry until it meets Armijo
    rule, thus ensuring the function is being sufficiently minimised.
    
    The method entry can be "BFGS" (default choice) or "DFP".\n
    
    Solution convrges when the biggest change in any coordinate is smaller
    than epsilon.\n
    
    Returns the coordinates of the minimum.
    """
    uvec0 = np.array(uvec, float)
    ulist = [uvec0 * 1]
    counter = 0
    e = 1
    Dim = len(uvec0)
    if not hasattr(h, "__len__"):
        h = [h for i in uvec0]
        
    alpha = []
    alphachange = alphamax / 1e-4
    alphareduce = np.power(alphachange, (1/(int(alphatry)-1)))
    for i in range(0, int(alphatry)):
        alpha.append(alphamax / alphareduce**i)
        
    
    G = np.identity(Dim)             #prepare the update matrix
    grad0 = np.zeros(Dim)            #prepare the old gradient vector
    grad1 = np.zeros(Dim)            #prepare the new gradient vector
    out_delta = np.zeros((Dim, Dim)) #prepare the outer product of delta
    out_gamma = np.zeros((Dim, Dim)) #prepare the outer product of gamma
    out_dg = np.zeros((Dim, Dim))    #prepare outer product delta gamma
    out_gd = np.zeros((Dim, Dim))    #prepare outer product gamma delta
    I = np.identity(Dim)             #prepare identity
    
    #First Iteration is "special", hence goes outside the loop
    
    #Get the Gradient
    for i in range(Dim):
        grad0[i] = ( der1(f, uvec0, h[i], i, params) ) 
        
    #Find delta, new position uvec1
    try:    f1 = f(uvec0, params)
    except: f1 = f(uvec0)
    for i in range(0, alphatry):
        deltatry = - alpha[i] * np.matmul(G, grad0) / 10 #/10 only 1st step
        uvec1try = uvec0 + deltatry
        try: f1try = f(uvec1try, params)
        except: f1try = f(uvec1try)
        armijoWolfe1 = f1try < f1 + 1e-4 * np.dot(deltatry, grad0)
        if armijoWolfe1:
            uvec1 = np.array(uvec1try, float)
            delta = np.array(deltatry, float)
            f1 = f1try * 1.
            break
        else:
            continue
    
    #Get the Gradient
    for i in range(Dim):
        grad1[i] = ( der1(f, uvec1, h[i], i, params) ) 
        
    # Prepare for next iteration:
    uvec0 = np.array(uvec1, float)
    gamma = grad1 - grad0
    grad0 = np.array(grad1, float)
    ulist.append(uvec0 * 1)
    
    counter = -1
    while e > epsilon:
        counter += 1
        if counter == Nstop:
            print(f"The Nstop={Nstop} iteration was reached at following use.")
            break
        
        elif method == "DFP":
            #Get the outer products
            for i in range(Dim):
                for j in range(Dim):
                    out_delta[i,j] = delta[i] * delta[j]
                    out_gamma[i,j] = gamma[i] * gamma[j]
            #Update G (DFP)
            gd = np.dot(gamma, delta)
            if gd == 0:
                print("uoo")
                break
            A = out_delta[i] / gd
            Bn = np.matmul(G, np.matmul(out_gamma, G))
            Bd = np.dot(gamma, np.matmul(G, gamma))
            G += A - Bn/Bd
        
        else: 
            #Get the outer products
            for i in range(Dim):
                for j in range(Dim):
                    out_dg[i,j]    = delta[i] * gamma[j]
                    out_gd[i,j]    = gamma[i] * delta[j]
                    out_delta[i,j] = delta[i] * delta[j]
            #Update G (BFGS):
            gd = np.dot(gamma, delta)
            if gd == 0:
                print("uoo")
                break
            A = I - out_dg / gd
            B = I - out_gd / gd
            C = out_delta / gd
            G = np.matmul(A, np.matmul(G, B)) + C
        
        #Find delta, new position uvec1
        delta = - 1e-3 * np.matmul(G, grad0)
        uvec1 = uvec0 + delta
        try:    f1 = f(uvec0, params)
        except: f1 = f(uvec0)
        for i in range(0, int(alphatry)):
            
            deltatry = - alpha[i] * np.matmul(G, grad0)
            uvec1try = uvec0 + deltatry
            
            try: f1try = f(uvec1try, params)
            except: f1try = f(uvec1try)
            for i in range(Dim):
                grad1[i] = der1(f, uvec1, h[i], i, params)
            
            dgtry = np.dot(deltatry, grad0)
            armijoWolfe1 = f1try <= f1 + 1e-4 * dgtry
            if armijoWolfe1: #Armijo-Wolfe
                uvec1 = np.array(uvec1try, float)
                delta = np.array(deltatry, float)
                break
            else:
                continue
        
        #Get the Gradient
        for i in range(Dim):
            grad1[i] = der1(f, uvec1, h[i], i, params)
        
        #Compute change
        e = max(abs(delta))
        
        # Prepare for next iteration:
        uvec0 = np.array(uvec1, float)
        gamma = grad1 - grad0
        grad0 = np.array(grad1, float)
        ulist.append(uvec0 * 1)
     
    #Loop finished
    for ui in uvec0:
        if np.isnan(ui):
            uvec0 = ulist[-2]
            ulist.pop(-1)
            break
    
    if track:
        return uvec0, np.array(ulist)
    else:
        return uvec0





def metropolis (f, interval, iters = 5e3, kT0 = 100, anneal = 0.5, scan = 1e3, 
                step = 0.2, close_factor = 2, track = False, halving = 0.5,
                params = float("nan")):
    """
    Perform a Montecarlo Metropolis (Annealing) minimisation of N-dimensional 
    function f. The probability of jumping around even if not minimum is based 
    on P = exp(-deltaf/kT). \n
    
    IMPORTANT: If function takes one extra numerical parameter, you have to 
    insert it even if the default value is wanted, otherwise returns NaN, 
    sorry :(.\n
            
    - interval defines the scanned region [[xmin, ymin,...],[xmax, ymax...]]\n
    
    - iters is the number or iterations.\n
    
    - kT0 is the starting kT, the bigger kT, the more probable to "jump" 
    around.\n
    
    - if anneal is given, lT will then be reduced at every iteration up to,
    at the end, kT*(1-anneal) amount (0 < anneal <=1).\n
    
    - scan defines the number of evaluation at the very beginning, to get a 
    fast image of the function's landscape.\n
    
    - each step is taken randomly with gaussian pdf, centred at latest point,
    with sigma being, at the most, step * size of interval (0 < step <=1). 
    Note, the point will be rejected if outside the interval, but another will
    be taken, in that same direction, with uniform distribution.\n
    
    - close_factor modifies sigma accoridng to how close we are to the 
    one we think, at the moment, is the minimum, such that: \n\n
        
     sigma = step*size*close_factor ** (abs(position_now - minimum)/size - 1)\n 
      
     hence sigma goes from step*size/close_factor to step*size.\n
     
    - if track, also return a list of all positions the algorithm visited.
    
    - if halving!=0, after halving*iters iterations (0 <= halving <= 1), go to 
    the one we think is the global minimum, half the interval around it, repeat
    procedure, recharging the temperature. 
    """
    
    interval = np.array(interval, float)
    size = abs(interval[1] - interval[0])
    step = step * size
    
    #starting point
    i = 0
    while i < scan:
        i += 1
        uvec = np.random.uniform(interval[0], interval[1])
        try:    E = f(uvec, params)
        except: E = f(uvec)
        if i == 1:
            Emin = E * 1.0
            umin = np.array(uvec, float)
        else:
            if E <= Emin:
                Emin = E * 1.0
                umin = np.array(uvec, float)
            else:
                continue
            
    reduc= anneal / iters # to anneal kT, see end of for loop
    uveclist = [umin]     # start keeping track
    
    if halving:
        times = int(1 // halving)
        sets = [(iters*halving)]*times + [iters - int(iters*halving)*times]
        sets = [int(setj) for setj in sets]
    else:
        sets = [iters]
        
    for setj in sets:
        uvec0 = np.array(umin, float) 
        E0 = Emin * 1.                
        kT   = kT0 * 1.
        for i in range(setj):
            # find new point
            sigma = step * close_factor ** (abs(uvec0 - umin)/size - 1)
            uvec  = np.random.normal(uvec0, sigma)
            
            # put new point's coordinates in the interval if outside
            for j in range(len(uvec)):
                if uvec[j] <= interval[0][j]:
                    uvec[j] = np.random.uniform(interval[0][j], uvec0[j])
                elif uvec[j] >= interval[1][j]:
                    uvec[j] = np.random.uniform(uvec0[j], interval[1][j])
            
            try:    E = f(uvec, params)
            except: E = f(uvec)
            deltaE = E - E0
            
            if deltaE <= 0: #we found a smaller energy: change point
                uvec0 = np.array(uvec, float)
                E0    = E * 1.
                uveclist.append(uvec0)
                if E < Emin: 
                    # set new total minimum
                    umin = np.array(uvec, float)
                    Emin = E * 1.
            else:
                # smaybe change starting point for next iteration
                if np.exp(- deltaE / kT) >= np.random.uniform(0, 1):
                    uvec0 = np.array(uvec, float)
                    E0    = E * 1.
                    uveclist.append(uvec0)
                else:
                    continue
            
            if anneal:
                kT -= reduc * kT0
        
        #Do the halving
        step /= 2
        delta = umin - interval
        interval = umin - delta/2
        
    if track:
        return umin, Emin, np.array(uveclist)
    
    else:
        return umin, Emin




"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
STATISTICS          STATISTICS          STATISTICS          STATISTICS
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
def chi2 (observed, expected, uncertainty = 0, Nparams = 1, correction = 0):
    """
    Compute Chi2 and p-value and return them as a tuple. \n
    If uncertainty is not specified, binned distribution is assumed, hence
    having sigma^2[i] = expected[i].
    
    If the expected number of entries is too low, might want a correction:
        - Yates (for Nparam = 1)
        - Williams in general, which will overwrite Yates if Nparam > 1.
    """
    N = len(observed)
    Chi2 = 0
    if len(expected) != N:
        raise Exception("Observed and expected arrays have different length!")
    
    #1
    if correction != "Yates" or Nparam != 1:
        #If multiple uncertainties are given, one per datapoint
        if hasattr(uncertainty, "__len__"):
            if len(uncertainty) != N:
                raise Exception("Observed data and uncertainty have different",
                                "lenght!")
            
            else:
                for i in range(N):
                    pull = (observed[i] - expected[i]) / uncertainty[i]
                    Chi2 += pull*pull
        
        # elif one same numerical uncertainty is provided for all datapoints
        elif uncertainty:
            for i in range(N):
                pull_numerator = observed[i] - expected[i]
                Chi2 += pull_numerator * pull_numerator
            Chi2 /= uncertainty*uncertainty
    
        # elif no uncertainty provided: assume binned distribution
        else:
            for i in range(N):
                pull_numerator = observed[i] - expected[i]
                Chi2 += pull_numerator * pull_numerator / expected[i]
    
    #2
    elif correction == "Yates" and Nparam == 1:
       #If multiple uncertainties are given, one per datapoint
       if hasattr(uncertainty, "__len__"):
           if len(uncertainty) != N:
               raise Exception("Observed data and uncertainty have",
                               "different lenght!")
           
           else:
               for i in range(N):
                   pull = abs(observed[i] - expected[i])-0.5
                   pull /= uncertainty[i]
                   Chi2 += pull*pull
       
       # elif one same numerical uncertainty is provided for all 
       elif uncertainty:
           for i in range(N):
               pull_numerator = abs(observed[i] - expected[i]) - 0.5
               Chi2 += pull_numerator * pull_numerator
           Chi2 /= uncertainty*uncertainty
   
       # elif no uncertainty provided: assume binned distribution
       else:
           for i in range(N):
               pull_numerator = abs(observed[i] - expected[i]) - 0.5
               Chi2 += pull_numerator * pull_numerator / expected[i]
    
    #3, but assuming 1
    if (correction == "Yates" and Nparam != 1) or (correction == "Williams"):
        dof = N - Nparams
        q = 1 + (N*N - 1) / (6 * sum(observed) * dof)
        Chi2 /= q
    
    return Chi2
        
     
        
def pvalue (Chi2, dof, epsilon = 1e-6, Nstop = 20):
    
    # dof taken from pvalue scope
    def integrand1 (t):
        if t == 0 and dof == 2:
            return 1
        elif t == 0:
            return 0
        else:
            #return t**(dof/2 - 1) * np.exp(-t)
            return np.exp( (dof/2 - 1.) * np.log(t) - t)
            
    gamma = trapsimps(integrand1, [0, Chi2/2      ], epsilon, Nstop)
    Gamma = trapsimps(integrand1, [0, float("inf")], epsilon, Nstop)

    p = 1 - gamma/Gamma
    if p < 0:
        p = 0.
    return p




###
#  DISCARDED OPTIONS FOR FUNCTIONS WHICH MAY AGAIN BE USEFUL
###


# def quasiNewMin (f, startvec, h = 1e-6, alpha = 1e-3, alpha0reduce = 10000, 
#                  method = "BFGS", epsilon = 1e-9, Nstop = 5e3, track = 0, 
#                      params = float("nan")):
#     """
#     Quasi-Newton Method of minimisation in N-dimensions for a function f(u) of
#     N coordiantes with starting guess uvec0.\n
    
#     IMPORTANT: If function takes one extra numerical parameter, you have to 
#     insert it even if the default value is wanted, otherwise returns NaN, 
#     sorry :(.\n
    
#     The finite difference derivatives are computed with finite difference 
#     h = [hx, hy, ...]. If h is a scalar, then we have h = [h, h, ..., h].\n
    
#     The coefficient alpha has default value 1e-3.If alpha0reduce si a number,
#     the first alpha will be divided by alphareduce, all the otehr will be
#     alpha. \n
    
#     The method entry can be "BFGS" (default choice) or "DFP".\n
    
#     Solution convrges when the biggest change in any coordinate is smaller
#     than epsilon.\n
    
#     Returns the coordinates of the minimum. If track, also return the list of
#     points. 
#     """
#     uvec0 = np.array(startvec, float)
#     ulist = [uvec0]
#     counter = 0
#     e = 1
#     Dim = len(uvec0)
#     if not hasattr(h, "__len__"):
#         h = [h for i in uvec0]
    
#     G = np.identity(Dim)             #prepare the update matrix
#     grad0 = np.zeros(Dim)            #prepare the old gradient vector
#     grad1 = np.zeros(Dim)            #prepare the new gradient vector
#     out_delta = np.zeros((Dim, Dim)) #prepare the outer product of delta
#     out_gamma = np.zeros((Dim, Dim)) #prepare the outer product of gamma
#     out_dg = np.zeros((Dim, Dim))    #prepare outer product delta gamma
#     out_gd = np.zeros((Dim, Dim))    #prepare outer product gamma delta
#     I = np.identity(Dim)             #prepare identity
    
#     #First Iteration is "special", hence goes outside the loop
    
#     #Get the Gradient
#     for i in range(Dim):
#         grad0[i] = ( der1(f, uvec0, h[i], i, params) )    
#     #Get delta and the new position
#     if alpha0reduce:
#         alpha0 = alpha / alpha0reduce
#         delta = - alpha0 * np.matmul(G, grad0)
#     else:
#         delta = - alpha * np.matmul(G, grad0)
#     uvec = uvec0 + delta
#     # Prepare for next iteration:
#     uvec0 = np.array(uvec, float)
#     ulist.append(uvec0)
    
#     counter = 0
#     while e > epsilon:
#         counter += 1
        
#         #Get the next gradient
#         for i in range(Dim):
#             grad1[i] = ( der1(f, uvec0, h[i], i, params) )
        
#         #Get gamma
#         gamma = grad1 - grad0
            
#         #Get the outer products
#         for i in range(Dim):
#             for j in range(Dim):
#                 out_delta[i,j] = delta[i] * delta[j]
#                 out_gamma[i,j] = gamma[i] * gamma[j]
#                 out_dg[i,j]    = delta[i] * gamma[j]
#                 out_gd[i,j]    = gamma[i] * delta[j]
        
#         if method == "DFP":
#             #Update G (DFP)
#             gd = np.dot(gamma, delta)
#             if gd == 0:
#                 break
#             A = out_delta[i] / gd
#             Bn = np.matmul(G, np.matmul(out_gamma, G))
#             Bd = np.dot(gamma, np.matmul(G, gamma))
#             G += + A - Bn/Bd
        
#         else: 
#         #Update G (BFGS):
#             gd = np.dot(gamma, delta)
#             if gd == 0:
#                 break
#             A = I - out_dg / gd
#             B = I - out_gd / gd
#             C = out_delta / gd
#             G = np.matmul(A, np.matmul(G, B)) + C
        
#         # # Get delta and the new position, performing an approximate
#         # # line search for minimum varying alpha
#         # fs = []; us = []
#         # delta0 = - alpha * np.matmul(G, grad1)
#         # for j in range(0,20):
#         #     delta = delta0 / np.sqrt(5) ** j
#         #     uvec = uvec0 + delta
#         #     us.append(uvec)
#         #     try:    fs.append(f(uvec, params))
#         #     except: fs.append(f(uvec))
#         # j = np.argmin(fs)
#         # uvec = us[j]
        
#         #Get delta and new position
#         delta = - alpha * np.matmul(G, grad1)
#         uvec = uvec0 + delta
        
#         #Compute change
#         e = max(abs(delta))
#         #e = max(abs(uvec - uvec0))
            
#         # Prepare for next iteration:
#         uvec0 = np.array(uvec, float)
#         grad0 = np.array(grad1, float)
#         ulist.append(uvec0)
        
#         if counter == Nstop:
#             print(f"The Nstop={Nstop} iteration was reached at following use.")
#             break
    
#     if track:
#         return uvec0, np.array(ulist)
#     else:
#         return uvec0



# def quasiNewMin (f, uvec, h = 1e-6, alphamax = 1e-2, method = "BFGS", \
#                  epsilon = 1e-6, Nstop = 1e4, \
#                      params = float("nan")):
#     """
#     Quasi-Newton Method of minimisation in N-dimensions for a function f(u) of
#     N coordiantes with starting guess uvec0.\n
    
#     IMPORTANT: If function takes one extra numerical parameter, you have to 
#     insert it even if the default value is wanted, otherwise returns NaN, 
#     sorry :(.\n
    
#     The finite difference derivatives are computed with finite difference 
#     h = [hx, hy, ...]. If h is a scalar, then we have h = [h, h, ..., h].\n
    
#     The coefficient alpha has default value 1e-3.If alpha0reduce si a number,
#     the first alpha will be divided by alphareduce, all the otehr will be
#     alpha. \n
    
#     The method entry can be "BFGS" (default choice) or "DFP".\n
    
#     Solution convrges when the biggest change in any coordinate is smaller
#     than epsilon.\n
    
#     Returns the coordinates of the minimum.
#     """
#     uvec0 = np.array(uvec, float)
#     counter = 0
#     e = 1
#     Dim = len(uvec0)
#     if not hasattr(h, "__len__"):
#         h = [h for i in uvec0]
    
#     G = np.identity(Dim)             #prepare the update matrix
#     grad0 = np.zeros(Dim)            #prepare the old gradient vector
#     grad1 = np.zeros(Dim)            #prepare the new gradient vector
#     out_delta = np.zeros((Dim, Dim)) #prepare the outer product of delta
#     out_gamma = np.zeros((Dim, Dim)) #prepare the outer product of gamma
#     out_dg = np.zeros((Dim, Dim))    #prepare outer product delta gamma
#     out_gd = np.zeros((Dim, Dim))    #prepare outer product gamma delta
#     I = np.identity(Dim)             #prepare identity
    
#     #First Iteration is "special", hence goes outside the loop
    
#     #Get the Gradient
#     for i in range(Dim):
#         grad0[i] = ( der1(f, uvec0, h[i], i, params) ) 
        
#     #Find delta, new position uvec1, new gradient grad1 with alpha 
#     #satisfying Wolfe conditions
#     wolfe = 0; count = 0; sqrt10 = np.sqrt(10)
#     delta = - alphamax * np.matmul(G, grad0)
#     while not wolfe:
#         count += 1
#         uvec1 = uvec0 + delta
#         delgrad  = np.dot(delta, grad0)
#         for i in range(Dim):
#                 grad1[i] = ( der1(f, uvec1, h[i], i, params) ) 
#         try: 
#             wolfe1 = f(uvec1, params) <= f(uvec0, params) + 1e-4 * delgrad
#             wolfe2 = np.dot(delta, grad1) <= 0.9 * delgrad
#         except:
#             wolfe1  = f(uvec1) <= f(uvec0) + 1e-4 * delgrad
#             wolfe2 = np.dot(delta, grad1) <= 0.9 * delgrad
#         print (wolfe1, wolfe2)
#         if (wolfe1 and wolfe2) or count == 9:
#             wolfe = 1
#         else:
#             delta /= sqrt10
        
#     # Prepare for next iteration:
#     uvec0 = np.array(uvec1, float)
#     gamma = grad1 - grad0
#     grad0 = np.array(grad1, float)
    
    
#     counter = -1
#     while e > epsilon:
#         counter += 1
#         if counter == Nstop:
#             print(f"The Nstop={Nstop} iteration was reached at following use.")
#             break
        
#         elif method == "DFP":
#             #Get the outer products
#             for i in range(Dim):
#                 for j in range(Dim):
#                     out_delta[i,j] = delta[i] * delta[j]
#                     out_gamma[i,j] = gamma[i] * gamma[j]
#             #Update G (DFP)
#             A = out_delta[i] / np.dot(gamma, delta)
#             Bn = np.matmul(G, np.matmul(out_gamma, G))
#             Bd = np.dot(gamma, np.matmul(G, gamma))
#             G += A - Bn/Bd
        
#         else: 
#             #Get the outer products
#             for i in range(Dim):
#                 for j in range(Dim):
#                     out_dg[i,j]    = delta[i] * gamma[j]
#                     out_gd[i,j]    = gamma[i] * delta[j]
#             #Update G (BFGS):
#             gd = np.dot(gamma, delta)
#             A = I - out_dg / gd
#             B = I - out_gd / gd
#             C = out_delta / gd
#             G = np.matmul(A, np.matmul(G, B)) + C
        
#         #Find delta, new position uvec1, new gradient grad1 with alpha 
#         #satisfying strong Wolfe conditions
#         wolfe = 0; count = 0; sqrt10 = np.sqrt(10)
#         delta = - alphamax * np.matmul(G, grad0)
#         while not wolfe:
#             count += 1
#             uvec1 = uvec0 + delta
#             delgrad  = np.dot(delta, grad0)
#             for i in range(Dim):
#                 grad1[i] = ( der1(f, uvec1, h[i], i, params) ) 
#             try: 
#                 wolfe1 = f(uvec1, params) <= f(uvec0, params) + 1e-4 * delgrad
#                 wolfe2 = np.dot(delta, grad1) <= 0.9 * delgrad
#             except:
#                 wolfe1  = f(uvec1) <= f(uvec0) + 1e-4 * delgrad
#                 wolfe2 = np.dot(delta, grad1) <= 0.9 * delgrad
#             print (wolfe1, wolfe2)
#             if (wolfe1 and wolfe2) or count == 9:
#                 wolfe = 1
#             else:
#                 delta /= sqrt10
#         print(delta / np.matmul(G, grad0))
#         print(delta)
        
#         #Compute change
#         e = max(abs(delta))
#         if np.isnan(e):
#             print("You had reached the point the difference between gradients",
#                   " is so small you would get NaN, hence I had to stop.")
#             break
        
#         # Prepare for next iteration:
#         uvec0 = np.array(uvec1, float)
#         gamma = grad1 - grad0
#         grad0 = np.array(grad1, float)
    
#     return uvec0







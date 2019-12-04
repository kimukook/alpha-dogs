#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:25:12 2017

@author: mousse
"""
import numpy as np
from scipy.spatial import Delaunay
from scipy import optimize
from dogs import Utils
from dogs import interpolation

'''
 constantK.py file contains the constant-K continuous search function designed for AlphaDOGS and DeltaDOGS. Using the package optimize
 from scipy.

 The constant-K continuous search function has the form:

 Sc(x) = P(x) - K*e(x):

     Sc(x):     constant-K continuous search function;
     P(x):      Interpolation function: 
                    For AlphaDOGS: regressionparameterization because the function evaluation contains noise;
                    For DeltaDOGS: interpolationparameterization;
     e(x):      The uncertainty function constructed based on Delaunay triangulation.

 Function contained:
     tringulation_search_bound_constantK:   Search for the minimizer of continuous search function over all the Delaunay simplices 
                                  over the entire domain.
     Constant_K_Search:                     Search over a specific simplex.
     Continuous_search_cost:                Calculate the value of continuous search function.

'''


#################################### Constant K method ####################################
# debugging
# inter_par = self.inter_par
# xi = xs
# K = self.K*self.K0
def tringulation_search_bound_constantK(inter_par, xi, K, ind_min):
    '''
    This function is the core of constant-K continuous search function.
    :param inter_par: Contains interpolation information w, v.
    :param xi: The union of xE(Evaluated points) and xU(Support points)
    :param K: Tuning parameter for constant-K, K = K*K0. K0 is the range of yE.
    :param ind_min: The correspoding index of minimum of yE.
    :return: The minimizer, xc, and minimum, yc, of continuous search function.
    '''
    inf = 1e+20
    n = xi.shape[0]
    # Delaunay Triangulation
    if n == 1:
        sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
        tri = np.zeros((xi.shape[1] - 1, 2))
        tri[:, 0] = sx[:xi.shape[1] - 1]
        tri[:, 1] = sx[1:]
        tri = tri.astype(np.int32)
    else:
        options = 'Qt Qbb Qc' if n <= 3 else 'Qt Qbb Qc Qx'
        tri = Delaunay(xi.T, qhull_options=options).simplices
        keep = np.ones(len(tri), dtype=bool)
        for i, t in enumerate(tri):
            if abs(np.linalg.det(np.hstack((xi.T[t], np.ones([1, n + 1]).T)))) < 1E-15:
                keep[i] = False  # Point is coplanar, we don't want to keep it
        tri = tri[keep]
    # Search the minimum of the synthetic quadratic model
    Sc = np.zeros([np.shape(tri)[0]])
    Scl = np.zeros([np.shape(tri)[0]])
    for ii in range(np.shape(tri)[0]):
        # R2-circumradius, xc-circumcircle center
        R2, xc = Utils.circhyp(xi[:, tri[ii, :]], n)
        # x is the center of the current simplex
        x = np.dot(xi[:, tri[ii, :]], np.ones([n + 1, 1]) / (n + 1))
        Sc[ii] = inter_par.inter_val(x) - K * (R2 - np.linalg.norm(x - xc) ** 2)
        if np.sum(ind_min == tri[ii, :]):
            Scl[ii] = np.copy(Sc[ii])
        else:
            Scl[ii] = inf

    index = np.array([np.argmin(Sc), np.argmin(Scl)])
    xm = np.zeros((n, 2))
    ym = np.zeros(2)
    for i in range(2):
        temp_x, ym[i] = Constant_K_Search(xi[:, tri[index[i], :]], inter_par, K)
        xm[:, i] = np.copy(temp_x)
    ymin = np.min(ym)
    xmin = xm[:, np.argmin(ym)].reshape(-1, 1)

    if np.argmin(ym) == 0:
        result = 'global'
    else:
        result = 'local'

    return xmin, ymin, result


def Constant_K_Search(simplex, inter_par, K, lb=[], ub=[]):
    n = simplex.shape[0]
    R2, xc = Utils.circhyp(simplex, n)
    x = np.dot(simplex, np.ones([n + 1, 1]) / (n + 1))
    costfun = lambda x: Continuous_search_cost(x, inter_par, xc, R2, K)[0]
    costjac = lambda x: Continuous_search_cost(x, inter_par, xc, R2, K)[1]
    opt = {'disp': False}
    bnds = tuple([(0, 1) for i in range(int(n))])
    res = optimize.minimize(costfun, x, jac=costjac, method='TNC', bounds=bnds, options=opt)
    x = res.x
    y = res.fun
    return x, y


# Value of constant K search
def Continuous_search_cost(x, inter_par, xc, R2, K):
    x = x.reshape(-1, 1)
    M = inter_par.inter_val(x) - K * (R2 - np.linalg.norm(x - xc) ** 2)
    DM = inter_par.inter_grad(x) + 2 * K * (x - xc)
    # if optm method is chosen as TNC, use DM.T[0]
    return M, DM.T[0]

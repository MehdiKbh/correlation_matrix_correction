# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:50:43 2020

@author: Kaabache
"""

import numpy as np
import pandas as pd
import scipy as sc
import scipy.linalg as la

import sys



#Higham Method
    
def Higham_method(A, W, max_iter=1e2, tol=1e-10, affich=0):
    
    """
    Reference:  N. J. Higham, Computing the nearest correlation
    matrix---A problem from finance. IMA J. Numer. Anal.,
    22(3):329-343, 2002.
    """
    
    n = A.shape[0] #Size of the problem

    dS = np.zeros(n)
    Y_new, Y_old = A, np.zeros(n)
    X_new, X_old = np.zeros(n), np.zeros(n)
    Y_dist, X_dist, XY_dist = float('inf'), float('inf'), float('inf')
    
    k = 0
    while(k<=max_iter and max(Y_dist, X_dist, XY_dist)>tol):
        X_old = np.copy(X_new)
        Y_old = np.copy(Y_new)
        
        R = Y_new - dS
        X_new = Projection_S(R, W)
        dS = X_new - R
        Y_new = Projection_U(X_new, W)
        
        Y_dist = np.linalg.norm(Y_new - Y_old, ord=np.inf)/np.linalg.norm(Y_new, ord=np.inf)
        X_dist = np.linalg.norm(X_new - X_old, ord=np.inf)/np.linalg.norm(X_new, ord=np.inf)
        XY_dist = np.linalg.norm(X_new - Y_new, ord=np.inf)/((np.linalg.norm(Y_new, ord=np.inf) + np.linalg.norm(X_new, ord=np.inf))/2.)
        
        k+=1
        
    if(k>max_iter):
        print("##############################################################################################")
        print("WARNING: No solution found in ", k-1, "iterations.\nYou can try with more iterations.")
        print("##############################################################################################")
    else:
        if(affich==0):
            print("##############################################################################################")
            print("Summary:")
            print("Number of iterations = {}\nConvergence XY = {}\nX sym positive semi-def? {}".format(k, XY_dist, isSymPosSemiDef(X_new)))
            print("##############################################################################################")

    return Y_new




#Some useful functions

def isDiag(M):
    i, j = M.shape
    assert i == j 
    test = M.reshape(-1)[:-1].reshape(i-1, j+1)
    return ~np.any(test[:, 1:])

def isSym(M, tol=1e-8):
    return np.allclose(M, M.T, tol)

def isSymPosSemiDef(M, tol=1e-8):
    if not (isSym(M)):
        return False   
    else:
        return np.all(np.linalg.eigvals(M) > -tol)
    
def isUnit(M):
    res = True
    k = 0
    while(k<M.shape[0] and res==True):
        if (M[k,k]!=1):
            res = False
        k+=1
    return res

def W_norm(A, W):
    n = A.shape[0] #Size of the problem
    
    #Check if the the W-norm is well defined (works only for squared matrices)
    if not (W.shape[0] == W.shape[1] == A.shape[0] == A.shape[1]): sys.exit("Matrices A and W should be squares and of same dimension")
    if not isSymPosSemiDef(W): sys.exit("Matrix W should be symetric positive semi-definite")
    
    W_rs = la.sqrtm(W)
    M = W_rs @ A @ W_rs
    
    return np.linalg.norm(M)



#Projections used on S and U
    
def Projection_U(A, W):
    n = A.shape[0] #Size of the problem
    
    #Check if the projection wrt the W-norm is well defined
    if not isSym(A): sys.exit("Matrix A should be symetric")
    if not (W.shape[0] == W.shape[1] == n): sys.exit("Matrices A and W should be of same dimension")
    if not isSymPosSemiDef(W): sys.exit("Matrix W should be symetric positive semi-definite")
    
    #Avoid any computations if not needed
    if isUnit(A):
        X = np.copy(A)
    
    else:
    
        if(isDiag(W)):
            X = np.copy(A)    #Return matrix
            for i in range(n):
                X[i,i] = 1.

        else:
            W_inv = np.array(np.linalg.inv(W))
            M = W_inv * W_inv
            b = A.diagonal() - 1
            theta = np.linalg.solve(M, b)
            X = A - (W_inv @ (theta*np.identity(n)) @ W_inv)

    
    return X

def Projection_S(A, W, tol=1e-8):
    n = A.shape[0] #Size of the problem
    
    #Check if the projection wrt the W-norm is well defined
    if not isSym(A): sys.exit("Matrix A should be symetric")
    if not (W.shape[0] == W.shape[1] == n): sys.exit("Matrices A and W should be of same dimension")
    if not isSymPosSemiDef(W): sys.exit("Matrix W should be symetric positive semi-definite")
        
    W_rs = la.sqrtm(W)
    M = np.dot(W_rs, np.dot(A, W_rs))
    
    eigvals, D = np.linalg.eig(M)
    eigvals = np.array([ max(eigvals[i], 0.) for i in range(n) ])
    
    M_positive = D @ (eigvals * np.identity(n)) @ D.T
    W_rs_inv = np.linalg.inv(W_rs)
    X = W_rs_inv @ M_positive @ W_rs_inv
    
    dist = W_norm(A-X,W)
    
    return X
    

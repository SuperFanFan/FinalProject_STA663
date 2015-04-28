
from __future__ import division
import numpy as np
import scipy.stats as stats

y = np.loadtxt("data_test")
sigma = np.sqrt(3.0)

# (1) Update S_n
def update_Sn(y, n, m, Ptran, theta, s):
    """ Update latent states S_n 
        Args: y - vector of observations
              n - number of obervations
              m - number of change point
              Ptran - Transition matrix
              theta - model parameters
              s - current state values for all time
        Return: F_lag - lag 1 predictive density
                F - posterior conditional density
                s_new - sampled latent states of length n """
    
    # check input
    if(any(np.delete(np.diag(Ptran), -1) <= 0.0)):
        return "Error - transition probabilities should be within range 0 to 1."
    
    # read the current s values
    s_new = s
    sigma = np.sqrt(3.0)
    
    # define quantities
    F_lag, F = np.zeros((n, m + 1)), np.zeros((n, m + 1))                                  
    F_lag[0, 0], F[0, 0] = 1, 1
    
    for i in range(1, n):
        for j in range(m + 1):
            F_lag[i,j] = (Ptran[:,j]).dot(F[i - 1,:])
        F[i,:] = F_lag[i,:] * stats.norm.pdf(y[i], loc = theta, scale = sigma)
        F[i,:] = F[i,:] / np.sum(F[i,:])
        
    # Sampling s_t
    for k in range(n - 2, 0, -1): # omit update s_n and s_1 because of their degeneracy
        pmfs = F[k,:] * Ptran[:,s_new[k + 1]]
        pmfs = pmfs / np.sum(pmfs)
        s_new[k] = np.random.choice(np.arange(m + 1), p = pmfs)
        
    return F_lag, F, s_new

# (2) Update P
def update_P(a, b, m, s, Ptran_star):
    """ Update transition matrix P 
        Args: a,b - prior beta parameters
              m - number of change points
              s - current sample of state
              Ptran_star - MLE of the transition matrix
        Return: nk - number of the same states
                Ptran - updated transition matrix 
                f_Ptran_star - marginal likelihood calculation involving Ptran """
    
    # define quantities
    nk = np.zeros(m + 1)
    Ptran = np.zeros((m + 1, m + 1))
    Ptran[-1, -1] = 1
    f = np.zeros(m)
    
    # number of same states
    for i in range(m + 1):
        nk[i] = np.sum(s == i)
    nii = nk - 1
    
    # update P
    for j in range(m):
        Ptran[j, j] = stats.beta.rvs(a + nii[j], b + 1)
        Ptran[j, j + 1] = 1.0 - Ptran[j, j]
        f[j] = stats.beta.pdf(Ptran_star[j, j], a + nii[j], b + 1)
    f_Ptran_star = np.prod(f)
    
    return nk, Ptran, f_Ptran_star

# (3) Update Theta - Gaussian Model
def update_Theta(c, d, m, y, s, nk, theta_star):
    """ Update model parameters Theta 
        Args: c,d - prior normal parameters
              m - number of change points
              y - vector of observations
              s - current sample of state
              nk - number of the same states
              theta_star - MLE of theta
        Return: theta - updated model parameters 
                f_theta_star - marginal likelihood calculation involving theta """
    
    # define quantities
    sigma = np.sqrt(3.0)
    theta = np.repeat(2.0, m + 1)
    f = np.zeros(m + 1)
    
    # Update Theta
    for i in range(m + 1):
        uk = np.sum(y[s == i])
        var_theta = 1. / (1./d**2. + nk[i]/sigma**2.)
        mu_theta = var_theta * (c/d**2. + uk/sigma**2.)
        theta[i] = stats.norm.rvs(mu_theta, np.sqrt(var_theta))
        f[i] = stats.norm.pdf(theta_star[i], mu_theta, np.sqrt(var_theta))
    f_theta_star = np.prod(f)
        
    return theta, f_theta_star
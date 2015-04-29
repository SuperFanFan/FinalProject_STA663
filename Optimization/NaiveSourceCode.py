
# Naive code - 04/02/15

from __future__ import division
import numpy as np
import scipy.stats as stats

#### Sampling functions

# (1) Update S_n
def update_Sn_naive(y, n, m, Ptran, theta, s):
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
    for i in range(m):
        if Ptran[i, i] <= 0.0:
            return "Error - transition probabilities should be within range 0 to 1."
    
    # read the current s values
    s_new = s
    sigma = np.sqrt(3.0)
    
    # define quantities
    F_lag, F = np.zeros((n, m + 1)), np.zeros((n, m + 1))                                  
    F_lag[0, 0], F[0, 0] = 1, 1
    pmfs = np.zeros(m + 1)
    
    for i in range(1, n):
        for j in range(m + 1):
            for k in range(m + 1):
                F_lag[i,j] += Ptran[k,j] * F[i - 1,k]
        for j in range(m + 1):
            F[i,j] = F_lag[i,j] * stats.norm.pdf(y[i], loc = theta[j], scale = sigma)
        row_sum = np.sum(F[i,:])
        for j in range(m + 1):
            F[i,j] = F[i,j] / row_sum
            
    # Sampling s_t
    for k in range(n - 2, 0, -1): # omit update s_n and s_1 because of their degeneracy
        for r in range(m + 1):
            pmfs[r] = F[k,r] * Ptran[r,s_new[k + 1]]
        row_sum = np.sum(pmfs)
        for r in range(m + 1):
            pmfs[r] = pmfs[r] / row_sum
        s_new[k] = np.random.choice(np.arange(m + 1), p = pmfs)
        
    return F_lag, F, s_new

# (2) Update P
def update_P_naive(a, b, n, m, s, Ptran_star):
    """ Update transition matrix P 
        Args: a,b - prior beta parameters
              n - number of observations
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
    f_Ptran_star = 1.0
    
    # number of same states
    for i in range(m + 1):
        for j in range(n):
            if s[j] == i:
                nk[i] += 1.0 
    nii = nk - 1
    
    # update P
    for j in range(m):
        Ptran[j, j] = stats.beta.rvs(a + nii[j], b + 1)
        Ptran[j, j + 1] = 1.0 - Ptran[j, j]
        f[j] = stats.beta.pdf(Ptran_star[j, j], a + nii[j], b + 1)
        f_Ptran_star = f_Ptran_star * f[j]
        
    return nk, Ptran, f_Ptran_star

# (3) Update Theta - Gaussian Model
def update_Theta_naive(c, d, m, y, s, nk, theta_star):
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
    n = len(y)
    sigma = np.sqrt(3.0)
    theta = np.repeat(2.0, m + 1)
    f = np.zeros(m + 1)
    f_theta_star = 1.0
    
    # Update Theta
    for i in range(m + 1):
        uk = 0.0
        for j in range(n):
            if s[j] == i:
                uk += y[j] 
        var_theta = 1. / (1./d**2. + nk[i]/sigma**2.)
        mu_theta = var_theta * (c/d**2. + uk/sigma**2.)
        theta[i] = stats.norm.rvs(mu_theta, np.sqrt(var_theta))
        f[i] = stats.norm.pdf(theta_star[i], mu_theta, np.sqrt(var_theta))
        f_theta_star = f_theta_star * f[i]
        
    return theta, f_theta_star

### Monte Carlo EM to compute the maximum likelihood estimates
def mcem_update_naive(y, m, tol):
    """ Finding the MLE by Monte Carlo Expectation Maximization
        Args: y - vector of observations
              m - number of change points
              tol - tolerance limit to claim convergence
        Return: theta - MLE of model parameters
                Ptran - MLE of transition probabilities """
    
    # Initial parameters
    n = len(y)
    theta = np.repeat(2.0, m + 1)
    sigma = np.sqrt(3.0)
    s = np.zeros(n - m)
    for i in range(1, m + 1):
        s = np.append(s, i)
    F_lag, F = np.zeros((n, m + 1)), np.zeros((n, m + 1))                                  
    F_lag[0, 0], F[0, 0] = 1, 1
    Ptran = np.zeros((m + 1, m + 1))
    Ptran[-1, -1] = 1
    for j in range(m):
        Ptran[j, j] = 0.5
        Ptran[j, j + 1] = 1 - Ptran[j, j]
    
    # Number of samples drawn to evaluate Q function
    N_seq = np.repeat(np.array([1, 50, 50, 50, 100, 100, 300, 300, 300, 300]), 10)
    nsim = N_seq.shape[0]
    
    # Check Convergence - start
    theta_current, Ptran_current = theta, np.delete(np.diag(Ptran), -1)
    
    # MECM updates
    np.random.seed(1234)
    for i in range(nsim):
        
        # E step
        ll_y, ll_s = 0.0, 0.0
        nk, uk = np.zeros(m + 1), np.zeros(m + 1)
        nk_sum, uk_sum = np.zeros(m + 1), np.zeros(m + 1)
        for j in range(N_seq[i]):
            F_lag, F, s = update_Sn_naive(y, n, m, Ptran, theta, s)
            for k in range(m + 1):
                for r in range(n):
                    if s[r] == k:
                        nk[k] += 1.0
                        uk[k] += y[r]
            for t in range(n):
                index = s[t]
                ll_y += np.log(stats.norm.pdf(y[t], loc = theta[index.astype(int)], scale = sigma))
            
            mi = 0
            for t in range(n - 1):
                index = s[range(t, t+2)]
                if index[1] == index[0]:
                    ll_s += np.log(Ptran[mi, mi])
                elif index[1] != index[0]:
                    ll_s += np.log(Ptran[mi, mi + 1])
                    mi += 1
            nk_sum += nk
            uk_sum += uk
        Q_hat = (ll_y + ll_s) / float(N_seq[i])
    
        # M step
        theta = uk_sum / nk_sum
        for r in range(m):
            Ptran[r, r] = nk_sum[r] / (nk_sum[r] + N_seq[r])
            Ptran[r, r + 1] = 1.0 - Ptran[r, r]
        
        # Check Convergence - stop
        if(N_seq[i] >= 100):
            converge_theta = np.linalg.norm(theta - theta_current)
            converge_Ptran = np.linalg.norm(np.delete(np.diag(Ptran), -1) - Ptran_current)
            theta_current, Ptran_current = theta, np.delete(np.diag(Ptran), -1)
            if(converge_theta < tol and converge_Ptran < tol):
                print "Convergence Reached"
                return theta, Ptran
            elif(i == nsim - 1):
                print "Maximum Iteration Reached"
                return theta, Ptran
            
        # Loop control
        if(i % 20 == 0):
            print theta
            
### Execution Functions
def model_fit_naive(y, m, vsim, burn, theta_star, Ptran_star, c, d):
    """ Update model parameters Theta 
        Args: y - vector of observations
              m - number of change points
              vsim - number of simulations in total
              burn - length of burn-in period
              theta_star - MLE of model parameters
              Ptran_star - MLE of transition probabilities
              c,d - prior normal parameters
        Return: Theta_p - posterior samples of model parameters
                F_lag_sum, F_sum - posterior sums of marginal probs for latent states (need to average across simulations)
                log_f_y - maximized log-likelihood for a given model
                log_m - marginal likelihood for a given model """
    
    # priors
    n = len(y)
    a, b = 8., 0.1

    # Inits - parameters
    theta = np.repeat(2.0, m + 1)
    sigma = np.sqrt(3.0)
    s = np.zeros(n - m)
    for i in range(1, m + 1):
        s = np.append(s, i)

    # Inits - useful quantities                           
    Ptran = np.zeros((m + 1, m + 1))
    Ptran[-1, -1] = 1
    for j in range(m):
        Ptran[j, j] = 0.5
        Ptran[j, j+1] = 1 - Ptran[j, j]

    # Store results
    Theta_p = np.zeros((vsim, m + 1))
    F_lag_sum, F_sum = np.zeros((n, m + 1)), np.zeros((n, m + 1))       
    F_theta_star, F_Ptran_star = np.zeros(vsim), np.zeros(vsim)

    np.random.seed(1234)
    # Gibbs steps
    for v in range(vsim):
        
        # Update S_n
        F_lag, F, s = update_Sn_naive(y, n, m, Ptran, theta, s)
    
        # Update P
        nk, Ptran, f_Ptran_star = update_P_naive(a, b, n, m, s, Ptran_star)
    
        # Update Theta
        theta, f_theta_star = update_Theta_naive(c, d, m, y, s, nk, theta_star)

        # Save results
        Theta_p[v,:] = theta
        F_theta_star[v] = f_theta_star
        F_Ptran_star[v] = f_Ptran_star
    
        if(v >= burn):
            F_lag_sum = F_lag_sum + F_lag
            F_sum = F_sum + F
        if(v % 2000 == 0):
            print v
            
    ### compute marginal likelihood
    f_y = np.zeros((n, m + 1))
    log_f_y, log_pi_theta_star, log_pi_Ptran_star = 0.0, 0.0, 0.0
    for i in range(m + 1):
        for j in range(n):
            f_y[j,i] = stats.norm.pdf(y[j], loc = theta_star[i], scale = sigma)
    F_lag_star, F_star, s_star = update_Sn_naive(y, n, m, Ptran_star, theta_star, s)
    for j in range(n):
        log_f_y += np.log(np.sum(f_y[i,:] * F_lag_star[i,:]))
    for k in range(m + 1):
        log_pi_theta_star += np.log(stats.norm.pdf(theta_star[k], loc = c, scale = d))
    for r in range(m):
        log_pi_Ptran_star += np.log(stats.beta.pdf(Ptran_star[r,r], a, b))
    log_f_theta_star = np.log(np.mean(F_theta_star[burn:]))
    log_f_Ptran_star = np.log(np.mean(F_Ptran_star[burn:]))
    log_m = log_f_y + log_pi_theta_star + log_pi_Ptran_star - log_f_theta_star - log_f_Ptran_star
    
    return Theta_p, F_lag_sum, F_sum, log_f_y, log_m          
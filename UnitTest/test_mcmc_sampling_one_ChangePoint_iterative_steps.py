
### configuration
import numpy as np
import scipy.stats as stats
from numpy.testing import assert_almost_equal
import imp
mdl = imp.load_source('OptimizedSourceCode', '/home/bitnami/FinalProject_STA663/Optimization/OptimizedSourceCode.py')

# Specify test data
n = 150
m = 1 # number of change points
sigma = np.sqrt(3.0)
tau_true = 50
tau_true_rep = np.array([tau_true, n - tau_true])
s_true = np.repeat(range(0, m+1), tau_true_rep)
theta_true = np.array([1., 3.])
y = np.random.normal(theta_true[s_true], sigma, size = n)

a, b = 8., 0.1
c, d = 2., 100.

# Inits - parameters
theta = np.array([2., 4.])
theta_star = theta_true
sigma = np.sqrt(3.0)
s = np.repeat(np.array([0., 1.]), np.array([70, 80]))

# Inits - useful quantities                           
Ptran = np.zeros((m + 1, m + 1))
Ptran[-1, -1] = 1
for j in range(m):
    Ptran[j, j] = 0.875
    Ptran[j, j+1] = 1 - Ptran[j, j]
Ptran_star = Ptran

# THE UNIT TESTS - iterative steps
def test_OneStepProbs_SumToOne_iterative():
    vsim = 10
    s_new, Ptran_new, theta_new = s, Ptran, theta
    for v in range(vsim):
        F_lag, F, s_new = mdl.update_Sn_optimized(y, n, m, Ptran_new, theta_new, s_new)
        nk, Ptran_new, f_Ptran_star = mdl.update_P_optimized(a, b, n, m, s_new, Ptran_star)
        theta_new, f_theta_star = mdl.update_Theta_optimized(c, d, m, y, s, nk, theta_star)
    assert_almost_equal(F_lag.sum(axis = 1), 1.)

def test_PosteriorProbs_SumToOne_iterative():
    vsim = 10
    s_new, Ptran_new, theta_new = s, Ptran, theta
    for v in range(vsim):
        F_lag, F, s_new = mdl.update_Sn_optimized(y, n, m, Ptran_new, theta_new, s_new)
        nk, Ptran_new, f_Ptran_star = mdl.update_P_optimized(a, b, n, m, s_new, Ptran_star)
        theta_new, f_theta_star = mdl.update_Theta_optimized(c, d, m, y, s, nk, theta_star)
    assert_almost_equal(F.sum(axis = 1), 1.)

def test_ordering_LatentStates_iterative():
    vsim = 10
    s_new, Ptran_new, theta_new = s, Ptran, theta
    for v in range(vsim):
        F_lag, F, s_new = mdl.update_Sn_optimized(y, n, m, Ptran_new, theta_new, s_new)
        nk, Ptran_new, f_Ptran_star = mdl.update_P_optimized(a, b, n, m, s_new, Ptran_star)
        theta_new, f_theta_star = mdl.update_Theta_optimized(c, d, m, y, s, nk, theta_star)
    order = np.zeros(s.shape[0] - 1)
    for i in range(order.shape[0]):
        order[i] = s[i + 1] - s[i]
    assert all(order >= 0.0) and sum(order) == m

def test_state_counts_range_iterative():
    vsim = 10
    s_new, Ptran_new, theta_new = s, Ptran, theta
    for v in range(vsim):
        F_lag, F, s_new = mdl.update_Sn_optimized(y, n, m, Ptran_new, theta_new, s_new)
        nk, Ptran_new, f_Ptran_star = mdl.update_P_optimized(a, b, n, m, s_new, Ptran_star)
        theta_new, f_theta_star = mdl.update_Theta_optimized(c, d, m, y, s, nk, theta_star)
    assert all(nk > 0.0) and all(nk < n) and np.sum(nk) == n
    
def test_transition_probs_support_iterative():
    vsim = 10
    s_new, Ptran_new, theta_new = s, Ptran, theta
    for v in range(vsim):
        F_lag, F, s_new = mdl.update_Sn_optimized(y, n, m, Ptran_new, theta_new, s_new)
        nk, Ptran_new, f_Ptran_star = mdl.update_P_optimized(a, b, n, m, s_new, Ptran_star)
        theta_new, f_theta_star = mdl.update_Theta_optimized(c, d, m, y, s, nk, theta_star)
    assert all(np.delete(np.diag(Ptran), -1) > 0) and all(np.delete(np.diag(Ptran), -1) < 1)
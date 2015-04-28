
### configuration
import numpy as np
from numpy.testing import assert_almost_equal
from mcmc_sampling import update_Sn, update_P, update_Theta

# Specify test data
y = np.loadtxt("data_test")
theta_true = np.array([1., 3.])
n, m = len(y), 1
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

# THE UNIT TESTS - single steps
def test_OneStepProbs_SumToOne_single():
    F_lag, F, s_new = update_Sn(y, n, m, Ptran, theta, s)
    assert_almost_equal(F_lag.sum(axis = 1), 1.)

def test_PosteriorProbs_SumToOne_single():
    F_lag, F, s_new = update_Sn(y, n, m, Ptran, theta, s)
    assert_almost_equal(F.sum(axis = 1), 1.)

def test_ordering_LatentStates_single():
    F_lag, F, s_new = update_Sn(y, n, m, Ptran, theta, s)
    order = np.zeros(s.shape[0] - 1)
    for i in range(order.shape[0]):
        order[i] = s[i + 1] - s[i]
    assert all(order >= 0.0) and sum(order) == m

def test_invalid_transition_probs_input():
    Ptran = np.zeros((m + 1, m + 1))
    Ptran[-1, -1] = 1
    for j in range(m):
        Ptran[j, j] = -0.01
        Ptran[j, j+1] = 1 - Ptran[j, j]
    assert update_Sn(y, n, m, Ptran, theta, s) == "Error - transition probabilities should be within range 0 to 1."
    
def test_state_counts_range_single():
    nk, Ptran, f_Ptran_star = update_P(a, b, m, s, Ptran_star)    
    assert all(nk > 0.0) and all(nk < n)
    
def test_transition_probs_support_single():
    nk, Ptran, f_Ptran_star = update_P(a, b, m, s, Ptran_star)
    assert all(np.delete(np.diag(Ptran), -1) > 0) and all(np.delete(np.diag(Ptran), -1) < 1)
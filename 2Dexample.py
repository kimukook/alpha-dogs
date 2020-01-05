from alphaDOGS import *
import numpy as np

# options setting
options = AlphaDOGSOptions()
options.set_option('Constant surrogate', True)
options.set_option('Scipy solver', True)

options.set_option('Initial mesh size', 4)
options.set_option('Number of mesh refinement', 4)
options.set_option('Initial time length', 2.0)
options.set_option('Incremental time step', 2.0)

options.set_option('Initial sites known', False)

options.set_option('Global minimizer known', True)
options.set_option('Target value', -1.6759 * 2)
options.set_option('Global minimizer', np.array([[0.8419], [0.8419]]))

options.set_option('Function evaluation cheap', True)
options.set_option('Plot saver', True)
options.set_option('Candidate distance summary', True)
options.set_option('Candidate objective value summary', True)
options.set_option('Iteration summary', True)
options.set_option('Optimization summary', True)

# input parameters setting
bnds = np.hstack((np.zeros((2, 1)), np.ones((2, 1))))


def func_eval(x, t):
    sigma0 = .3
    t = int(t)
    return np.array(np.mean(np.sum(-2 * x * np.sin(np.sqrt(500 * x))) + sigma0 * np.random.normal(0, 1, t)))


def sigma_eval(x, t):
    sigma0 = .3
    return sigma0 / np.sqrt(int(t))


def truth_eval(x, t):
    return np.array(np.sum(-2 * x * np.sin(np.sqrt(500 * x))))


n = 2
Ain = np.concatenate((np.identity(n), -np.identity(n)), axis=0)
Bin = np.concatenate((np.ones((n, 1)), np.zeros((n, 1))), axis=0)

adogs = AlphaDOGS(bnds, func_eval, sigma_eval, truth_eval, options, Ain, Bin)
adogs.alphadogs_optimizer()

import numpy as np
import os
from scipy import io
import shutil
from alphaDOGS import *
from lorenz import lorenz
from lorenz import uq
from dogs import Utils


def lorenz_func_eval(x, t):
    h = 0.001
    ROOT = os.getcwd()
    file_name = 'points.mat'
    file_path = os.path.join(ROOT, file_name)
    if not os.path.exists(file_path):
        # First function evaluation, create data points and save it in file
        data = {'xE': x}
        index = 0
        type = 'identify'

        io.savemat(file_path, data)

    else:
        data = io.loadmat(file_path)
        xE = data['xE']
        val, idx, _ = Utils.mindis(x, xE)
        if val < 1e-6:
            # x has already been evaluated
            index = idx
            type = 'additional'
        else:
            index = xE.shape[1]
            type = 'identify'

            xE = np.hstack((xE, x))
            data = {'xE': xE}

            io.savemat(file_path, data)

    l = lorenz.Lorenz(x, t, h, 23.57, index, type)
    l.lorenz_eval()

    # J, sig = l.main()
    # return J, sig

    return l.J


def lorenz_noise_eval(x, t):
    h = 0.001
    ROOT = os.getcwd()
    file_name = 'points.mat'
    file_path = os.path.join(ROOT, file_name)

    data = io.loadmat(file_path)
    xE = data['xE']
    index = Utils.mindis(x, xE)[1]

    points_eval_data_path = os.path.join(ROOT, 'AllPoints/pt' + str(index) + '.mat')
    points_eval_data = io.loadmat(points_eval_data_path)
    zs = points_eval_data['zs'][0]

    length = int(min((t / h), 1000))
    xx = uq.data_moving_average(zs, length)
    sig = np.sqrt(uq.stationary_statistical_learning_reduced(xx, 18)[0])
    return sig


if __name__ == '__main__':
    ROOT = os.getcwd()
    data_folder = 'AllPoints'
    path = os.path.join(ROOT, data_folder)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(os.path.join(ROOT, data_folder))

    points_path = os.path.join(ROOT, 'points.mat')
    if os.path.exists(points_path):
        os.remove(points_path)

    bnds = np.hstack((25 * np.ones((1, 1)), 29 * np.ones((1, 1))))
    options = AlphaDOGSOptions()
    options.set_option('Constant surrogate', True)
    options.set_option('Scipy solver', True)
    options.set_option('Constant L', 3.)
    options.set_option('Constant K', 15.)
    x = np.array([[25., 29.]])

    options.set_option('Initial sites known', True)
    options.set_option('Initial sites', x)

    options.set_option('Global minimizer known', True)
    options.set_option('Target value', 0.)
    options.set_option('Global minimizer', np.array([[28.]]))

    options.set_option('Initial mesh size', 4)
    options.set_option('Number of mesh refinement', 2)
    options.set_option('Initial time length', 100.0)
    options.set_option('Incremental time step', 50.0)

    options.set_option('Function evaluation cheap', False)
    options.set_option('Plot saver', False)
    options.set_option('Candidate distance summary', True)
    options.set_option('Candidate objective value summary', True)
    options.set_option('Iteration summary', True)
    options.set_option('Optimization summary', True)

    opt = AlphaDOGS(bnds, lorenz_func_eval, lorenz_noise_eval, lorenz_func_eval, options)
    xmin = opt.alphadogs_optimizer()


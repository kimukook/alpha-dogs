'''
//
//  alphaDOGS.py
//  Alpha-DOGS: Optimize the time-averaged statistics using Delaunay-based Derivative-free optimziation
//              via Global Surrogates
//
//  Created by Muhan Zhao on 5/12/19.
//  Copyright © 2019 Muhan Zhao. All rights reserved.
//
'''

import  scipy
import  os
import  inspect
import  shutil
import  numpy               as np
from    functools           import partial
from    scipy               import io
from    dogs                import Utils
from    dogs                import interpolation
from    dogs                import cartesian_grid
from    dogs                import constantK
from    dogs                import plot


class OptionsClass:
    """
    Options Class
    """

    def __init__(self):
        self.options = None
        self.solverName = 'None'

    def set_option(self, key, value):
        try:
            if type(value) is self.options[key][2]:
                self.options[key][0] = value
            else:
                print(f"The type of value for the keyword '{key}' should be '{self.options[key][2]}'.")
        except:
            raise ValueError('Incorrect option keyword or type: ' + key)

    def get_option(self, key):
        try:
            value = self.options[key][0]
            return value
        except:
            raise ValueError('Incorrect option keyword: ' + key)

    def reset_options(self, key):
        try:
            self.options[key] = self.options[key][1]
        except:
            raise ValueError('Incorrect option keyword: ' + key)


class AlphaDOGSOptions(OptionsClass):
    """
    Options class for alphaDOGS
    """
    def __init__(self):
        OptionsClass.__init__(self)
        self.setup()
        self.solverName = 'AlphaDOGS'

    def setup(self):
        self.options = {
            # [Current value, default value, type]
            'Objective function name'           : [None, None, str],
            'Constant surrogate'                : [False, False, bool],
            'Constant K'                        : [6.0, 6.0, float],
            'Constant L'                        : [2.0, 2.0, float],

            'Adaptive surrogate'                : [False, False, bool],
            'Target value'                      : [None, None, float],

            'Scipy solver'                      : [False, False, bool],
            'Snopt solver'                      : [False, False, bool],

            'Initial mesh size'                 : [3, 3, int],
            'Number of mesh refinement'         : [8, 8, int],

            'Initial sites known'               : [False, False, bool],
            'Initial sites'                     : [None, None, np.ndarray],
            'Initial function values'           : [None, None, np.ndarray],
            'Initial funtion noise'             : [None, None, np.ndarray],
            'Initial time length'               : [10.0, 10.0, float],
            'Incremental time step'             : [10.0, 10.0, float],
            'Maximum evaluation times'          : [20.0, 20.0, float],

            'Global minimizer known'            : [False, False, bool],
            'Global minimizer'                  : [None, None, np.ndarray],

            'Function evaluation cheap'         : [True, True, bool],
            'Function prior file path'          : [None, None, str],
            'Plot saver'                        : [True, True, bool],
            'Figure format'                     : ['png', 'png', str],
            'Candidate distance summary'        : [False, False, bool],
            'Candidate objective value summary' : [False, False, bool],
            'Iteration summary'                 : [False, False, bool],
            'Optimization summary'              : [False, False, bool]
        }


class AlphaDOGS:
    def __init__(self, bounds, func_eval, noise_eval, truth_eval, options, A=None, b=None):
        """
        Alpha DOGS is an efficient optimization algorithm for time-averaged statistics.
        :param bounds           :   The physical bounds of the input for the test problem,
                                    e.g. lower bound = [0, 0, 0] while upper bound = [1, 1, 1] then
                                    bnds = np.hstack((np.zeros((3,1)), np.ones((3,1))))

        :param func_eval        :   The objective function

        :param noise_eval       :   The function for noise evaluation

        :param truth_eval       :   The function to evaluate the truth values

        :param A                :   The linear constraints of parameter space

        :param b                :   The linear constraints of parameter space

        :param options          :   The options for alphaDOGS
        """
        # n: The dimension of input parameter space
        self.n = bounds.shape[0]
        # physical lb & ub: Normalize the physical bounds
        self.physical_lb = bounds[:, 0].reshape(-1, 1)
        self.physical_ub = bounds[:, 1].reshape(-1, 1)
        # lb & ub: The normalized search bounds
        self.lb = np.zeros((self.n, 1))
        self.ub = np.ones((self.n, 1))

        # ms: The mesh size for each iteration. This is the initial definition.
        self.initial_mesh_size = options.get_option('Initial mesh size')
        self.ms = 2 ** self.initial_mesh_size
        self.num_mesh_refine = options.get_option('Number of mesh refinement')
        self.max_mesh_size = 2 ** (self.initial_mesh_size + self.num_mesh_refine)

        self.iter = 0

        # Define the surrogate model
        if options.get_option('Constant surrogate') and options.get_option('Adaptive surrogate'):
            raise ValueError('Constant and Adaptive surrogate both activated. Set one to False then rerun code.')
        elif not options.get_option('Constant surrogate') and not options.get_option('Adaptive surrogate'):
            raise ValueError('Constant and Adaptive surrogate both inactivated. Set one to True then rerun code.')
        elif options.get_option('Constant surrogate'):
            self.surrogate_type = 'c'
        elif options.get_option('Adaptive surrogate'):
            self.surrogate_type = 'a'
        else:
            pass

        if self.surrogate_type == 'c':
            # Define the parameters for discrete and continuous constant search function
            self.L = options.get_option('Constant L')
            self.L0 = self.L
            self.K = options.get_option('Constant K')

        elif self.surrogate_type == 'a':
            self.y0 = options.get_option('Target value')

        # Define the linear constraints, Ax <= b. if A and b are None type, set them to be the box domain constraints.
        if (A and b) is None:
            self.Ain = np.concatenate((np.identity(self.n), -np.identity(self.n)), axis=0)
            self.Bin = np.concatenate((np.ones((self.n, 1)), np.zeros((self.n, 1))), axis=0)
        else:
            pass

        # Define the statistics for time length
        self.T0 = options.get_option('Initial time length')
        self.dt = options.get_option('Incremental time step')
        self.eval_times = options.get_option('Maximum evaluation times')
        self.Tmax = self.T0 + self.eval_times * self.dt
        self.Tmax_reached = None

        if options.get_option('Scipy solver') and options.get_option('Snopt solver'):
            raise ValueError('More than one optimization solver specified!')
        elif not options.get_option('Scipy solver') and not options.get_option('Snopt solver'):
            raise ValueError('No optimization solver specified!')
        elif options.get_option('Scipy solver'):
            self.solver_type = 'scipy'
        elif options.get_option('Snopt solver'):
            self.solver_type = 'snopy'
        else:
            pass

        # Initialize the function evaluation, noise sigma evaluation and truth function evaluation.
        # capsulate those three functions with physical bounds
        self.func_eval = partial(Utils.fun_eval, func_eval, self.physical_lb, self.physical_ub)
        self.sigma_eval = partial(Utils.fun_eval, noise_eval, self.physical_lb, self.physical_ub)
        self.truth_eval = partial(Utils.fun_eval, truth_eval, self.physical_lb, self.physical_ub)

        # Define the global optimum and its values
        if options.get_option('Global minimizer known'):
            self.xmin = Utils.normalize_bounds(options.get_option('Global minimizer'), self.physical_lb,
                                               self.physical_ub)
            self.y0 = options.get_option('Target value')
        else:
            self.xmin = None
            self.y0 = None

        # Define the iteration type for each sampling iteration.
        self.iter_type = None

        # Define the initial sites and their function evaluations
        if options.get_option('Initial sites known'):
            physical_initial_sites = options.get_option('Initial sites')
            # Normalize the bound
            self.xE = Utils.normalize_bounds(physical_initial_sites, self.physical_lb, self.physical_ub)
        else:
            self.xE = Utils.random_initial(self.n, 2 * self.n, self.ms, self.Ain, self.Bin, self.xU)

        if options.get_option('Initial function values') is not None:
            self.yE = options.get_option('Initial function values')
            self.T = self.T0 * np.ones(self.xE.shape[1], dtype=float)
            self.sigma = options.get_option('Initial funtion noise')
        else:
            # Compute the function values, time length, and noise level at initial sites
            self.yE = np.zeros(self.xE.shape[1])
            self.T = self.T0 * np.ones(self.xE.shape[1], dtype=float)
            self.sigma = np.zeros(self.xE.shape[1])

            for i in range(2 * self.n):
                self.yE[i] = self.func_eval(self.xE[:, i], self.T[i])
                self.sigma[i] = self.sigma_eval(self.xE[:, i], self.T0)

        self.iteration_summary_matrix = {}
        # Define the initial support points
        self.xU = Utils.bounds(self.lb, self.ub, self.n)
        self.xU = Utils.unique_support_points(self.xU, self.xE)
        self.yu = None

        self.K0 = np.ptp(self.yE, axis=0)
        # Define the interpolation
        self.inter_par = None
        self.yp = None

        # Define the discrete search function
        self.sd = None

        # Define the minimizer of continuous search function, parameter to be evaluated, xc & yc.
        self.xc = None
        self.yc = None

        # Define the minimizer of discrete search function
        self.xd = None
        self.yd = None
        self.index_min_yd = None

        # Although it is stochastic, just display the behavior instead of the actual data to show the trend.
        # Define the name of directory to store the figures
        self.algorithm_name = options.solverName

        #  ==== Plot section here ====
        self.plot = plot.PlotClass()

        # Define the parameter to save image or no, and what format to save for images
        self.save_fig = options.get_option('Plot saver')
        self.fig_format = options.get_option('Figure format')

        # Generate the folder path
        self.func_path    = options.get_option('Objective function name')  # Function folder, e.g. Lorenz
        self.current_path = None           # Directory path
        self.plot_folder  = None           # Plot storage folder
        self.folder_path_generator()       # Determine the folder paths above
        self.func_name = options.get_option('Objective function name')     # Define the name of the function called.

        # All the prior function values should be provided by user, instead of solver calculating those values.
        # Generate the plot of test function
        self.func_initial_prior = True
        # The objective function values are stored in func_prior_yE
        # for the future iteration, just change the point of func_prior_sigma that is close to the evaluated point
        self.func_prior_xE_2DX = None
        self.func_prior_xE_2DY = None
        self.func_prior_xE_2DZ = None

        if options.get_option('Function prior file path') is not None:
            self.func_prior_file_name = options.get_option('Function prior file path')
            data = io.loadmat(self.func_prior_file_name)
            # The prior data must have the following keyword
            self.func_prior_xE = data['x']
            self.func_prior_yE = data['y'][0]
            self.func_prior_sigma = data['sigma'][0] * 2
            if self.n == 1:
                # Define the range of plot in y-axis
                self.plot_ylow = np.min(self.func_prior_yE) * 2
                self.plot_yupp = np.max(self.func_prior_yE) * 2

        else:
            self.func_prior_file_name = None
            if options.get_option('Function evaluation cheap') and self.n < 3:
                if self.n == 1:
                    self.plot.initial_calc1D(self)
                elif self.n == 2:
                    self.plot.initial_calc2D(self)
                else:
                    pass
                if self.n == 1:
                    # Define the range of plot in y-axis
                    self.plot_ylow = np.min(self.func_prior_yE) * 2
                    self.plot_yupp = np.max(self.func_prior_yE) * 2

            else:
                self.func_initial_prior = False
                self.func_prior_xE = None
                self.func_prior_yE = None
                # Define the range of plot in y-axis
                self.plot_ylow = np.min(self.yE) * 2
                self.plot_yupp = np.max(self.yE) * 2

                print('Function evaluation is expensive and no file that contains prior function values information '
                      'is found.')
                if self.n >= 3:
                    print('Parameter space dimenion > 3, the plot for each iteration is unavailable.')

        if self.save_fig and self.func_initial_prior:
            if self.n == 1:
                self.initial_plot = self.plot.initial_plot1D
                self.iter_plot = self.plot.plot1D
            elif self.n == 2:
                self.initial_plot = self.plot.initial_plot2D
                self.iter_plot = self.plot.plot2D
            else:
                self.initial_plot = None
                print("Parameter space higher than 2, set 'Plot saver' to be False.")
            self.initial_plot(self)

        self.iter_summary = options.get_option('Iteration summary')
        self.optm_summary = options.get_option('Optimization summary')

    def folder_path_generator(self):
        '''
        Determine the root path and generate the plot folder.
        :return:
        '''
        self.current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.plot_folder = os.path.join(self.current_path, 'plot')
        if os.path.exists(self.plot_folder):
            shutil.rmtree(self.plot_folder)
        os.makedirs(self.plot_folder)

    def discrete_function_evaluation(self, x):
        '''
        Evaluate the discrete search function value at support poitns.
        :param x:
        :return:
        '''
        return (self.inter_par.inter_val(x) - min(self.yp))[0] / Utils.mindis(x, self.xE)[0]

    def alphadogs_optimizer(self):
        '''
        Main optimization function.
        :return:
        '''
        for kk in range(self.num_mesh_refine):
            for k in range(20):
                if self.surrogate_type == 'c':
                    self.constant_surrogate_solver()
                else:
                    pass

                if self.iter_type == 'refine' or self.Tmax_reached:
                    self.iter_type = None
                    break
                else:
                    pass

        if self.optm_summary:
            self.plot.summary_plot(self)
        self.plot.result_saver(self)

    def constant_surrogate_solver(self):
        '''
        Surrogate solver using constant K algorithm iteratively perform additional sampling/ mesh refinement/
        identifying sampling.
        :return:
        '''
        self.iter += 1

        self.K0 = np.ptp(self.yE, axis=0)
        self.inter_par = interpolation.InterParams(self.xE)
        self.yp = self.inter_par.regressionparameterization(self.yE, self.sigma)
        # Calculate the discrete function.

        self.sd = np.amin((self.yp, 2 * self.yE - self.yp), 0) - self.L * self.sigma
        ind_min = np.argmin(self.yp)

        self.yd = np.amin(self.sd)
        self.index_min_yd = np.argmin(self.sd)
        self.xd = np.copy(self.xE[:, self.index_min_yd].reshape(-1, 1))

        self.yu = np.zeros(self.xU.shape[1])
        if self.xU.shape[1] != 0:
            for ii in range(self.xU.shape[1]):
                self.yu[ii] = self.discrete_function_evaluation(self.xU[:, ii])
        else:
            pass

        if self.xU.shape[1] != 0 and np.amin(self.yu) < np.min(self.yp):
            ind = np.argmin(self.yu)
            self.xc = np.copy(self.xU[:, ind].reshape(-1, 1))
            self.yc = -np.inf
            self.xU = scipy.delete(self.xU, ind, 1)
        else:
            while 1:
                xs, ind_min = cartesian_grid.add_sup(self.xE, self.xU, ind_min)
                xc, self.yc, result = constantK.tringulation_search_bound_constantK(self.inter_par, xs,
                                                                                    self.K * self.K0, ind_min)
                if self.inter_par.inter_val(xc) < min(self.yp):
                    self.xc = np.round(xc * self.ms) / self.ms
                    break
                else:
                    self.xc = np.round(xc * self.ms) / self.ms
                    if Utils.mindis(self.xc, self.xE)[0] < 1e-6:
                        break
                    self.xE, self.xU, success, newadd = cartesian_grid.points_neighbers_find(self.xc, self.xE,
                                                                                                      self.xU, self.Bin,
                                                                                                      self.Ain)
                    if success == 1:
                        break
                    else:
                        self.yu = np.hstack((self.yu, self.discrete_function_evaluation(self.xc)))
                if self.xU.shape[1] != 0 and np.amin(self.yu) < np.min(self.yp):
                    xc_discrete = self.discrete_function_evaluation(self.xc)
                    if np.amin(self.yu) < xc_discrete:
                        ind = np.argmin(self.yu)
                        self.xc = np.copy(self.xU[:, ind].reshape(-1, 1))
                        self.yc = -np.inf
                        self.xU = scipy.delete(self.xU, ind, 1)

        if self.yd < self.yc:
            self.iter_type = 'sdmin'
            self.T[self.index_min_yd] += self.dt
            self.yE[self.index_min_yd] = self.func_eval(self.xd, self.T[self.index_min_yd])
            self.sigma[self.index_min_yd] = self.sigma_eval(self.xd, self.T[self.index_min_yd])
            self.iteration_summary_matrix[self.iter] = {'x': self.xd.tolist(), 'y': self.yE[self.index_min_yd],
                                                        'T': self.T[self.index_min_yd],
                                                        'sig': self.sigma[self.index_min_yd]}
            if self.T[self.index_min_yd] >= self.Tmax:
                # Function evaluation has reached the limit, refine the mesh
                # TODO refine the mesh here??????
                # tODO or, directly goes to eval xc min?
                # TODO cuz xcmin is the initial evaluation, not expensive
                self.Tmax_reached = True
            else:
                self.Tmax_reached = False

        else:
            if Utils.mindis(self.xc, self.xE)[0] < 1e-6:
                self.iter_type = 'refine'
                self.K *= 2
                self.ms *= 2
                self.L += self.L0

            else:
                self.iter_type = 'scmin'
                self.T = np.hstack((self.T, self.T0))
                self.xE = np.hstack((self.xE, self.xc))
                self.yE = np.hstack((self.yE, self.func_eval(self.xc, self.T0)))
                self.sigma = np.hstack((self.sigma, self.sigma_eval(self.xc, self.T0)))
                self.iteration_summary_matrix[self.iter] = {'x': self.xc.tolist(), 'y': self.yE[-1],
                                                            'T': self.T0,
                                                            'sig': self.sigma[-1]}

        if self.save_fig:
            self.iter_plot(self)
        if self.iter_summary:
            self.plot.summary_display(self)

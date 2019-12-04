import  os
import  matplotlib.pyplot   as plt
import  numpy               as np
from    dogs                import Utils
import  scipy.io            as io
from    matplotlib.ticker   import PercentFormatter
from    scipy.spatial       import Delaunay
from    mpl_toolkits.mplot3d import axes3d, Axes3D


class PlotClass:
    def __init__(self):
        self.size = 200

    def fig_saver(self, name, adogs):
        if adogs.save_fig:
            name = os.path.join(adogs.plot_folder, name) + '.' + adogs.fig_format
            plt.savefig(name, format=adogs.fig_format, dpi=200)
        else:
            pass

    def initial_calc1D(self, adogs):
        """
        Plot the initial objective function together with the uncertainty of 1D parameter space.
        :return:
        """

        adogs.func_prior_xE = np.linspace(adogs.physical_lb[0], adogs.physical_ub[0], self.size)
        adogs.func_prior_yE = np.zeros(adogs.func_prior_xE.shape[0])
        adogs.func_prior_sigma = np.zeros(adogs.func_prior_xE.shape[0])
        for i in range(self.size):
            adogs.func_prior_yE[i] = adogs.truth_eval(adogs.func_prior_xE[i], adogs.T0)
            adogs.func_prior_sigma[i] = adogs.sigma_eval(adogs.func_prior_xE[i], adogs.T0) * 2

        # For the evaluated data point, the uncertainty has been reduced
        for i in range(adogs.xE.shape[0]):
            val, idx, xmin = Utils.mindis(adogs.xE[:, i], adogs.func_prior_xE)
            if val < 1e-10:
                adogs.func_prior_sigma[idx] = adogs.sigma[i]

    def initial_plot1D(self, adogs):
        fig = plt.figure(figsize=[16, 9])
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor((0.773, 0.769, 0.769))

        # Display the range of the plot
        plt.ylim(adogs.plot_ylow, adogs.plot_yupp)
        plt.xlim(adogs.physical_lb - .05, adogs.physical_ub + .05)

        # plot the objective function
        plt.plot(adogs.func_prior_xE, adogs.func_prior_yE, 'k', label=r'$f(x)$')
        ucb, lcb = adogs.func_prior_yE + 2 * adogs.func_prior_sigma, \
                   adogs.func_prior_yE - 2 * adogs.func_prior_sigma

        # plot the uncertainty
        plt.fill_between(adogs.func_prior_xE, ucb, lcb, color='grey', alpha=.3)

        plt.grid(color='white')
        plt.legend(loc='lower left')
        self.fig_saver('initial1D', adogs)
        plt.close(fig)

    def initial_calc2D(self, adogs):
        """
        Plot the initial objective function of 2D parameter space.
        :return:
        """
        x = np.linspace(adogs.physical_lb[0], adogs.physical_ub[0], self.size)
        y = np.linspace(adogs.physical_lb[1], adogs.physical_ub[1], self.size)
        adogs.func_prior_xE_2DX, adogs.func_prior_xE_2DY = np.meshgrid(x, y)
        adogs.func_prior_xE_2DZ = np.zeros((self.size, self.size))

        for i in range(self.size):
            for j in range(self.size):
                point = np.array([[adogs.func_prior_xE_2DX[i, j]], [adogs.func_prior_xE_2DY[i, j]]])
                adogs.func_prior_xE_2DZ[i, j] = adogs.truth_eval(point, adogs.T0)

    def initial_plot2D(self, adogs):
        fig = plt.figure(figsize=[16, 9])
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor((0.773, 0.769, 0.769))

        l = np.linspace(np.min(adogs.func_prior_xE_2DZ), np.max(adogs.func_prior_xE_2DZ), 10)
        cp = ax.contourf(adogs.func_prior_xE_2DX, adogs.func_prior_xE_2DY, adogs.func_prior_xE_2DZ,
                         levels=l, cmap='gray')
        ax.contour(cp, colors='k')

        self.fig_saver('initial2D', adogs)
        plt.close(fig)

    def plot1D(self, adogs):
        """
        1D iteration information ploter:
        1) Every additional sampling, update the uncertainty
        2) Every identifying sampling, plot the minimizer of continuous search function

        :return:
        """
        fig = plt.figure(figsize=[16, 9])
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor((0.773, 0.769, 0.769))

        if adogs.iter_type =='scmin':
            plot_xE = np.copy(adogs.xE[:, :-1])
            plot_yE = np.copy(adogs.yE[:-1])
            plot_sigma = np.copy(adogs.sigma[:-1])

        else:
            plot_xE = np.copy(adogs.xE)
            plot_yE = np.copy(adogs.yE)
            plot_sigma = np.copy(adogs.sigma)
            val, idx, xmin = Utils.mindis(adogs.func_prior_xE, adogs.xE[:, -1])
            adogs.func_prior_sigma[idx] = adogs.sigma[-1]

        # plot the objective function
        plt.plot(adogs.func_prior_xE, adogs.func_prior_yE, 'k')
        ucb, lcb = adogs.func_prior_yE + 2 * adogs.func_prior_sigma, \
                   adogs.func_prior_yE - 2 * adogs.func_prior_sigma

        # plot the uncertainty sigma
        plt.fill_between(adogs.func_prior_xE, ucb, lcb, color='grey', alpha=.3)

        plt.errorbar(plot_xE[0], plot_yE, yerr=plot_sigma, fmt='o', ecolor='b', mec='b', ms=5, mew=8,
                     label='Error bar', zorder=10)

        # plot the continuous search function
        self.continuous_search_plot1D(adogs)
        # plot the discrete search function
        self.discrete_search_plot1D(adogs)

        # Display the range of the plot
        plt.ylim(adogs.plot_ylow, adogs.plot_yupp)
        plt.xlim(adogs.physical_lb - .05, adogs.physical_ub + .05)

        plt.grid(color='white')
        plt.legend(loc='lower left')
        self.fig_saver('plot1D' + str(adogs.iter), adogs)
        plt.close(fig)

    def discrete_search_plot1D(self, adogs):
        '''
        Plot the discrete search function.
        Notice that if identifying samling iteration, there is one more points in xE than sd
        :return:
        '''
        if adogs.iter_type == 'scmin':
            plt.scatter(adogs.xE[0, :-1], adogs.sd, marker='^', s=15, c='C5', zorder=15, label=r'$S_d(x)$')
        else:
            plt.scatter(adogs.xE[0, :], adogs.sd, marker='^', s=15, c='C5', zorder=15, label=r'$S_d(x)$')

        # Illustrate the minimizer of discrete search function
        if adogs.iter_type == 'sdmin':
            plt.scatter(adogs.xE[0, adogs.index_min_yd], adogs.sd[adogs.index_min_yd], marker='^',
                        s=15, c='C6', zorder=25, label=r'min $S_d(x)$')

    def continuous_search_plot1D(self, adogs):
        '''
        Plot the continuous search function.
        :return:
        '''
        xU = Utils.bounds(adogs.lb, adogs.ub, adogs.n)
        if adogs.iter_type == 'scmin' and Utils.mindis(adogs.xc, xU)[0] > 1e-6:
            xi = np.hstack((adogs.xE[:, :-1], adogs.xU))
        else:
            xi = np.hstack((adogs.xE, adogs.xU))

        sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
        tri = np.zeros((xi.shape[1] - 1, 2))
        tri[:, 0] = sx[:xi.shape[1] - 1]
        tri[:, 1] = sx[1:]
        tri = tri.astype(np.int32)

        num_plot_points = 2000
        xe_plot = np.zeros((tri.shape[0], num_plot_points))
        e_plot = np.zeros((tri.shape[0], num_plot_points))
        sc_plot = np.zeros((tri.shape[0], num_plot_points))

        for ii in range(len(tri)):
            simplex_range = np.copy(xi[:, tri[ii, :]])

            # Discretized mesh grid on x direction in one simplex
            x = np.linspace(simplex_range[0, 0], simplex_range[0, 1], num_plot_points)

            # Circumradius and circumcenter for current simplex
            R2, xc = Utils.circhyp(xi[:, tri[ii, :]], adogs.n)

            for jj in range(len(x)):
                # Interpolation p(x)
                p = adogs.inter_par.inter_val(x[jj])

                # Uncertainty function e(x)
                e_plot[ii, jj] = (R2 - np.linalg.norm(x[jj] - xc) ** 2)

                # Continuous search function s(x)
                sc_plot[ii, jj] = p - adogs.K * adogs.K0 * e_plot[ii, jj]

            xe_plot[ii, :] = np.copy(x)

        for i in range(len(tri)):
            # Plot the uncertainty function e(x)
            # plt.plot(xe_plot[i, :], e_plot[i, :] - 5.5, c='g', label=r'$e(x)$')
            # Plot the continuous search function sc(x)
            plt.plot(xe_plot[i, :], sc_plot[i, :], 'r--', zorder=20, label=r'$S_c(x)$')

        yc_min = sc_plot.flat[np.abs(xe_plot - adogs.xc).argmin()]
        plt.scatter(adogs.xc, yc_min, c=(1, 0.769, 0.122), marker='D', zorder=15, label=r'min $S_c(x)$')

    def plot2D(self, adogs):
        fig = plt.figure(figsize=[16, 9])
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # plot the truth function
        w = ax.plot_wireframe(adogs.func_prior_xE_2DX, adogs.func_prior_xE_2DY, adogs.func_prior_xE_2DZ,
                              label=r'$f(x)$')

        # # plot the continuous search function
        # continuous_search_Z = np.zeros(adogs.func_prior_xE_2DX.shape)
        #
        # xi = np.hstack((adogs.xE, adogs.xU))
        # options = 'Qt Qbb Qc' if adogs.n <= 3 else 'Qt Qbb Qc Qx'
        # DT = Delaunay(xi.T, qhull_options=options)
        #
        # for i in range(self.size):
        #     for j in range(self.size):
        #         point = np.array([[adogs.func_prior_xE_2DX[i, j]], [adogs.func_prior_xE_2DY[i, j]]])
        #         simplex_index = DT.find_simplex(point.T)
        #         simplex = xi[:, DT.simplices[simplex_index][0]]
        #         R2, xc = Utils.circhyp(simplex, adogs.n)
        #
        #         p = adogs.inter_par.inter_val(point)
        #         e = R2 - np.linalg.norm(point - xc) ** 2
        #         continuous_search_Z[i, j] = p - adogs.K * adogs.K0 * e
        # w = ax.plot_wireframe(adogs.func_prior_xE_2DX, adogs.func_prior_xE_2DY, continuous_search_Z,
        #                       label=r'$s(x)$')

        # plot the point
        if adogs.iter_type == 'scmin':
            temp_xE = adogs.xE[:, :-1]
        else:
            temp_xE = adogs.xE
        ax.scatter(temp_xE[0, :], temp_xE[1, :], marker='s', s=15, c='r')
        if adogs.iter_type == 'sdmin':
            ax.scatter(adogs.xE[0, adogs.index_min_yd], adogs.xE[1, adogs.index_min_yd],
                       marker='^', s=40, c='g')

        self.fig_saver('plot2D' + str(adogs.iter), adogs)
        plt.close(fig)

    def summary_display(self, adogs):
        if adogs.xmin is not None:
            pos_reltv_error = str(np.round(np.linalg.norm(adogs.xmin - adogs.xE[:, np.argmin(adogs.yE)])
                                           / np.linalg.norm(adogs.xmin) * 100, decimals=4)) + '%'
            val_reltv_error = str(np.round(np.abs(np.min(adogs.yE) - adogs.y0)
                                           / np.abs(adogs.y0) * 100, decimals=4)) + '%'
        else:
            pos_reltv_error = 0
            val_reltv_error = 0

        if adogs.y0 is not None:
            cur_pos_reltv_err = str(np.round(np.linalg.norm(adogs.xmin - adogs.xE[:, -1])
                                             / np.linalg.norm(adogs.xmin) * 100, decimals=4)) + '%'
            cur_val_reltv_err = str(np.round(np.abs(adogs.yE[-1] - adogs.y0) / np.abs(adogs.y0) * 100,
                                             decimals=4)) + '%'
        else:
            cur_pos_reltv_err = 0
            cur_val_reltv_err = 0

        if adogs.iter_type == 'sdmin':
            iteration_name = 'Additional sampling'
        elif adogs.iter_type == 'scmin':
            iteration_name = 'Identifying sampling'
        elif adogs.iter_type == 'refine':
            iteration_name = 'Mesh refine iteration'
        else:
            iteration_name = 'Bug happens!'
        print('============================   ', iteration_name, '   ============================')
        print(' %40s ' % 'No. Iteration', ' %30s ' % adogs.iter)
        print(' %40s ' % 'Mesh size', ' %30s ' % adogs.ms)
        print(' %40s ' % 'X-min', ' %30s ' % adogs.xmin.T[0])
        print(' %40s ' % 'Target Value', ' %30s ' % adogs.y0)
        print("\n")
        print(' %40s ' % 'Candidate point', ' %30s ' % adogs.xE[:, np.argmin(adogs.yE)])
        print(' %40s ' % 'Candidate FuncValue', ' %30s ' % np.min(adogs.yE))
        print(' %40s ' % 'Candidate FuncTime', ' %30s ' % adogs.T[np.argmin(adogs.yE)])
        print(' %40s ' % 'CandidatePosition RelativeError', ' %30s ' % pos_reltv_error)
        print(' %40s ' % 'CandidateValue RelativeError', ' %30s ' % val_reltv_error)
        print("\n")
        print(' %40s ' % 'Current point', ' %30s ' % adogs.xE[:, -1])
        print(' %40s ' % 'Current FuncValue', ' %30s ' % adogs.yE[-1])
        print(' %40s ' % 'Current FuncTime', ' %30s ' % adogs.T[-1])
        print(' %40s ' % 'Current Position RelativeError', ' %30s ' % cur_pos_reltv_err)
        print(' %40s ' % 'Current Value RelativeError', ' %30s ' % cur_val_reltv_err)
        if not adogs.iter_type == 'refine':
            print(' %40s ' % 'CurrentEval point', ' %30s ' % adogs.xE[:, -1])
            print(' %40s ' % 'FuncValue', ' %30s ' % adogs.yE[-1])
        print("\n")

    def summary_plot(self, adogs):
        '''
        This function generates the summary information of DeltaDOGS optimization
        :param yE:  The function values evaluated at each iteration
        :param y0:  The target minimum of objective function.
        :param folder: Identify the folder we want to save plots. "DDOGS" or "DimRed".
        :param xmin: The global minimizer of test function, usually presented in row vector form.
        :param ff:  The number of trial.
        '''
        N = adogs.yE.shape[0]  # number of iteration
        if adogs.y0 is not None:
            yE_best = np.zeros(N)
            yE_reltv_error = np.zeros(N)
            for i in range(N):
                yE_best[i] = min(adogs.yE[:i + 1])
                yE_reltv_error[i] = (np.min(adogs.yE[:i + 1]) - adogs.y0) / np.abs(adogs.y0) * 100
            # Plot the function value of candidate point for each iteration
            fig, ax1 = plt.subplots()
            plt.grid()
            # The x-axis is the function count, and the y-axis is the smallest value DELTA-DOGS had found.
            ax1.plot(np.arange(N) + 1, yE_best, label='Function value of Candidate point', c='b')
            ax1.plot(np.arange(N) + 1, adogs.y0 * np.ones(N), label='Global Minimum', c='r')
            ax1.set_ylabel('Function value', color='b')
            ax1.tick_params('y', colors='b')
            plt.xlabel('Number of Evaluated Datapoints')
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

            # Plot the relative error on the right twin-axis.
            ax2 = ax1.twinx()
            ax2.plot(np.arange(N) + 1, yE_reltv_error, 'g--', label=r'Relative Error=$\frac{f_{min}-f_{0}}{|f_{0}|}$')
            ax2.set_ylabel('Relative Error', color='g')

            ax2.yaxis.set_major_formatter(PercentFormatter())
            ax2.tick_params('y', colors='g')
            plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8))
            # Save the plot
            self.fig_saver('Candidate_point', adogs)
            plt.close(fig)
        else:
            print('Target value y0 is not available, no candidate plot saved.')
        ####################   Plot the distance of candidate x to xmin of each iteration  ##################
        if adogs.xmin is not None:
            fig2 = plt.figure()
            plt.grid()
            xE_dis = np.zeros(N)
            for i in range(N):
                index = np.argmin(adogs.yE[:i + 1])
                xE_dis[i] = np.linalg.norm(adogs.xE[:, index].reshape(-1, 1) - adogs.xmin)
            plt.plot(np.arange(N) + 1, xE_dis, label="Distance with global minimizer")
            plt.ylabel('Distance value')
            plt.xlabel('Number of Evaluated Datapoints')
            plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
            self.fig_saver('Distance', adogs)
            plt.close(fig2)
        else:
            print('Global minimizer xmin is not available, no candidate plot saved.')

    def result_saver(self, adogs):
        adogs_data = {}
        adogs_data['xE'] = adogs.xE
        adogs_data['yE'] = adogs.yE
        adogs_data['sigma'] = adogs.sigma
        adogs_data['T'] = adogs.T
        if adogs.inter_par is not None:
            adogs_data['inter_par_method'] = adogs.inter_par.method
            adogs_data['inter_par_w'] = adogs.inter_par.w
            adogs_data['inter_par_v'] = adogs.inter_par.v
            adogs_data['inter_par_xi'] = adogs.inter_par.xi
        name = os.path.join(adogs.plot_folder, 'data.mat')
        io.savemat(name, adogs_data)

    @staticmethod
    def result_reader(name):
        data = io.loadmat(name)
        xE = data['xE']
        yE = data['yE']
        sigma = data['sigma']
        T = data['T']
        return xE, yE, sigma, T

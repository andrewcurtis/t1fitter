import scipy
import scipy.linalg
from scipy.optimize import minimize
from traits.api import HasTraits, Float, List, Int, Array, Double

class T1Optimize(HasTraits):



    def init_guess(self):
        # make guess based on options:
        # zero

        # use emos or vfa

        # smooth data to reduce noise then use vfa

        pass

    def init_model(self):
        pass

    def init_penalties(self):
        pass

    def objective(self, x):

        # flatten x?
        self.to_flat(x)

        # compute obj fun
        # Options here are to extract valid image region from mask and only compute on
        # interior, or compute all and mask after.
        # Testing with realistic data sizes seems to indicate extracting and copying
        # is slower than just doing the compute .
        # might change for really large data?
        sim = self.model_func(x[:,0], x[:,1],
                                self.params.flips,
                                self.params.b1,
                                self.params.tr)

        # try: replace with numexpr
        retval = 0.5 * np.sum( ((self.ydata - sim)*self.mask)**2 )


        for lambda_scale, penalty in zip(self.params.lambdas, self.params.penalties):
            tmp = lambda_scale * penalty(self.to_vol(x) * self.mask)
            retval += tmp

        # for optimizer
        x.shape = (-1)


    def gradient(self):

        self.to_flat(x)
        sim = self.model_func(x[:,0], x[:,1],
                                self.params.flips,
                                self.params.b1,
                                self.params.tr)

        l2diff = self.ydata - sim

        deriv = self.model_deriv(x[:,0], x[:,1],
                                 self.params.flips,
                                 self.params.b1,
                                 self.params.tr)

        deriv = np.sum( - (diffs[np.newaxis, :, :] * deriv) , axis=1)

    def run_optimize(self):
        pass

    def to_vol(self, x):
        x.shape = self.params.volshape

    def to_flat(self, x):
        x.shape = (-1, self.params.volshape[-1])




def wrap_obj(x, fit_params):
    x.shape = (fit_params.nx * fit_params.ny * fit_params.nz, -1)
    res = objective(x, fit_params)
    x.shape = (-1)
    return res


def wrap_jac(x, fit_params):
    jac = jacobian(x, fit_params).copy()
    jac.shape = (-1)
    return jac


def wrap_opt(x0, fit_params, mlim=20):
    #flatten
    x = x0.copy().reshape(-1)
    nx = x.shape[0]
    bnds = np.zeros((nx, 2))

    #mo
    bnds[::2,0] = 0.00001
    bnds[::2,1 ] = mlim
    #t1
    bnds[1::2,0] = 0.01
    bnds[1::2,1 ] = 10.0

    res = minimize(fun = wrap_obj, x0=x, args = [fit_params],
                    method='L-BFGS-B', jac = wrap_jac, bounds = bnds,
                    options={'maxcor':20, 'ftol':fit_params.ftol})

    return res

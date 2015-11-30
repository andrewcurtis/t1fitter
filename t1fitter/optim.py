import scipy
import scipy.linalg
from scipy.optimize import minimize
from traits.api import HasTraits, Float, List, Int, Array, Double, Object

class T1Fit(HasTraits):

    params = Object

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
        pass


    def gradient(self):
        pass

    def run_fit(self):
        pass

    def to_vol(self, x):
        x.shape = self.params.volshape

    def to_flat(self, x):
        x.shape = (-1, self.params.volshape[-1])





class T1FitNLLSReg(T1Fit):

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

        # TODO: how to mult in mask? need to avoid transpose, fix model deriv shape
        deriv = np.sum( - (diffs[np.newaxis, :, :] * deriv) , axis=1)




    def wrap_opt(x0, fit_params, mlim=20):
        #flatten
        nx = x0.shape[0]
        bnds = np.zeros((nx, 2))

        #mo
        bnds[::2,0] = 0.00001
        bnds[::2,1 ] = mlim
        #t1
        bnds[1::2,0] = 0.01
        bnds[1::2,1 ] = 10.0

        # TODO: version check to know if args needs a list wrapper or not

        res = minimize(fun = wrap_obj, x0=x0, args = [fit_params],
                        method='L-BFGS-B', jac = wrap_jac, bounds = bnds,
                        options={'maxcor':20, 'ftol':fit_params.ftol})

        return res

    def run_fit(self):
        pass

    def multisolve(self):
        """ Iteratuve solution with gradual penalty reduction """
        pass





class T1FitDirect(T1Fit):

    def run_fit():
        pass


    def vfa_polyfit(flips, data, tr, b1):

        flips.shape=(-1,1)
        b1.shape=(1,-1)
        sa = np.sin(flips*b1)
        ta = np.tan(flips*b1)

        ys = data / sa
        xs = data / ta

        fits = np.zeros((xs.shape[1],2))
        mask = b1.ravel() > 0.05*np.max(abs(b1))

        for j in range(xs.shape[1]):
            if mask[j]:
                fits[j,:] = np.polyfit(xs[:,j], ys[:,j], 1)

        t1s = -tr/np.log(fits[:,0])
        t1s[np.isnan(t1s)]=0
        t1s[np.isinf(t1s)]=0
        t1s[mask<1]=0

        m0 = (fits[:,1])

        mnot =  m0 / (1-np.exp(-(tr)/t1s))

        mnot[np.isnan(t1s)]=0
        mnot[np.isinf(t1s)]=0

        return mnot, t1s, mask


    def vfa_fit(flips, data, tr, b1):

        flips.shape = (-1,1)
        b1.shape = (1,-1)
        sa = np.sin(flips*b1)
        ta = np.tan(flips*b1)

        ys = data / sa
        xs = data / ta

        fits = np.zeros((xs.shape[1],2))

        mask = b1.ravel() > 0.05*np.max(np.abs(b1))

        fits[:,0] = (ys[1,:] - ys[0,:])/(xs[1,:] - xs[0,:])
        fits[:,1] = ys[1,:] - fits[:,0].T*xs[1,:]

        t1s = -tr/np.log(fits[:,0])
        t1s[np.isnan(t1s)]=0
        t1s[np.isinf(t1s)]=0
        t1s[mask<1]=0

        m0 = (fits[:,1])
        mnot =  m0 / (1-np.exp(-(tr)/t1s))
        mnot[np.isnan(mnot)]=0
        mnot[np.isinf(mnot)]=0
        mnot[mask<1]=0

        return mnot, t1s, mask


    def emos_fit(flips, data, tr, b1):

        flips.shape=(2)
        b1.shape=(-1)

        s1 = data[0,:]
        s2 = data[1,:]

        mask = b1.ravel() > 0.05*np.max(np.abs(b1))

        s0 = s1/(b1*flips[0])

        e1calc = (s0*np.sin(b1*flips[1]) - s2) / (s0*np.sin(b1*flips[1])-s2*np.cos(b1*flips[1]))
        t1s = -tr/np.log(e1calc)

        t1s[np.isnan(t1s)]=0
        t1s[np.isinf(t1s)]=0
        t1s[mask<1]=0

        s0[np.isnan(s0)]=0
        s0[np.isinf(s0)]=0
        s0[mask<1]=0

        return s0, t1s, mask

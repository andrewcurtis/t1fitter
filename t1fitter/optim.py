import scipy
import scipy.linalg
from scipy.optimize import minimize
from traits.api import HasTraits, Float, List, Int, Array, Generic

class T1Fit(HasTraits):

    params = Generic


    def __init__(self, t1pars):
        self.params = t1pars
        
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
        x.shape = self.params.volshape + 2

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
        sim = self.params.model_func(x[:,0], x[:,1],
                                self.params.flips,
                                self.params.b1,
                                self.params.tr)

        # try: replace with numexpr
        retval = 0.5 * np.sum( ((self.params.data - sim) * self.params.mask)**2 )


        for lambda_scale, penalty in zip(self.params.lambdas, self.params.penalties):
            tmp = lambda_scale * penalty.reg_func(self.to_vol(x) * self.mask)
            retval += tmp

        # for optimizer
        x.shape = (-1)


    def gradient(self, x):

        self.to_flat(x)
        sim = self.params.model_func(x[:,0], x[:,1],
                                self.params.flips,
                                self.params.b1,
                                self.params.tr)

        l2diff = self.params.data - sim

        deriv = self.params.model_deriv(x[:,0], x[:,1],
                                 self.params.flips,
                                 self.params.b1,
                                 self.params.tr)

        # TODO: how to mult in mask? need to avoid transpose, fix model deriv shape
        deriv = np.sum( - (diffs[np.newaxis, :, :] * deriv) , axis=1)

        for lambda_scale, penalty in zip(self.params.lambdas, self.params.penalties):
            tmp = lambda_scale * penalty.reg_deriv(self.to_vol(x) * self.mask)
            retval += tmp


    def run_fit(self):
        
        #flatten
        nx = self.params.x0.shape[0]
        bnds = np.zeros((nx, 2))

        #mo
        bnds[::2,0] = 0.001
        bnds[::2,1 ] = 8.0 * self.params.huber_scale
        #t1
        bnds[1::2,0] = 0.01
        bnds[1::2,1 ] = 8.0 * self.params.huber_scale

        # TODO: version check to know if args needs a list wrapper or not

        res = minimize(fun = self.objective, x0=self.params.x0, args = self.params,
                        method='L-BFGS-B', jac = self.gradient, bounds = bnds,
                        options={'maxcor':self.params.maxcor, 'ftol':self.params.ftol})

        return res
        
        
    def multisolve(self):
        """ Iterative solution with gradual penalty reduction """
        pass





class T1FitDirect(T1Fit):

    def run_fit():
        pass


    def vfa_polyfit(self, flips, data, tr, b1):

        flips.shape=(-1,1)
        b1.shape=(1,-1)
        sa = np.sin(flips*b1)
        ta = np.tan(flips*b1)

        ys = data / sa
        xs = data / ta

        fits = np.zeros((xs.shape[1],2))

        for j in range(xs.shape[1]):
            if self.params.mask[j]:
                fits[j,:] = np.polyfit(xs[:,j], ys[:,j], 1)

        t1s = -tr/np.log(fits[:,0])
        t1s[np.isnan(t1s)]=0
        t1s[np.isinf(t1s)]=0
        t1s[self.params.mask<1]=0

        m0 = (fits[:,1])

        mnot =  m0 / (1-np.exp(-(tr)/t1s))

        mnot[np.isnan(t1s)]=0
        mnot[np.isinf(t1s)]=0

        return np.concatenate((mnot, t1s), axis=3)


    def vfa_fit(self, flips, data, tr, b1):

        flips.shape = (-1,1)
        b1.shape = (1,-1)
        sa = np.sin(flips*b1)
        ta = np.tan(flips*b1)

        ys = data / sa
        xs = data / ta

        fits = np.zeros((xs.shape[1],2))

        fits[:,0] = (ys[1,:] - ys[0,:])/(xs[1,:] - xs[0,:])
        fits[:,1] = ys[1,:] - fits[:,0].T*xs[1,:]

        t1s = -tr/np.log(fits[:,0])
        t1s[np.isnan(t1s)]=0
        t1s[np.isinf(t1s)]=0
        t1s[self.params.mask<1]=0

        m0 = (fits[:,1])
        mnot =  m0 / (1-np.exp(-(tr)/t1s))
        mnot[np.isnan(mnot)]=0
        mnot[np.isinf(mnot)]=0
        mnot[self.params.mask<1]=0

        sz = mnot.shape
        
        return np.concatenate((mnot, t1s), axis=3)


    def emos_fit(self, flips, data, tr, b1):

        flips.shape=(2)
        b1.shape=(-1)

        s1 = data[0,:]
        s2 = data[1,:]

        s0 = s1/(b1*flips[0])

        e1calc = (s0*np.sin(b1*flips[1]) - s2) / (s0*np.sin(b1*flips[1])-s2*np.cos(b1*flips[1]))
        t1s = -tr/np.log(e1calc)

        t1s[np.isnan(t1s)]=0
        t1s[np.isinf(t1s)]=0
        t1s[self.params.mask<1]=0

        s0[np.isnan(s0)]=0
        s0[np.isinf(s0)]=0
        s0[self.params.mask<1]=0

        return np.concatenate((s0, t1s), axis=3)

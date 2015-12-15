import numpy as np
import logging
import scipy
import os
import scipy.linalg
import nibabel as nib
from scipy.optimize import minimize
from traits.api import HasTraits, Float, List, Int, Array, Generic

class T1Fit(HasTraits):

    params = Generic


    def __init__(self, t1pars, debugpath=None):
        self.params = t1pars
        self.log = logging.getLogger('T1Fit')
        self.obj_scale = 1.0
        self.itercount = 0
        self.debugpath = debugpath

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
        x.shape = self.params.volshape + [2]

    def to_flat(self, x):
        x.shape = (-1, 2)





class T1FitNLLSReg(T1Fit):



    def objective(self, x):



        # reshape x to (flat , mo/t1)
        self.log.debug('x shape: {}'.format(x.shape))
        self.to_flat(x)
        self.log.debug('x shape: {}'.format(x.shape))


        if self.debugpath is not None:
            if self.itercount % 10 == 0:
                self.scratch *= 0.0;
                self.to_flat(self.scratch)
                self.scratch[self.mask_flat,:] = x
                self.to_vol(self.scratch)
                mid2 = self.scratch.shape[1]
                tmp = self.scratch[:,int(mid2/2),:,:]
                scipy.misc.toimage(np.c_[tmp[:,:,0],tmp[:,:,1]],
                        cmin=0.0, cmax=6.0).save(os.path.join(self.debugpath, 'iter{}.png'.format(self.itercount)))


        # compute obj fun
        # Options here are to extract valid image region from mask and only compute on
        # interior, or compute all and mask after.
        # Testing with realistic data sizes seems to indicate extracting and copying
        # is slower than just doing the compute .
        # might change for really large data?
        sim = self.params.model_func(x[:,0], x[:,1],
                                self.params.flips,
                                self.b1[self.mask_flat],
                                self.params.trs)

        # try: replace with numexpr
        self.log.debug('datashape: {}'.format(self.params.data.shape))
        self.log.debug('sim shape: {}'.format(sim.shape))

        retval = self.obj_scale * 0.5 * np.sum( ((self.params.data[:,self.mask_flat] - sim))**2 )
        self.log.info('fit to data term: {}'.format(retval))


        if self.params.l1_lam > 0:
            self.to_flat(self.scratch)
            self.scratch *= 0.0;
            self.scratch[self.mask_flat,:] = x
            self.to_vol(self.scratch)
            tmp = self.params.l1_lam * self.params.spatialreg.reg_func(self.scratch)
            self.log.info('l1 term: {}'.format(tmp))
            retval += tmp

        if self.params.l2_lam > 0:
            tmp = self.params.l2_lam * self.params.l2reg.reg_func(x[:,1])
            self.log.info('l2 term: {}'.format(tmp))
            retval += tmp

        # ravel for optimizer
        x.shape = (-1)

        self.itercount += 1
        self.log.debug('obj called iter {}'.format(self.itercount))

        return retval


    def gradient(self, x):

        self.log.debug('x shape: {}'.format(x.shape))

        self.to_flat(x)

        self.log.debug('x shape: {}'.format(x.shape))

        self.grad *= 0
        self.to_flat(self.grad)

        if self.params.l1_lam > 0:

            self.scratch *= 0.0;
            self.to_flat(self.scratch)
            self.log.debug('x shape: {}'.format(x.shape))
            self.log.debug('scratch shape: {}'.format(self.scratch.shape))
            self.scratch[self.mask_flat,:] = x[:,:]
            self.to_vol(self.scratch)


            self.grad_scratch *= 0.0;
            self.to_vol(self.grad_scratch)
            self.params.spatialreg.reg_deriv(self.scratch, self.grad_scratch)
            self.to_flat(self.grad_scratch)

            tmp =  self.params.l1_lam * self.grad_scratch[self.mask_flat,:]
            self.log.info('l1 grad norm: {}'.format(np.sqrt(np.sum(tmp**2))))

            self.grad -= tmp



        if self.params.l2_lam > 0:

            l2dif = self.params.l2reg.reg_deriv(x[:,1])
            self.log.debug('l2dif shape: {}'.format(l2dif.shape))

            tmp = self.params.l2_lam * l2dif
            self.log.info('l2 grad norm: {}'.format(np.sqrt(np.sum(tmp**2))))

            self.grad[:,1] += tmp


        sim = self.params.model_func(x[:,0], x[:,1],
                                self.params.flips,
                                self.b1[self.mask_flat],
                                self.params.trs)

        #flipvols X vox
        l2diff = self.params.data[:,self.mask_flat] - sim
        self.log.debug('l2diff shape: {}'.format(l2diff.shape))

        # pars X flipvosl X vox
        deriv = self.params.model_deriv(x[:,0], x[:,1],
                                 self.params.flips,
                                 self.b1[self.mask_flat],
                                 self.params.trs)

        # TODO: how to mult in mask? need to avoid transpose, fix model deriv shape
        deriv = np.sum( - (l2diff[np.newaxis, :, :] * deriv) , axis=1)
        self.log.debug('deriv shape: {}'.format(deriv.shape))

        self.log.info('obj grad norm: {}'.format(np.sqrt(np.sum(deriv**2))))

        # vox X pars
        self.grad += self.obj_scale * deriv[:,:].T

        # ravel for optimizer
        x.shape = (-1)

        return self.grad.reshape(-1)



    def run_fit(self, x0, bounds, prep_only=False):

        #extract inner region so optimization domain is smaller
        #make logical mask
        self.mask_flat = self.params.mask.copy().ravel() > 0

        self.to_flat(x0)

        self.x0 = x0[self.mask_flat,:].copy().reshape(-1)
        #we only fit whats in the mask, so gradient is that size X num pars
        self.grad = np.zeros_like(self.x0)



        self.b1 = self.params.b1map.copy().ravel()

        if self.params.l1_lam > 0:
            self.scratch = np.zeros_like(x0)
            self.grad_scratch = np.zeros_like(x0)

        #flatten
        nx = self.x0.shape[0]
        bnds = np.zeros((nx, 2))

        self.nx = nx/2.0

        #mo
        bnds[::2,0] = bounds[0,0]
        bnds[::2,1 ] = bounds[1,0]
        #t1
        bnds[1::2,0] = bounds[0,1]
        bnds[1::2,1 ] = bounds[1,1]


        if not prep_only:
            res = minimize(fun = self.objective, x0=self.x0,
                            method='L-BFGS-B', jac = self.gradient, bounds = bnds,
                            options={'maxcor':self.params.maxcor, 'ftol':self.params.fit_tol,
                                     'maxiter':self.params.maxiter, 'maxfun':self.params.maxfun})

            print('results : {}'.format(res))

            if res.success is not True:
                self.log.error('Fitting error! {}'.format(res))

            tmp = res.x
            tmp.shape=(-1,2)

            result = np.zeros_like(x0)
            result[self.mask_flat,:] = tmp[:,:]

            return result


    def multisolve(self):
        """ Iterative solution with gradual penalty reduction like SOR """

        self.scratch = np.zeros_like(self.params.x0)
        self.grad_scratch = np.zeros_like(self.params.x0)

        #extract inner region so optimization domain is smaller
        self.mask_flat = self.to_flat(self.params.mask)>0
        self.x0 = self.to_flat(self.params.x0)[self.mask_flat,:]

        #flatten
        nx = self.x0.shape[0]
        bnds = np.zeros((nx, 2))

        #mo
        bnds[::2,0] = 0.001
        bnds[::2,1 ] = 10.0
        #t1
        bnds[1::2,0] = 0.01
        bnds[1::2,1 ] = 10.0


        # ravel for optimizer
        self.x0.shape=(-1)


        for j in [25, 15, 10, 5 ,1]:

                self.params.l1_lam = j*self.params.l1_lam
                self.params.l2_lam = j/2.0*self.params.l2_lam

                res = minimize(fun = self.objective, x0=self.x0, args = self.params,
                                method='L-BFGS-B', jac = self.gradient, bounds = bnds,
                                options={'maxcor':self.params.maxcor, 'ftol':self.params.ftol})

                self.x0 = res.x.copy()

        return res





class T1FitDirect(T1Fit):

    def run_fit():
        pass


    def vfa_polyfit(self, flips, data, tr, b1):
        self.to_flat(data)

        flips.shape=(-1,1)
        b1.shape=(1,-1)
        sa = np.sin(flips*b1)
        ta = np.tan(flips*b1)

        ys = data / sa
        xs = data / ta

        fits = np.zeros((xs.shape[1],2))
        fmask = self.params.mask.ravel()

        for j in range(xs.shape[1]):
            if fmask[j]:
                fits[j,:] = np.polyfit(xs[:,j], ys[:,j], 1)

        t1s = -tr/np.log(fits[:,0])
        t1s[np.isnan(t1s)]=0
        t1s[np.isinf(t1s)]=0


        t1s[fmask<1]=0

        m0 = (fits[:,1])

        mnot =  m0 / (1-np.exp(-(tr)/t1s))

        mnot[np.isnan(t1s)]=0
        mnot[np.isinf(t1s)]=0

        return np.concatenate((mnot, t1s), axis=3)


    def vfa_fit(self, flips, data, tr, b1):

        data.shape = (2, -1)
        flips.shape = (-1,1)
        b1.shape = (1,-1)
        sa = np.sin(flips*(b1+1e-9))
        ta = np.tan(flips*(b1+1e-9))

        ys = data / sa
        xs = data / ta

        fits = np.zeros((xs.shape[1],2))

        fits[:,0] = (ys[1,:] - ys[0,:])/(xs[1,:] - xs[0,:])
        fits[:,1] = ys[1,:] - fits[:,0].T*xs[1,:]


        t1s = -tr/np.log(fits[:,0]).reshape(-1,1)
        t1s[np.isnan(t1s)]=0
        t1s[np.isinf(t1s)]=0

        fmask = self.params.mask.ravel()

        t1s[fmask<1]=0

        m0 = (fits[:,1]).reshape(-1,1)
        mnot =  m0 / (1-np.exp(-(tr)/t1s))
        mnot[np.isnan(mnot)]=0
        mnot[np.isinf(mnot)]=0
        mnot[fmask<1]=0

        sz = mnot.shape

        return np.concatenate((mnot, t1s), axis=1)


    def emos_fit(self, flips, data, tr, b1):
        self.to_flat(data)

        flips.shape=(2)
        b1.shape=(-1)

        s1 = data[0,:]
        s2 = data[1,:]

        s0 = s1/(b1*flips[0])

        e1calc = (s0*np.sin(b1*flips[1]) - s2) / (s0*np.sin(b1*flips[1])-s2*np.cos(b1*flips[1]))
        t1s = -tr/np.log(e1calc)

        t1s[np.isnan(t1s)]=0
        t1s[np.isinf(t1s)]=0

        fmask = self.params.mask.ravel()

        t1s[fmask<1]=0

        s0[np.isnan(s0)]=0
        s0[np.isinf(s0)]=0
        s0[fmask<1]=0

        return np.concatenate((s0, t1s), axis=3)

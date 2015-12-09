

import sys
sys.path.append('../t1fitter/')
import t1fitter
import util
import optim

import scipy
import numpy as np



def main():

    fitter = t1fitter.T1Fitter()

    #make some small fake data so we can test gradients

    sz = (10,10,8)

    dat = np.zeros(sz + (2,))

    dat[:5,:,:,0] = 1.5
    dat[5:,:,:,0] = 2

    dat[:5,:,:,1] = 2
    dat[5:,:,:,1] = 2

    dat[:,5:,...] *= 0.5

    dat[:,:,:5,:] *= 2

    b1 = np.ones(sz)
    mask = np.zeros_like(b1)

    mask[2:-2,2:-2,2:-2] = 1

    fitter.data = dat
    fitter.mask = mask
    fitter.b1map = b1
    fitter.volshape = list(sz)

    fitter.l1_lam = 0
    fitter.kern_sz = 1
    fitter.huber_scale = 0.25

    fitter.l2_lam = 0
    fitter.outpath='.'
    fitter.l2_mode = 'zero'
    fitter.start_mode = 'zero'
    fitter.fit_method='nlreg'


    fitter.flips = np.array([14.0, 3.0])*np.pi/180.0
    fitter.trs = np.array([10.0, 10.0])*1e-3


    fitter.nlreg_fit(prep_only=True)

    # get t1 optimizer instance once everything is prepped
    tfit = fitter.tfit


    #check close to data
    scipy.optimize.check_grad(tfit.objective, tfit.gradient, x0 = dat*0.9.ravel())





if __name__ == '__main__':
    main()

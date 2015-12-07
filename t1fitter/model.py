'''
model.py

    Functions to model the T1 behaviour of our data.
    SPGR, VFA, and eMOS.


    Model functions for signal equation.
    Using numexpr as a fast way to speed things up.  Numexpr will automatically multi-
    thread numpy array computations, in addition to avoiding intermediate arrays.


    Likely want to move to numba in future (or C).


    TODO: test performance of more complex expressions...
        e.g. if I don't break cos/sin/exp out as separate evals, should save som inter-
            mediate work space
        - Figure out nice way of generalizing which model to use.
        - Handling of complex data?
        - alloc space for deriv once and pass in?
'''


import numpy as np
import numexpr as ne


class T1Models(object):
    """docstring for T1Models"""
    def __init__(self, nthreads=4):
        super(T1Models, self).__init__()
        ne.set_num_threads(nthreads)

    @staticmethod
    def spgr(flips, m, t1, b1, tr):
        b1.shape = (1,-1)
        flips.shape = (-1,1)
        m.shape = (1,-1)
        t1.shape= (1,-1)

        ca = ne.evaluate('cos(b1*flips)')
        sa = ne.evaluate('sin(b1*flips)')
        e1 = ne.evaluate('exp(-tr/t1)')
        sig = ne.evaluate('m * sa * (1-e1)/(1-ca*e1)')

        return sig

    # flatten SPGR into a single numexpr call to try and avoid intermediates
    @staticmethod
    def spgr_flat(flips, m, t1, b1, tr):

        return ne.evaluate('m * sin(b1*flips) * (1-exp(-tr/t1)) / (1-cos(b1*flips) * exp(-tr/t1))')

    @staticmethod
    def spgr_deriv_mt(flips, m, t1, b1, tr):
        b1.shape = (1,-1)
        flips.shape = (-1,1)
        m.shape = (1,-1)
        t1.shape= (1,-1)

        e1 = ne.evaluate('exp(-tr/t1)')
        e1p = ne.evaluate('exp(tr/t1)')
        ca = ne.evaluate('cos(b1*flips)')
        sa = ne.evaluate('sin(b1*flips)')

        dm = ne.evaluate('sa * (1-e1)/(1-ca*e1)')
        dt1 = ne.evaluate('((ca-1) * m*sa * tr * e1p)/(t1**2 * (ca-e1p)**2)')

        return np.array([dm, dt1])

    @staticmethod
    def spgr_deriv_m(flips, m, t1, b1, tr):
        b1.shape = (1,-1)
        flips.shape = (-1,1)
        m.shape = (1,-1)
        t1.shape= (1,-1)
        
        e1 = ne.evaluate('exp(-tr/t1)')
        e1p = ne.evaluate('exp(tr/t1)')
        ca = ne.evaluate('cos(b1*flips)')
        sa = ne.evaluate('sin(b1*flips)')

        dm = ne.evaluate('sa * (1-e1)/(1-ca*e1)')

        return np.array([dm])

    @staticmethod
    def spgr_deriv_t(flips, m, t1, b1, tr):

        e1 = ne.evaluate('exp(-tr/t1)')
        e1p = ne.evaluate('exp(tr/t1)')
        ca = ne.evaluate('cos(b1*flips)')
        sa = ne.evaluate('sin(b1*flips)')

        dt1 = ne.evaluate('((ca-1) * m*sa * tr * e1p)/(t1**2 * (ca-e1p)**2)')

        return np.array([ dt1])


    @staticmethod
    def spgr_deriv_mtb(flips, m, t1, b1, tr):

        e1 = ne.evaluate('exp(-tr/t1)')
        e1p = ne.evaluate('exp(tr/t1)')
        ca = ne.evaluate('cos(b1*flips)')
        sa = ne.evaluate('sin(b1*flips)')

        dm = ne.evaluate('sa * (1-e1)/(1-ca*e1)')
        dt1 = ne.evaluate('((ca-1) * m*sa * tr * e1p)/(t1**2 * (ca-e1p)**2)')
        db1 =  ne.evaluate('flips * m * (e1 - 1)*(e1 - ca) / (e1*ca - 1)**2')

        return np.array([dm, dt1, db1])

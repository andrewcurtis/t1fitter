"""
model.py

Functions to model the T1 behaviour of our data.
SPGR, VFA, and eMOS.


"""

import numpy as np
import numexpr as ne




'''
    Model functions for signal equation, and objective and jacobian for fitting
'''

def spgr_ne(flips, m, t1, b1, tr):
    b1.shape = (1,-1)
    flips.shape = (-1,1)
    m.shape = (1,-1)
    t1.shape= (1,-1)

    ca = ne.evaluate('cos(b1*flips)')
    sa = ne.evaluate('sin(b1*flips)')
    e1 = ne.evaluate('exp(-tr/t1)')
    sig = ne.evaluate('m * sa * (1-e1)/(1-ca*e1)')

    return sig


def spgr_deriv_ne(flips, m, t1, b1, tr):

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


def spgr_deriv_ne_b1(flips, m, t1, b1, tr):

    flips.shape = (-1,1)
    b1.shape= (1,-1)
    m.shape = (1,-1)
    t1.shape= (1,-1)

    e1 = ne.evaluate('exp(-tr/t1)')
    e1p = ne.evaluate('exp(tr/t1)')
    ca = ne.evaluate('cos(b1*flips)')
    sa = ne.evaluate('sin(b1*flips)')

    dm = ne.evaluate('sa * (1-e1)/(1-ca*e1)')
    dt1 = ne.evaluate('((ca-1) * m*sa * tr * e1p)/(t1**2 * (ca-e1p)**2)')

    db1 =  ne.evaluate('flips * m* (e1 - 1)*(e1 - ca) / (e1*ca - 1)**2')

    tmp = np.array([dm, dt1, db1])

    return tmp

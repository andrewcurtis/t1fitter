

import numpy as np

from numpy.fft import fftshift, fftn, fft, ifft, ifftn



#some image utility functions


def interpz(x, sz2):
    tmp = fftshift(fft(fftshift(x, axes=[-1]), axis=-1), axes=[-1])
    sz = tmp.shape
    filt = np.kaiser(sz[2], 1.5).reshape(1,1,-1)
    tmp2 = np.zeros((sz[0], sz[1], sz2), dtype=tmp.dtype)
    tmp2[:,:,sz2/2-sz[2]/2:sz2/2+sz[2]/2] = (tmp[:,:,:]* filt)
    return np.abs(fftshift(ifft(fftshift(tmp2, axes=[-1]), axis=-1), axes=[-1]))

def interp3d(x, sz2):
    tmp = fftshift(fftn(fftshift(x, axes=[0,1,2]), axes=[0,1,2]), axes=[0,1,2])
    sz = tmp.shape

    #x
    filt = np.kaiser(sz[0], 1.5).reshape(-1,1,1)
    tmp = tmp * filt
    #y
    filt = np.kaiser(sz[1], 1.5).reshape(1,-1,1)
    tmp = tmp * filt
    #z
    filt = np.kaiser(sz[2], 1.5).reshape(1,1,-1)
    tmp = tmp * filt

    tmp2 = np.zeros(sz2, dtype=tmp.dtype)
    tmp2[sz2[0]/2-sz[0]/2:sz2[0]/2+sz[0]/2, sz2[1]/2-sz[1]/2:sz2[1]/2+sz[1]/2, sz2[2]/2-sz[2]/2:sz2[2]/2+sz[2]/2] = tmp[:,:,:]
    return np.abs(fftshift(fftn(fftshift(tmp2, axes=[0,1,2]), axes=[0,1,2]), axes=[0,1,2]))

def filt3d(x, lvl):
    tmp = fftshift(fftn(fftshift(x, axes=[0,1,2]), axes=[0,1,2]), axes=[0,1,2])
    sz = tmp.shape

    #x
    filt = np.kaiser(sz[0], lvl).reshape(-1,1,1)
    tmp = tmp * filt
    #y
    filt = np.kaiser(sz[1], lvl).reshape(1,-1,1)
    tmp = tmp * filt
    #z
    filt = np.kaiser(sz[2], lvl).reshape(1,1,-1)
    tmp = tmp * filt


    return np.abs(fftshift(ifftn(fftshift(tmp, axes=[0,1,2]), axes=[0,1,2]), axes=[0,1,2]))




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
    mnot[np.isnan(t1s)]=0

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

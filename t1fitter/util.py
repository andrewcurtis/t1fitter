

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

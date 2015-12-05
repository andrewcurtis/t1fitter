

import numpy as np
import sh
import os

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


def split_nifti_pathname(x):
    if x.endswith('.nii.gz'):
        return os.path.splitext(os.path.splitext(x)[0])[0]
    elif x.endswith('.nii')
        return os.path.splitext(x)[0] 

def preproc_spgr_align_and_crop(ref_vol, other_vols, crop_edge=1, outprefix='proc', ext = '.nii.gz'):
    #align all to ref
    
    proc_vols = []
    for fname in other_vols:
        outname = split_nifti_pathname(fname)
        base, name = os.path.split(outname) 
        
        outname = os.path.join(base, outprefix + name + '_align')
        proc_vols.append(outname)
        sh.flirt('-dof','6','-in',fname,'-ref',ref_vol,'-out',outname,)
        
    
    
    #bet on the ref vol then apply to other vols
    outname = split_nifti_pathname(ref_vol)
    base, name = os.path.split(outname) 
    outname = os.path.join(base, outprefix + name + '_bet')
    maskname = outname + '_mask'
    sh.bet(refvol, outname, '-m','-f','0.4')
    
    betted = []
    betted.append(outname)
    
    for fname in proc_vols:
        betname = fname + '_bet'
        sh.fslmaths(fname, '-mul', maskname, betname )
        betted.append(betname)
        
    
    clip_lims = sh.fslstats(betted[0], '-w')
    
    clip_lims = np.array(clip_lims.split()).astype(int)
    clip_lims.shape = (4,2)
    
    max_shape = nib.load(betted[0]+ext).get_shape()
    
    #extend clip vol
    clip_lims[:,0] -= crop_edge
    clip_lims[:,1] += crop_edge
    
    #check that we're in ovlume
    clip_lims[ clip_lims[:,0] < 0 ,0] = 0
    
    over = clip_lims[:3,0] + clip_lims[:3,1] >= max_shape[:3]
    clip_lims[over , 1] = max_shape[ over ] - clip_lims[over, 0]
    
    clip_lims.shape=(-1)
    clims = list(clip_lims)
    
    cropped = []
    
    for fname in betted:
        cropname = fname + '_crop'
        cropped.append(cropname)
        sh.fslroi(fname, cropname, *clims)
        
    return cropped
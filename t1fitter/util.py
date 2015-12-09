

import numpy as np
import sh
import os

from numpy.fft import fftshift, fftn, fft, ifft, ifftn
import nibabel as nib


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
    elif x.endswith('.nii'):
        return os.path.splitext(x)[0]

def preproc_spgr_align_and_crop(ref_vol, other_vols, crop_edge=1, outprefix='proc',outpath='', ext = '.nii.gz', skip=0):
    #align all to ref

    proc_vols = []
    for fname in other_vols:
        outname = split_nifti_pathname(fname)
        base, name = os.path.split(outname)

        outname = os.path.join(base, outpath, outprefix + name + '_align')
        proc_vols.append(outname)
        if skip < 1:
            sh.flirt('-dof','6','-in',fname,'-ref',ref_vol,'-out',outname,)



    #bet on the ref vol then apply to other vols
    outname = split_nifti_pathname(ref_vol)
    base, name = os.path.split(outname)
    outname = os.path.join(base,outpath, outprefix + name + '_bet')
    maskname = outname + '_mask'
    if skip < 2:
        sh.bet(ref_vol, outname, '-m','-f','0.4')

    betted = []
    betted.append(outname)

    for fname in proc_vols:
        betname = fname + '_bet'
        if skip < 3:
            sh.fslmaths(fname, '-mul', maskname, betname )
        betted.append(betname)


    clip_lims = sh.fslstats(betted[0], '-w')

    clip_lims = np.array(clip_lims.split()).astype(int)
    clip_lims.shape = (4,2)

    max_shape = nib.load(betted[0]+ext).get_shape()

    #extend clip vol
    clip_lims[:3,0] -= crop_edge
    clip_lims[:3,1] += 2*crop_edge
    # Note: roi info is (start, len) so if we move start back, we need to
    # extend len by double

    #check that we're in ovlume
    clip_lims[ clip_lims[:,0] < 0 ,0] = 0


    for j in range(3):
        if (clip_lims[j,0] + clip_lims[j,1]) >= max_shape[j]:
            clip_lims[j , 1] = max_shape[ j ] - clip_lims[j, 0]

    clip_lims.shape=(-1)
    clims = list(clip_lims)

    cropped = []

    for fname in betted:
        cropname = fname + '_crop'+ext
        cropped.append(cropname)
        sh.fslroi(fname, cropname, *clims)

    cropmask = maskname+'_crop'+ext
    if skip < 4:
        sh.fslroi(maskname, cropmask, *clims)

    return cropped, cropmask, clims



def preproc_b1mos(ref_vol, cropped_mask_vol, clip_lims, in_vols, flips, outprefix='mos',outpath='', ext = '.nii.gz'):

    #align first fa to ref

    fa1_align = split_nifti_pathname(in_vols[0])
    base, name = os.path.split(fa1_align)
    fa1_align = os.path.join(base,outpath, outprefix + name + '_align')

    sh.flirt('-dof','7','-in',in_vols[0],'-ref',ref_vol,
                '-out',fa1_align,'-omat','b1mos_to_ref.mat')

    fa2_align = split_nifti_pathname(in_vols[1])
    base, name = os.path.split(fa2_align)
    fa2_align = os.path.join(base,outpath, outprefix + name + '_align')

    sh.flirt('-in',in_vols[1],'-ref',ref_vol,'-out',fa2_align,
            '-paddingsize',0.0,'-interp','trilinear',
            '-applyxfm','-init', 'b1mos_to_ref.mat')


    cropped = []

    for fname in [fa1_align, fa2_align]:
        cropname = fname + '_crop'
        cropped.append(cropname)
        sh.fslroi(fname, cropname, *clip_lims)

    masked = []
    for fname in cropped:
        maskname = fname + '_mask'
        masked.append(maskname)
        sh.fslmaths(fname, '-mul', cropped_mask_vol, maskname )


    # now load them in and finish processing in python

    fa1 = nib.load(masked[0]+ext).get_data().astype('float')
    fa2 = nib.load(masked[1]+ext).get_data().astype('float')

    affine =  nib.load(masked[0]+ext).get_affine()

    mask = fa1>0

    smooth1 = filt3d(fa1, 30)
    smooth2 = filt3d(fa2, 30)

    # fit b1
    slope = (smooth2 - smooth1) / (flips[1]-flips[0])
    yinter = -smooth1 + slope*flips[0]

    slope = slope*mask
    yinter = yinter*mask

    xinter = yinter/slope
    xinter = xinter*mask

    scale = 180.0/xinter
    scale = scale*mask
    scale[np.isinf(scale)]=0.0
    scale[np.isnan(scale)]=0.0
    scale[scale>2.0]=2.0
    scale[scale<0.0] = 0.0
    scale = scale*mask

    tmp = nib.Nifti1Image( scale, affine )
    b1map = os.path.join(base, outpath, 'b1_mos.nii.gz')
    tmp.to_filename(b1map)


    return b1map

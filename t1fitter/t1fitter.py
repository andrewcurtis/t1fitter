"""
Main T1fitter class. 
Handles problem setup, choices for regularizers and fitting methods.
"""


import numpy as np
import argparse
import logging
import nibabel as nib
import logging
import os
import sys

import model
import optim
import util
import regularization

from traits.api import HasTraits, Float, List, Int, Array, \
    Directory, Bool, Enum, String, List




class T1Fitter(HasTraits):

    # paths
    file_path = Directory
    file_list = List
    file_list_orig = List
    mask_path = String
    b1vol = String


    outpath = String
    outname = String
    debugpath = String(None)

    b1scale = Float(1.0)

    #data
    trs = Array
    flips = Array
    b1map = Array
    data = Array
    mask = Array
    # NLreg Fit returns a OptimizationResult which is like a dict. 
    # other fits just create an Array. 
    fit = Python
    prior = Array

    #Store base volume shape 
    volshape = List
    #remember affine matrix for output
    base_image_affine = Array

    fit_method = Enum('vfa','emos','nlreg')

    #fitting parameters
    t1_range = Array
    m0_range = Array
    kern_sz = Int(1)
    #recriprocal huber cutoff
    delta = Float(0.3)
    #params for BFGS
    fit_tol = Float(1e-4)
    maxcor = Int(15)
    maxiter = Int(300)
    maxfun = Int(3000)
    nthreads = Int(4)


    #Regularization options
    l1_lam = Float(5e-4)
    l1_mode = Enum('huber','welsch')

    l2_lam = Float(2e-6)
    l2_prior = Bool(False)
    l2_mode = Enum('zero','vfa','smooth_vfa')
    
    start_mode = Enum('zero','vfa','smooth_vfa','file')

    lambdas = List

    #params for preproc
    smooth_fac = Float(25.0)
    crop_padding = Int(4)
    clip_lims = List()

    debug = Bool(False)



    def __init__(self):

        FORMAT = "[%(filename)s:%(lineno)4s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(format=FORMAT)
        self.log = logging.getLogger('T1Fit')
        #default off
        self.log.setLevel(0)


    def run_fit(self):

        if self.fit_method == 'vfa':
            self.vfa_fit()
        elif self.fit_method == 'emos':
            self.emos_fit()
        elif self.fit_method == 'nlreg':
            self.nlreg_fit()

        self.save_fit_results()


    def save_fit_results(self):

        fname = os.path.join(self.outpath, 'm0_{}.nii.gz'.format(self.outname))
        self.log.info('Saving M0 volume to {}'.format(fname))
        self.write_nifti(self.fit[...,0], fname)

        fname = os.path.join(self.outpath, 't1_{}.nii.gz'.format(self.outname))
        self.log.info('Saving T1 volume to {}'.format(fname))
        self.write_nifti(self.fit[...,1], fname)

        # did we also fit for b1+ ?
        if self.fit.shape[-1] > 2:
            fname = os.path.join(self.outpath, 'b1_{}.nii.gz'.format(self.outname))
            self.log.info('Saving B1 volume to {}'.format(fname))
            self.write_nifti(self.fit[...,2], fname)



    def run_preproc(self):
        # skull strip main vol and keep mask
        self.log.info('Preproc calling spgr_align_and_crop')

        # align volumes.  preproc(referce, remaining)
        processed, mask, cliplims = util.preproc_spgr_align_and_crop(self.file_list[0],
                                         self.file_list[1:],
                                         self.crop_padding,
                                         outpath = self.outpath)

        #replace input file list with processed files. order is reference:rest
        #explicity copy
        self.file_list_orig = list(self.file_list)

        #replace old params with new ones
        self.file_list = processed
        self.clip_lims = cliplims
        self.mask_path = mask


        self.log.info('Preproc complete, files: {}\n clipping: {}\nmask: {}\n'.format(
                self.file_list, self.clip_lims, self.mask_path ))



    def run_preproc_b1mos(self, mos_args):
        ### b1 map prep
        self.log.info('B1 MOS calculation from args: {}'.format(mos_args))

        if len(mos_args) != 2:
            self.log.debug('Found wrong number of MOS volumes for b1 calibration!')

        in_vols = [mos_args[0][0], mos_args[1][0]]
        flips = map(float, [mos_args[0][1], mos_args[1][1]])

        self.b1vol = util.preproc_b1mos(self.file_list_orig[0], self.mask_path,
                    self.clip_lims, in_vols, flips, outpath=self.outpath)

        self.log.info('B1MOS map saved to: {}... Loading.'.format(self.b1vol))

        self.b1map = nib.load(self.b1vol).get_data()



    def run_preproc_b1bs(self, args):
        # bloch siegert from GE has two volumes in the nifti.
        # vol 0 is the image reference# vol 1 is the map.
        #use vol 0 to align to our t1 data, then interp and smooth the b1+
        pass


    def write_nifti(self, vol, fname):
        self.log.info('Writing nifti to {}'.format(fname))

        tmp = nib.Nifti1Image(vol, affine=self.base_image_affine)
        tmp.to_filename(fname)


    def load_vols(self, dest, files):

        nvols = len(files)
        #find datasize
        tmp = nib.load(files[0])
        dat_sz = tmp.get_shape()

        self.volshape = list(dat_sz)

        self.log.debug('First vol had shape: {}'.format(dat_sz))

        dat_sz = [nvols] + list(dat_sz)

        self.log.debug('Alloc array memory with shape {}'.format(dat_sz))

        self.data = np.zeros(dat_sz, dtype=tmp.get_data_dtype())

        for idx, f in enumerate(files):
            self.log.info('Loading volume {}: {}'.format(idx, f))
            tmp  = nib.load(f).get_data()
            self.data[idx, ...] = tmp

        self.base_image_affine = nib.load(files[0]).get_affine()
        if self.mask_path:
            self.mask = nib.load(self.mask_path).get_data()


    def load_startvols(self):

        if self.init_files is not None:

            self.log.info('Loading data files for optimizer init x0.')

            m0 = nib.load(self.init_files[0][0]).get_data()
            t1 = nib.load(self.init_files[0][1]).get_data()

            assert(list(m0.shape) == self.volshape)
            assert(list(t1.shape) == self.volshape)

            m0.shape = list(m0.shape) + [1]
            t1.shape = list(t1.shape) + [1]

            return np.concatenate((m0,t1), axis=3).copy()



    def make_smooth_vfa(self):
        tfit = optim.T1FitDirect(self)

        takeflips = [0,1]
        #make a copy we can discard after, since some of the fitting munges the shapes
        flips = self.flips[takeflips,...].copy()
        data = self.data[takeflips,...].copy()

        self.log.debug('Smoothing data by {}'.format(self.smooth_fac))

        data.shape = [2] + self.volshape
        data[0,...] = util.filt3d(data[0,...], self.smooth_fac)
        data[1,...] = util.filt3d(data[1,...], self.smooth_fac)

        trs = self.trs[takeflips,...].copy()
        b1 = self.b1map.copy()

        self.log.debug('Fitting.')

        self.fit = tfit.vfa_fit(flips, data, trs[0], b1 )
        self.fit.shape = self.volshape + [2]

        if self.debug:
            self.log.debug('Saving smoothed prior volume.')
            fname = os.path.join(self.outpath, 'm0_prior_smooth.nii.gz')
            self.write_nifti(self.fit[...,0], fname)

            fname = os.path.join(self.outpath, 't1_prior_smooth.nii.gz')
            self.write_nifti(self.fit[...,1], fname)



    def vfa_fit(self):
        tfit = optim.T1FitDirect(self)

        takeflips = [0,1]
        #make a copy we can discard after, since some of the fitting munges the shapes
        flips = self.flips[takeflips,...].copy()
        data = self.data[takeflips,...].copy()
        trs = self.trs[takeflips,...].copy()
        b1 = self.b1map.copy()

        self.fit = tfit.vfa_fit(flips, data, trs[0], b1 )
        self.fit.shape = self.volshape + [2]

        toobig = self.fit[...,1]>10.0
        self.fit[toobig] = 10.0
        self.fit[self.fit<0.0] = 0.0

        self.log.info('after vfa_fit, datasize is: {}'.format(self.data.shape))
        self.log.info('after vfa_fit, fit size is: {}'.format(self.fit.shape))



    def nlreg_fit(self, prep_only=False):

        # TODO: run vfa fit to find m0, in order to scale data properly.
        # want mean(t1) ~ mean(m0)

        def apply_bounds(x, bounds):
            temp_sz = x.shape
            x.shape = (-1,2)

            tmp = x[:,0]<bounds[0,0]
            x[tmp,0]=bounds[0,0]

            tmp = x[:,1]<bounds[0,1]
            x[tmp,1]=bounds[0,1]

            tmp = x[:,0] > bounds[1,0]
            x[tmp,0]=bounds[1,0]

            tmp = x[:,1]>bounds[1,1]
            x[tmp,1]=bounds[1,1]

            x.shape = temp_sz

            return x

        self.vfa_fit()

        tmp = self.fit[...,0]
        mean_m0 = np.mean(tmp[ self.mask>0].ravel())
        tmp = self.fit[...,1]
        mean_t1 = np.mean(tmp[ self.mask>0].ravel())

        datascale = mean_m0 / mean_t1

        self.log.info('Computed VFA for data scaling. Mean m0:{}, T1:{}, scale: {}'.format(
                mean_m0, mean_t1, datascale ))

        self.data *= 1.0/datascale

        bounds = np.array([[0.001,0.01],[20.0,8.0]])


        #setup prior if we're using it
        if self.l2_lam > 0:

            regl = None
            if self.l2_mode == 'smooth_vfa':

                #smooth and fit
                self.make_smooth_vfa()
                self.prior = self.fit.copy()

                tmp_mask =  self.mask.copy().ravel() > 0

                self.prior.shape=(-1,2)
                self.prior = apply_bounds(self.prior, bounds)
                self.prior = self.prior[tmp_mask,1]

                print(sum(self.prior>0))
                regl = regularization.TikhonovDiffReg3D(self.prior)
            else:
                regl = regularization.TikhonovReg3D()


            self.l2reg = regl

        if self.l1_lam > 0:
            if self.l1_mode == 'welsch':
                self.spatialreg = regularization.ParallelWelsch2ClassReg3D(self.volshape + [2],
                                                    delta=self.delta,
                                                    kern_radius=self.kern_sz,
                                                    nthreads=self.nthreads)
            else:
                self.spatialreg = regularization.ParallelHuber2ClassReg3D(self.volshape + [2],
                                                    delta=self.delta,
                                                    kern_radius=self.kern_sz,
                                                    nthreads=self.nthreads)


        # set up functions for optimizer
        t1model = model.T1Models(nthreads=self.nthreads)

        self.model_func = t1model.spgr_flat
        self.model_deriv = t1model.spgr_deriv_mt



        #make x0
        self.data.shape = (len(self.flips), -1)
        # init fitter with our params
        self.tfit = optim.T1FitNLLSReg(self, debugpath=self.debugpath)

        x0 = []

        if self.start_mode == 'vfa':
            self.vfa_fit()
            x0 = self.fit.copy()
        elif self.start_mode == 'smooth_vfa':
            self.make_smooth_vfa()
            x0 = self.fit.copy()
        elif self.start_mode == 'file':
            x0 = self.load_startvols()
        else:
            #if self.start_mode == 'zero':
            x0 = np.zeros(self.volshape + [2])

        # make sure we're starting within a valid region for our bounds
        x0 = apply_bounds(x0, bounds)


        if not prep_only:
            self.fit = self.tfit.run_fit( x0, bounds ).reshape(self.volshape + [2])
        else:
            return x0



    def check_data_sizes(self):

        self.log.info('b1map_size: {}'.format(self.b1map.shape))
        self.log.info('data_size: {}'.format(self.data.shape))
        self.log.info('mask_size: {}'.format(self.mask.shape))
        self.log.info('flips_size: {}'.format(self.flips.shape))

        assert(self.b1map.shape == self.mask.shape)
        #data is #vols x space, each vol needs to match mask/b1
        assert(self.data.shape[1:] == self.mask.shape)

        assert(self.data.shape[0] == self.flips.shape[0])


    def init_traits_gui(self):
        pass



"""
Driver for t1fitter.

Handles CLI or GUI interface.
Reads image volumes and speficications, set up problem, runs, and saves output.
"""


import numpy as np
import argparse
import logging
import nibabel as nib
import logging
import os

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

    trs = Array
    flips = Array

    outname = String

    #data
    b1map = Array
    data = Array
    mask = Array
    fit = Array
    prior = Array

    volshape = List

    fit_method = Enum('vfa','emos','nlreg')

    #fitting parameters
    t1_range = Array
    m0_range = Array
    kern_sz = Int(1)
    #recriprocal huber cutoff
    huber_scale = Float(3.0)
    #params for BFGS
    fit_tol = Float(1e-5)
    maxcor = Int(25)

    #remember affine matrix for output
    base_image_affine = Array

    l1_lam = Float(1e-3)
    l2_lam = Float(1e-4)
    l2_prior = Bool(False)
    l2_mode = Enum('zero','vfa','smooth_vfa')
    start_mode = Enum('zero','vfa','smooth_vfa')



    # to pass to solver
    lambdas = List
    penalties = List



    #params for preproc
    smooth_fac = Float(25.0)
    crop_padding = Int(4)

    clip_lims = List()


    debug = Bool(False)



    def __init__(self):

        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
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


        fname = os.path.join(self.outname, 'm0_{}.nii.gz'.format(self.fit_method))
        self.log.info('Saving M0 volume to {}'.format(fname))
        self.write_nifti(self.fit[...,0], fname)

        fname = os.path.join(self.outname, 't1_{}.nii.gz'.format(self.fit_method))
        self.log.info('Saving T1 volume to {}'.format(fname))
        self.write_nifti(self.fit[...,1], fname)

        # did we also fit for b1+ ?
        if self.fit.shape[-1] > 2:
            fname = os.path.join(self.outname, 'b1_{}.nii.gz'.format(self.fit_method))
            self.log.info('Saving B1 volume to {}'.format(fname))
            self.write_nifti(self.fit[...,2], fname)




    def run_preproc(self):
        # skull strip main vol and keep mask
        self.log.info('Preproc calling spgr_align_and_crop')

        # align volumes.  preproc(referce, remaining)
        processed, mask, cliplims = util.preproc_spgr_align_and_crop(self.file_list[0],
                                         self.file_list[1:],
                                         self.crop_padding)

        #replace input file list with processed files. order is reference:rest
        self.file_list_orig = self.file_list
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
                    self.clip_lims, in_vols, flips)

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

        self.log.info('First vol had shape: {}'.format(dat_sz))

        dat_sz = [nvols] + list(dat_sz)

        self.log.info('Alloc array memory with shape {}'.format(dat_sz))

        self.data = np.zeros(dat_sz, dtype=tmp.get_data_dtype())

        for idx, f in enumerate(files):
            self.log.info('Loading volume {}: {}'.format(idx, f))
            tmp  = nib.load(f).get_data()
            self.data[idx, ...] = tmp

        self.base_image_affine = nib.load(files[0]).get_affine()
        if self.mask_path:
            self.mask = nib.load(self.mask_path).get_data()


    def init_from_cli(self, args):

        self.log.info('Parsing CLI: {}'.format(args))

        self.l1_lam = args.l1lam
        self.l2_lam = args.l2lam

        if args.l1 == False:
            self.l1_lam = 0.0

        if args.l2 == False:
            self.l2_lam = 0.0

        self.l2_mode = args.l2mode

        self.start_mode = args.startmode

        self.kern_sz = args.kern_radius
        self.huber_scale = args.huber_scale

        self.fit_method = args.fit_method
        self.smooth_fac = args.smooth

        self.outname = args.out

        if args.verbose:
            self.log.setLevel(logging.INFO)

        if args.debug:
            self.log.setLevel(logging.DEBUG)
            self.debug = True

        if args.addvol is None:
            self.log.error('Need input volumes!')

        # get filenames
        tmp_tr = []
        tmp_fa = []
        for vol in args.addvol:
            self.log.info('addvol args: {}'.format(vol))
            self.file_list.append(vol[0])
            tmp_fa.append(vol[1])
            tmp_tr.append(vol[2])


        self.flips = np.array(tmp_fa).astype(float) * np.pi / 180.0
        self.trs = np.array(tmp_tr).astype(float) * 1e-3
        self.log.info('found flips: {}, trs: {}'.format(self.flips, self.trs))


        if args.preproc:
            self.log.info('preprocessing selected, running')
            self.run_preproc()

        # preprocessing will change file_list entries, so load after.
        self.load_vols(self.data, self.file_list)


        if args.maskvol is not None:
            self.log.info('Found mask volume {}, overriding self.mask'.format(args.maskvol))
            self.mask = nib.load(args.maskvol).get_data()

        if args.b1vol is not None:
            self.log.info('Found b1 volume {}'.format(args.b1vol))
            self.b1vol = args.b1vol
            self.b1map = nib.load(args.b1vol).get_data()
        else:
            self.log.info('No b1 map given, looking for source data to generate map.')
            #if no b1, check if we can process it from the arguments
            if args.mosvol is not None:
                self.log.info('B1 MOS data found, processing.')
                self.run_preproc_b1mos(args.mosvol)

        self.check_data_sizes()


    def make_smooth_vfa(self):
        tfit = optim.T1FitDirect(self)

        takeflips = [0,1]
        #make a copy we can discard after, since some of the fitting munges the shapes
        flips = self.flips[takeflips,...].copy()
        data = self.data[takeflips,...].copy()

        self.log.info('Smoothing data by {}'.format(self.smooth_fac))
        data[0,...] = util.filt3d(data[0,...], self.smooth_fac)
        data[1,...] = util.filt3d(data[1,...], self.smooth_fac)

        trs = self.trs[takeflips,...].copy()
        b1 = self.b1map.copy()

        self.log.info('Fitting.')

        self.fit = tfit.vfa_fit(flips, data, trs[0], b1 )
        self.fit.shape = self.volshape + [2]

        if self.debug:
            self.log.debug('Saving smoothed prior volume.')
            fname = os.path.join(self.outname, 'm0_prior_smooth.nii.gz')
            self.write_nifti(self.fit[...,0], fname)

            fname = os.path.join(self.outname, 't1_prior_smooth.nii.gz')
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
        self.log.info('after vfa_fit, datasize is: {}'.format(self.data.shape))
        self.log.info('after vfa_fit, fit size is: {}'.format(self.fit.shape))



    def nlreg_fit(self):

        # TODO: run vfa fit to find m0, in order to scale data properly.
        # want mean(t1) ~ mean(m0)

        self.vfa_fit()

        tmp = self.fit[...,0]
        mean_m0 = np.mean(tmp[ self.mask>0].ravel())
        tmp = self.fit[...,1]
        mean_t1 = np.mean(tmp[ self.mask>0].ravel())

        datascale = mean_m0 / mean_t1

        self.log.info('Computed VFA for data scaling. Mean m0:{}, T1:{}, scale: {}'.format(
                mean_m0, mean_t1, datascale ))

        self.data *= 1.0/datascale


        #setup prior if we're using it
        if self.l2_lam > 0:

            regl = None
            if self.l2_mode == 'smooth_vfa':

                #smooth and fit
                self.make_smooth_vfa()
                self.prior = self.fit.copy()

                tmp_mask = self.mask.copy()
                tmp_mask.shape = (-1)

                self.prior.shape=(-1,2)
                self.prior = self.prior[tmp_mask>0,:]

                regl = regularization.TikhonovDiffReg3D(self.prior)
            else:
                regl = regularization.TikhonovReg3D()


            self.l2reg = regl

        if self.l1_lam > 0:
            self.hubreg = regularization.ParallelHuber2ClassReg3D(self.volshape + [2],
                                                        hub_scale=self.huber_scale,
                                                        kern_radius=self.kern_sz)
        # set up functions for optimizer
        t1model = model.T1Models()

        self.model_func = t1model.spgr
        self.model_deriv = t1model.spgr_deriv_mt


        self.data.shape = (len(self.flips), -1)
        # init fitter with our params
        tfit = optim.T1FitNLLSReg(self)

        #make x0

        x0 = []

        if self.start_mode == 'vfa':
            x0 = self.fit.copy()
        else:
        #if self.start_mode == 'zero':
            x0 = np.zeros(self.volshape + [2])


        self.fit = tfit.run_fit( x0 )



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




def t1fit_cli():

    parser = argparse.ArgumentParser(description='T1fitter. VFA, eMOS, and NLReg, with optional preprocessing.')

    parser.add_argument('--addvol', '-a', nargs=3,  action='append', metavar=('vol','flip','tr'), required=True,
                        help='Add volume for fitting with flip angle (deg) and tr (ms)')

    parser.add_argument('--out', '-o', default='t1fit',
                        help='Output volume base name for t1 and m0 fit.')

    parser.add_argument('--preproc', action="store_true",
                        help='Run preprocessing on input volumes (alignment, brain extraction, cropping).')

    parser.add_argument('--crop_padding', type=int, default=6,
                        help='Edge padding for minimum volume crop size.')

    parser.add_argument('--maskvol',
                        help='Brain mask volume (must be provided if no preprocessing is used).')

    parser.add_argument('--b1vol',
                        help='Pre calculated B1 map (as relative scaling of base FAs).')

    parser.add_argument('--smooth', type=float, default=25.0,
                        help='Additional smoothing factor for b1, and/or smoothing for mos b1 calc\'n')

    parser.add_argument('--mosvol', nargs=2,  action='append', metavar=('vol','flip'),
                        help='Add volume for MOS B1 map calculations with associated flip angle (deg)')



    #fitting options

    parser.add_argument('--l1', action='store_true',
                        help='Enable Huber penalty. See  -l1lam to set penatly scaling. ')
    parser.add_argument('--l1lam', type=float, default=1e-3,
                        help='l1 lambda: scaling factor for Huber penalty')
    parser.add_argument('--kern_radius', type=int, default=1,
                        help='Huber spatial kernel radius.')
    parser.add_argument('--huber_scale', type=float, default=3.0,
                        help='Huber spatial kernel radius.')


    parser.add_argument('--l2', action='store_true',
                        help='Enable l2 Tikhonov penalty. See  -l2lam to set penatly scaling. ')
    parser.add_argument('--l2lam', type=float, default=1e-3,
                        help='l2 lambda: scaling factor for Tikhonov penalty')
    parser.add_argument('--l2mode', choices=['zero','vfa','smooth_vfa'], default='zero',
                        help='l2 Tikhonov penalty mode -- Distance from smooth prior, or zero (normal Tik). ')

    parser.add_argument('--startmode', choices=['zero','vfa','smooth_vfa'], default='zero',
                        help='Start mode -- start from zero or from vfa guess. ')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output.')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Debug output.')

    parser.add_argument('--fit_method', choices=['vfa','emos','nlreg'], default='vfa',
                        help='Fit method.')


    #simulation opions
    parser.add_argument('--addnoise', type=float, default=-1,
                        help='Add noise. Normaize input data to 1 and add this std() worth of noise.')

    cmd_args = parser.parse_args()

    print(cmd_args)


    fitter = T1Fitter()
    fitter.init_from_cli(cmd_args)
    fitter.run_fit()




if __name__ == '__main__':
    t1fit_cli()

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

from traits.api import HasTraits, Float, List, Int, Array, Directory, Bool, Enum

class T1Fitter(HasTraits):
    
    # inputs
    file_path = Directory
    file_list = List
    trs = Array
    flips = Array
    
    debug_lvl = Int(0)
    out_path = Directory
    
    #data
    b1map = Array
    data = Array
    mask = Array
    fit = Array
    prior = Array
    
    #fitting parameters
    t1_range = Array
    m0_range = Array
    
    #remember affine matrix for output
    base_image_affine = Array
    
    l1_lam = Float(1e-3)
    l2_lam = Float(1e-4)
    l2_prior = Bool(False)
    
    kern_sz = Int(1)
    #recriprocal huber cutoff
    huber_scale = Float(3.0)
    
    smooth_fac = Float(25.0)
    crop_padding = Int(4)
    
    fit_method = Enum('vfa','emos','nlreg')
    
    
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
        
        self.log.info('')
        
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
        #fsl['bet']()
        
        # estimate extents fslstats -write_nifti
        
        # parse extents and grow by max kern size
        
        # crop roi
        
        # align volumes
    

        
        
        pass

    
    def run_preproc_b1mos(self, mos_args):
        ### b1 map prep
        # bet fa130
        
        # align to main image, saving transformation
        
        # extract and transform fa150
        # get filenames
        
        if len(mos_args) != 2:
            self.log.debug('Found wrong number of MOS volumes for b1 calibration!')
        
        pass


    
    def run_preproc_b1bs(self, args):
        
        pass
    
    def write_nifti(self, vol, fname):
        
        tmp = nib.Nifti1Image(vol, affine=self.affine)
        tmp.to_filename(fname)
        
    
    def load_vols(self, dest, files):
        
        nvols = len(files)
        #find datasize
        tmp = nib.load(files[0])
        dat_sz = tmp.get_shape()
        
        self.log.info('First vol had shape: {}'.format(dat_sz))
        
        dat_sz = nvols + list(dat_sz)
        
        self.log.info('Alloc array memory with shape {}'.format(dat_sz))
        
        dest = np.zeros(dat_sz, dtype=tmp.get_data_dtype())
        
        for idx, f in enumerate(files):
            self.log.info('Loading volume {}: {}'.format(idx, f))
            tmp  = nib.load(f).get_data()
            dest[idx, ...] = tmp



    
    def init_from_cli(self, args):
        
        
        self.l1_lam = args.l1lam
        self.l2_lam = args.l2lam
        
        if args.l1 == False:
            self.l1_lam = 0.0
        
        if args.l2 == False:
            self.l2_lam = 0.0
        
        self.l2_mode = args.l2mode
        
        self.kern_sz = args.kern_radius
        self.huber_scale = args.huber_scale
        
        self.fit_method = args.huber_scale
        self.smooth_fac = args.smooth
        
        
        if args.verbose:
            self.log.setLevel(logging.INFO)
        
        if args.debug:
            self.log.setLevel(logging.DEBUG)
            
        
        # get filenames
        tmp_tr = []
        tmp_fa = []
        for vol in args.addvol:
            self.file_list.append(vol.vol)
            tmp_fa.append(vol.flip)
            tmp_tr.append(vol.tr)
        
        self.flips = np.array(tmp_fa)
        self.trs = np.array(tmp_tr)
        
        
        if args.preproc:
            log.info('preprocessing selected, running')
            self.run_preproc()
        
        # preprocessing will change file_list entries, so load after.
        self.load_vols(self.data, self.file_list)
        
        
        if args.maskvol is not None:
            log.info('Found mask volume, overriding self.mask')
            self.mask = nib.load().get_data()
        
        if args.b1vol is not None:
            self.b1map = nib.load(args.b1vol).get_data()
        else:
            log.info('No b1 map given, looking for source data to generate map.')
            #if no b1, check if we can process it from the arguments
            if args.mosvol is not None:
                log.info('B1 MOS data found, processing.')
                self.run_preproc_b1mos(args.mosvol)
        
        self.check_data_sizes()
    
    
    def check_data_sizes():
        
        assert(self.b1map.shape == self.mask.shape)
        #data is #vols x space, each vol needs to match mask/b1
        assert(self.data.shape[1:] == self.mask.shape)
        
        assert(self.data.shape[0] == self.flips.shape[0])

    
    def init_traits_gui(self):
        pass




def t1fit_cli():
    
    parser = argparse.ArgumentParser(description='T1fitter. VFA, eMOS, and NLReg, with optional preprocessing.')
    parser.add_argument('--addvol', '-a', nargs=3,  action='append', metavar=('vol','flip','tr'),
                        help='Add volume for fitting with flip angle (deg) and tr (ms)')

    
    parser.add_argument('--out', '-o', default='t1fit',
                        help='Output volume base name for t1 and m0 fit.')
    
    parser.add_argument('--preproc', action="store_true",
                        help='Run preprocessing on input volumes (alignment, brain extraction, cropping).')
    
    parser.add_argument('--crop_padding', type=int, default=4,
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
    parser.add_argument('--l2mode', choices=['zero','smooth_vfa'], default='zero',
                        help='l2 Tikhonov penalty mode -- Distance from smooth prior, or zero (normal Tik). ')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output.')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Debug output.')

    
    cmd_args = parser.parse_args()
    
    print(cmd_args)
    
    
    fitter = T1Fitter()
    fitter.init_from_cli(cmd_args)
    
    



if __name__ == '__main__':
    t1fit_cli()

"""
Driver for t1fitter.

Handles CLI or GUI interface.
Reads image volumes and speficications, set up problem, runs, and saves output.
"""


import numpy as np
import argparse
from traits.api import HasTraits, Float, List, Int, Array, Double, Directory

class T1Fitter(HasTraits):

    # inputs
    file_path = Directory
    file_list = List
    trs = Array
    flips = Array

    out_path = Directory

    #data
    b1map = Array
    data = Array
    mask = Array

    #fitting parameters
    t1_range = Array
    m0_range = Array

    #remember affine matrix for output 
    base_image_affine = Array



    def load_file_list(self, files):
        pass

    def set_fit_params(self):
        pass

    def run_fit(self):
        pass

    def run_preproc_spgr(self):
        pass

    def run_preproc_b1mos(self):
        pass

    def run_preproc_b1bs(self):
        pass

    def write_nifti(self):
        pass

    def init_from_cli(self):
        pass

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
                        help='B1 map (as relative scaling of base FAs).')
 
    parser.add_argument('--b1smooth', type=float, default=25.0,
                        help='smoothing factor for b1')
                    
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


    cmd_args = parser.parse_args()
    
    fitter = T1Fitter()
    fitter.init_from_cli(cmd_args)
    


if __name__ == '__main__':
    main()
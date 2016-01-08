"""
Driver for t1fitter.

Handles CLI interface and execution logic.

"""


import argparse
import os
import sys
import numpy as np
import t1fitter
import logging

import nibabel as nib

def gen_cli():

    parser = argparse.ArgumentParser(description='T1fitter. VFA, eMOS, and NLReg, ' +
                                            'with optional preprocessing.')

    basic_group = parser.add_argument_group('Main Arguments')
    basic_group.add_argument('--addvol', nargs=3,  action='append', metavar=('vol','flip','tr'), required=True,
                        help='Add volume for fitting with flip angle (deg) and tr (ms)')
    basic_group.add_argument('--out', default='t1fit',
                        help='Output path.')

    basic_group.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output.')
    basic_group.add_argument('--debug', '-d', action='store_true',
                        help='Debug output.')
    basic_group.add_argument('--descriptive_names', action='store_true',
                        help='Output fit names with param descriptions.')
    basic_group.add_argument('--debug_image_path', default=None,
                        help='Path for fit progress image dump.')
    basic_group.add_argument('--nthreads', default=4, type=int,
                        help='Number of threads to use for computation.')
    basic_group.add_argument('--b1scale', default=1.0, type=float,
                        help='Scale b1.')


    extravols_group = parser.add_argument_group('Additional Input Volumes')
    extravols_group.add_argument('--maskvol',
                        help='Brain mask volume (must be provided if no preprocessing is used).')

    extravols_group.add_argument('--b1vol',
                        help='Pre calculated B1 map (as relative scaling of base FAs).')


    preproc_group = parser.add_argument_group('Preprocessing')
    preproc_group.add_argument('--preproc', action="store_true",
                        help='Run preprocessing on input volumes ' +
                            '(alignment, brain extraction, cropping).')

    preproc_group.add_argument('--preproc_only', action="store_true",
                        help='Don\'t run fit, only preprocessing steps.')

    preproc_group.add_argument('--crop_padding', type=int, default=6,
                        help='Edge padding for minimum volume crop size.')

    preproc_group.add_argument('--mosvol', nargs=2,  action='append', metavar=('vol','flip'),
                        help='Add volume for MOS B1 map calculations with associated flip angle (deg)')

    preproc_group.add_argument('--smooth', type=float, default=25.0,
                        help='Smoothing for mos b1 calc\'n')


    #fitting options
    fit_group = parser.add_argument_group('Fitting options')

    fit_group.add_argument('--fit_method', choices=['vfa','emos','nlreg','hub_wel'], default='vfa',
                        help='Fit method.')
    fit_group.add_argument('--l1lam', type=float, default=1e-3,
                        help='l1 lambda: scaling factor for spatial regularizer, 0 == disabled')
    fit_group.add_argument('--l1mode', choices=['huber','welsch'], default='huber',
                        help='pseudo-l1 spatial regularizer mode -- huber or welsch. ')
    fit_group.add_argument('--kern_radius', type=int, default=1,
                        help='Spatial regularizer kernel radius.')
    fit_group.add_argument('--delta', type=float, default=0.4,
                        help='Spatial regularizer scale factor.')

    fit_group.add_argument('--tol', type=float, default=1e-5,
                        help='Fitting tolerance.')
    fit_group.add_argument('--maxiter', type=int, default=300,
                        help='Optimizer iteration limit.')
    fit_group.add_argument('--maxfun', type=int, default=3000,
                        help='Function evaluation limit.')


    fit_group.add_argument('--l2lam', type=float, default=0.0,
                        help='l2 lambda: scaling factor for Tikhonov regularizer. 0 == disabled')
    fit_group.add_argument('--l2mode', choices=['zero','vfa','smooth_vfa'], default='zero',
                        help='l2 Tikhonov regularizer mode -- Distance from [smooth] prior, or zero (normal Tik). ')
    fit_group.add_argument('--smooth_prior', type=float, default=12.0,
                        help='Smoothing for Prior\'n')
    fit_group.add_argument('--startmode', choices=['zero','vfa','smooth_vfa','file'], default='zero',
                        help='Start mode -- start from zero or from vfa guess. ')
    fit_group.add_argument('--startvols', nargs=2,  action='append', metavar=('m0','t1'),
                        help='Specify initialization volume for fitting (required with startmode=file).')




    return parser




def main():

    parser = gen_cli()
    args = parser.parse_args()


    fitter = t1fitter.T1Fitter()
    fitter.log.info('Parsing CLI: {}')

    new_cli = ''

    fitter.nthreads = args.nthreads
    fitter.b1scale = args.b1scale

    fitter.fit_tol = args.tol
    fitter.maxiter = args.maxiter
    fitter.maxfun  = args.maxfun

    fitter.l1_lam = args.l1lam
    fitter.l2_lam = args.l2lam

    fitter.l1_mode = args.l1mode
    fitter.l2_mode = args.l2mode

    fitter.start_mode = args.startmode
    fitter.init_files = args.startvols

    fitter.kern_sz = args.kern_radius
    fitter.delta = args.delta

    fitter.fit_method = args.fit_method
    fitter.smooth_fac = args.smooth

    fitter.outpath = args.out

    if args.verbose:
        fitter.log.setLevel(logging.INFO)

    if args.debug:
        fitter.log.setLevel(logging.DEBUG)
        fitter.debug = True

    if args.addvol is None:
        fitter.log.error('Need input volumes!')

    # get filenames
    tmp_tr = []
    tmp_fa = []
    for vol in args.addvol:
        fitter.log.info('addvol args: {}'.format(vol))
        fitter.file_list.append(vol[0])
        tmp_fa.append(vol[1])
        tmp_tr.append(vol[2])


    fitter.flips = np.array(tmp_fa).astype(float) * np.pi / 180.0
    fitter.trs = np.array(tmp_tr).astype(float) * 1e-3
    fitter.log.debug('found flips: {}, trs: {}'.format(fitter.flips, fitter.trs))


    if args.preproc:
        fitter.log.debug('preprocessing selected, running')
        fitter.run_preproc()

        for j, fname in enumerate(fitter.file_list):
            new_cli = new_cli + ' --addvol {} {} {} '.format(fname,
                        fitter.flips[j]*180.0/np.pi, fitter.trs[j] *1e3)

        new_cli = new_cli + ' --mask {} '.format(fitter.mask_path)


    # preprocessing will change file_list entries, so load after.
    fitter.load_vols(fitter.data, fitter.file_list)


    if args.maskvol is not None:
        fitter.log.info('Found mask volume {}, overriding fitter.mask'.format(args.maskvol))
        fitter.mask = nib.load(args.maskvol).get_data()

    if args.b1vol is not None:
        fitter.log.info('Found b1 volume {}'.format(args.b1vol))
        fitter.b1vol = args.b1vol
        fitter.b1map = nib.load(args.b1vol).get_data()
    else:
        fitter.log.info('No b1 map given, looking for source data to generate map.')
        #if no b1, check if we can process it from the arguments
        if args.mosvol is not None:
            fitter.log.info('B1 MOS data found, processing.')
            fitter.run_preproc_b1mos(args.mosvol)
            new_cli = new_cli + ' --b1vol {} '.format(fitter.b1vol)


    #Check data sizes will also determine if we are missing volumes (mask, b1, etc)
    # and create some sensible defaults.
    fitter.check_data_sizes()

    fitter.b1map *= fitter.b1scale


    # set ouput name. Basic is t1_method and m0_method.
    fitter.outname = fitter.fit_method

    # add detail to filename. Useful for when you're doing many fits
    # e.g. arraying over parameters.
    #Can always add more info the the names here.
    if args.descriptive_names:

        if fitter.fit_method is not 'vfa':
            if fitter.l1_lam > 0:
                fitter.outname = fitter.outname + '_l1{}_k{}_d{}_m{}'.format(fitter.l1_lam,
                                        fitter.kern_sz, fitter.delta, fitter.l1_mode)
            if fitter.l2_lam > 0:
                fitter.outname = fitter.outname + '_l2{}_s{}_m{}'.format(fitter.l2_lam,
                                        fitter.smooth_fac, fitter.l2_mode)
            fitter.outname = fitter.outname + '_sm{}_ftol{}_ncv{}_nvol{}'.format(fitter.start_mode,
                                        fitter.fit_tol, fitter.maxcor, len(fitter.file_list))


    # report modified filenames for future use.
    if len(new_cli) > 0:
        print('\nProcessing changed files. If you want to rerun the fitting ' +
             ' with different arguments, please use the following command line options:\n\n')
        print(new_cli + '\n')

    if args.debug_image_path is not None:
        fitter.debugpath = args.debug_image_path

    if not args.preproc_only:
        fitter.run_fit()




if __name__ == '__main__':
    main()

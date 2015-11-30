"""
Driver for t1fitter.

Handles CLI or GUI interface.
Reads image volumes and speficications, set up problem, runs, and saves output.
"""


import numpy as np

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

    base_image_affine = Array

    def __init__(self):
        pass

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

    def init_from_cmd_line(self):
        pass

    def init_traits_gui(self):
        pass

    

"""
Driver for t1fitter.

Handles CLI or GUI interface.
Reads image volumes and speficications, set up problem, runs, and saves output.
"""


import numpy as np

from traits.api import HasTraits, Float, List, Int, Array, Double

class T1Fitter(HasTraits):

    fileList = List
    trs = Array
    flips = Array
    data = Array

    mask = Array

    def __init__(self):
        pass

    def load_file_list(self, files):
        pass

    def set_fit_params(self):
        pass

    def run_fit(self):
        pass

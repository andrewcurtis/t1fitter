
"""
This module contains regularizers for the fitting problem.
Both generic (dimension and class independant) and specialized
(multi threaded, data size aware) functions are here, and the
driver routine should choose the appropriate one.

"""
import numpy as np
import numexpr as ne
import logging
import threading
from ctypes import pythonapi, c_void_p

from numba import autojit, jit, void, double, int64

from traits.api import HasTraits, Float, List, Int, Array, CFloat

from numpy import sqrt, exp

savethread = pythonapi.PyEval_SaveThread
savethread.argtypes = []
savethread.restype = c_void_p

restorethread = pythonapi.PyEval_RestoreThread
restorethread.argtypes = [c_void_p]
restorethread.restype = None


class Regularizer3D(HasTraits):
    """ Basic form of regularizers expose reg_func and reg_deriv
        Implements a l2 penalty on the parameter vector.
        Subclasses override the interface reg_func and reg_deriv.
    """

    def __init__(self, nthreads=8):
        ne.set_num_threads(nthreads)

    def reg_func(self, x):
        return ne.evaluate('sum(x**2)')

    def reg_deriv(self, x):
        return ne.evaluate('2*x')

    def init_reg(self):
        pass


#Alias
TikhonovReg3D = Regularizer3D


class TikhonovDiffReg3D(Regularizer3D):
    """ Similar to basic L2 regularizer, but with distance from prior.

    """

    x0 = Array

    def __init__(self, x0, nthreads=8):
        self.x0 = x0.copy()
        ne.set_num_threads(nthreads)

    def reg_func(self, x):
        x0 = self.x0
        return ne.evaluate('sum((x - x0)**2)')


    def reg_deriv(self, x):
        x0 = self.x0
        return ne.evaluate('2.0*(x-x0)')



class SpatialReg(Regularizer3D):
    """ Abstract mixin for regularizers that know about spatial dimensions.
        Include dim info, spatial kernel setup, and some multithreading
        boilerplate.
    """
    scratch = Array
    kern_sz = Int
    kern_weights = Array
    delta = CFloat(3.0)

    def __init__(self, img_sz, nthreads=8, kern_radius=1, delta=3.0):
        self.nthreads = nthreads
        self.scratch = np.zeros(img_sz)
        self.kern_sz = kern_radius
        self.delta = delta

        # Old Nearest neighbor kernel
        # if kern_radius == 1:
        #     self.init_kern1()
        # elif kern_radius == 2:
        #     self.init_kern2()
        # else:
        #     self.init_bi_tv_kern()

        #optimized kernel
        self.init_bi_tv_kern(radius = kern_radius)
        self.init_reg()

    def init_kern1(self):
        g = np.zeros(4)
        g[1] = 1.0/6.0

        self.kern_weights = g


    def init_kern2(self):

        kern2 = np.zeros(13)
        kern2[1] = 1.0/12.0
        kern2[2] = 1.0/24.0
        kern2[3] = 1/36.0;
        kern2[4] = 1/48.0;

        kern2 = kern2 / 1.34722

        self.kern_weights = kern2


    def init_bi_tv_kern(self, radius=4):
        # Needs to be 3*kern_sz^2+1 since we index with squared distance
        g = np.zeros(49)
        g[1] = 8.58513e-3;
        g[4] = 5.63861e-3;
        g[9] = 3.12348e-3;
        g[16] = 9.0526e-4;
        g[2] = 5.96762e-3;
        g[5] = 4.08198e-3;
        g[10] = 2.74016e-3;
        g[17] = 1.44567e-3;
        g[8] = 3.91213e-3;
        g[13] = 2.68139e-3;
        g[18] = 1.3644e-3;
        g[3] = 7.36271e-3;
        g[6] = 4.12361e-3;
        g[11] = 2.78088e-3;
        g[14] = 2.24185e-3;
        g[12] = 2.26886e-3;

        #adjust normalization if we use a smaller convolution radius
        if radius==1:
            g = g * 5.493784058
        elif radius==2:
            g = g * 1.808809946
        elif radius==3:
            g = g * 1.078601393

        self.kern_weights = g


    # helper functions for multi threading common to all spatial regs
    # Based on Anaconda demo:
    def make_multithread(self, inner_func, numthreads):
        def func_mt(*args):
            length = args[0].shape[0]
            ksz = args[2]
            computelen = length-2*ksz

            chunklen = (computelen) // numthreads

            res = np.zeros(numthreads+1)
            ranges = [(ksz + chunklen*i, ksz + chunklen*(i+1)) for i in range(0,numthreads)]
            #print(ranges)

            # You should make sure inner_func is compiled at this point, because
            # the compilation must happen on the main thread. This is the case
            # in this example because we use jit().
            threads = [threading.Thread(target=inner_func, args= args + (res,) +  rng + (cnt,) )
                       for cnt,rng in enumerate(ranges)]
            for thread in threads:
                thread.start()

            # the main thread handles the last chunk
            #TODO CHECK

            if (ksz + chunklen*numthreads) < (length-ksz):
                lastargs = args + (res,) + (ksz + chunklen*numthreads, length-ksz) + (numthreads,)
                #print (ksz + chunklen*numthreads, length-ksz)
                inner_func(*lastargs)

            for thread in threads:
                thread.join()

            return np.sum(res)

        return func_mt

    def make_grad_multithread(self, inner_func, numthreads):
        def func_mt(*args):
            length = args[0].shape[0]
            ksz = args[2]
            computelen = length-2*ksz

            chunklen = (computelen) // numthreads


            ranges = [(ksz + chunklen*i, ksz + chunklen*(i+1)) for i in range(0,numthreads)]
            #print(ranges)

            # You should make sure inner_func is compiled at this point, because
            # the compilation must happen on the main thread. This is the case
            # in this example because we use jit().
            threads = [threading.Thread(target=inner_func, args= args + rng)
                       for rng in ranges]
            for thread in threads:
                thread.start()

            # the main thread handles the last chunk
            #TODO CHECK
            if (ksz + chunklen*numthreads) < (length-ksz):
                lastargs = args + (ksz + chunklen*numthreads, length-ksz)
                #print (ksz + chunklen*numthreads, length-ksz)
                inner_func(*lastargs)

            for thread in threads:
                thread.join()

        return func_mt


class ParallelHuber2ClassReg3D(SpatialReg):
    """
    Two-class huber penalty. Huber l1/l2 crossover is at delta (huber_scale).
    Classes are assumed to be interleaved (e.g. datashape is (nz, ny, nx, 2))

    """

    def init_reg(self):
        # JITTED version of huber penalty.
        huber_sig = void(double[:,:,:,:], double[:], int64, double, double[:], int64, int64, int64)
        huber_jitted = jit(huber_sig, nopython=True)(self.huber3d)

        # JITTED version of huber gradient.
        ghuber_sig = void(double[:,:,:,:], double[:], int64, double, double[:,:,:,:], int64, int64)
        ghuber_jitted = jit(ghuber_sig, nopython=True)(self.ghuber3d)

        # multi_* are our parallelized versions
        # this wraps the calculation in a multi threaded evaluator
        self.multi_ghuber = self.make_multithread(ghuber_jitted, self.nthreads)
        self.multi_huber = self.make_grad_multithread(huber_jitted, self.nthreads)


    def reg_func(self, x):
        return self.multi_huber(x, self.kern_weights, self.kern_sz, self.delta)


    def reg_deriv(self, x, inout):
        self.multi_ghuber(x, self.kern_weights, self.kern_sz, self.delta, inout)


    @staticmethod
    def huber3d(imgs, weights, kernel_sz, delta, res, start, stop, tidx):
        threadstate = savethread()
        n_slices, n_rows, n_cols, n_channels = imgs.shape
        max_sz = weights.shape[0]

        assert(max_sz >= kernel_sz*kernel_sz)


        accum = 0

        #TODO: look at cost of visiting all the locations where
        # weights are zero (e.g. corners of the domain)
        # Maybe accelerate via pre-computing the indices?
        # Surely checking weights isn't worth the cost of
        # all the branching
        #Alternatively, find bounds as a function of zpos and only check
        # in outermost loop
        #Alternatively, change data storage order to improve locality
        #Best still: write this in C or Cuda...

        for ss in range(start, stop):
            for rr in range(kernel_sz, n_rows - kernel_sz ):
                for cc in range(kernel_sz, n_cols - kernel_sz ):

                    lvals = imgs[ss, rr, cc, :]

                    # loop over kernel

                    for zz in range(-kernel_sz, kernel_sz + 1):
                        for hh in range(-kernel_sz, kernel_sz + 1):
                            for ww in range(-kernel_sz, kernel_sz + 1):

                                widx = hh*hh+ww*ww+zz*zz

                                imgvals = imgs[ss + zz, rr + hh, cc + ww, :]
                                zsq = 0

                                # This used to be a loop over channels, but an
                                # explicit unroll turned out to be a lot faster
                                # I guess numba isn't unrolling by default.
                                # Sadly, this means we need different functions
                                # specialized for # of channels
                                diffs = imgvals[0] - lvals[0]
                                zsq += diffs*diffs

                                diffs = imgvals[1] - lvals[1]
                                zsq += diffs*diffs

                                w = weights[widx]

                                tmp = 0
                                z = sqrt(zsq)

                                # using a continuous approximation of huber,
                                # d^2 (sqrt(1+a^2/d^2) - 1) is about 20% faster
                                # despite the extra sqrt call.
                                if z < delta:
                                    tmp = 0.5*zsq
                                else:
                                    tmp = delta*(z-0.5*delta)

                                accum += tmp * w * 0.5
        res[tidx] = accum
        restorethread(threadstate)



    @staticmethod
    def ghuber3d(imgs, weights, kernel_sz, delta, output, start, stop):
        threadstate = savethread()
        n_slices, n_rows, n_cols, n_channels = imgs.shape
        max_sz = weights.shape[0]

        assert(max_sz >= kernel_sz*kernel_sz)

        eps = 1e-8

        accum = 0


        for ss in range(start, stop):
            for rr in range(kernel_sz, n_rows - kernel_sz ):
                for cc in range(kernel_sz, n_cols - kernel_sz ):

                    lvals = imgs[ss, rr, cc, :]

                    # loop over kernel
                    for zz in range(-kernel_sz, kernel_sz + 1):

                        for hh in range(-kernel_sz, kernel_sz + 1):
                            for ww in range(-kernel_sz, kernel_sz + 1):
                                zsq = 0.0

                                widx = hh*hh+ww*ww+zz*zz

                                imgvals = imgs[ss + zz, rr + hh, cc + ww, :]

                                diffs1 = imgvals[0] - lvals[0]
                                zsq = diffs1*diffs1

                                diffs2 = imgvals[1] - lvals[1]
                                zsq += diffs2*diffs2

                                w = weights[widx]

                                z = sqrt(zsq)

                                rval = 1.0
                                if z >= delta:
                                    rval = delta/z


                                output[ss, rr, cc, 0] += rval*w*diffs1
                                output[ss, rr, cc, 1] += rval*w*diffs2

        restorethread(threadstate)






class Huber2ClassReg3D(SpatialReg):
    """ Implements a single threaded version of huber regularizer for testing """


    def reg_func(self, x):
        return self.huber_single(x, self.kern_weights, self.kern_sz, self.delta)


    def reg_deriv(self, x, inout):
        self.ghuber_single(x, self.kern_weights, self.kern_sz, self.delta, inout)


    @staticmethod
    @autojit(nopython=True)
    def huber_single(imgs, weights, kernel_sz, delta):
        n_slices, n_rows, n_cols, n_channels = imgs.shape
        max_sz = weights.shape[0]

        assert(max_sz >= kernel_sz*kernel_sz)

        accum = 0

        for ss in range(kernel_sz, n_slices - kernel_sz ):
            for rr in range(kernel_sz, n_rows - kernel_sz ):
                for cc in range(kernel_sz, n_cols - kernel_sz ):

                    lvals = imgs[ss, rr, cc, :]

                    # loop over kernel
                    for zz in range(-kernel_sz, kernel_sz + 1):
                        for hh in range(-kernel_sz, kernel_sz + 1):
                            for ww in range(-kernel_sz, kernel_sz + 1):

                                widx = hh*hh+ww*ww+zz*zz

                                imgvals = imgs[ss + zz, rr + hh, cc + ww, :]
                                zsq = 0

                                # explicit unroll turned out to be a lot faster
                                diffs1 = imgvals[0] - lvals[0]
                                zsq += diffs1*diffs1

                                diffs2 = imgvals[1] - lvals[1]
                                zsq += diffs2*diffs2

                                w = weights[widx]

                                tmp = 0

                                z = sqrt(zsq)

                                if z < delta:
                                    tmp = 0.5*zsq
                                else:
                                    tmp = delta*(z-0.5*delta)

                                accum += tmp * w
        return accum

    @staticmethod
    @autojit(nopython=True)
    def ghuber_single(imgs, weights, kernel_sz, delta, output ):
        n_slices, n_rows, n_cols, n_channels = imgs.shape
        max_sz = weights.shape[0]

        assert(max_sz >= kernel_sz*kernel_sz)

        eps = 1e-8

        accum = 0


        for ss in range(kernel_sz, n_slices - kernel_sz):
            for rr in range(kernel_sz, n_rows - kernel_sz ):
                for cc in range(kernel_sz, n_cols - kernel_sz ):

                    lvals = imgs[ss, rr, cc, :]

                    # loop over kernel
                    for zz in range(-kernel_sz, kernel_sz + 1):

                        for hh in range(-kernel_sz, kernel_sz + 1):
                            for ww in range(-kernel_sz, kernel_sz + 1):
                                zsq = 0
                                diffs = 0

                                widx = hh*hh+ww*ww+zz*zz

                                imgvals = imgs[ss + zz, rr + hh, cc + ww, :]

                                diffs1 = imgvals[0] - lvals[0]
                                zsq = diffs1*diffs1

                                diffs2 = imgvals[1] - lvals[1]
                                zsq += diffs2*diffs2

                                w = weights[widx]

                                z = sqrt(zsq)

                                rval = 1.0
                                if z >= delta:
                                    rval = delta/z


                                output[ss, rr, cc, 0] += rval*w*diffs1
                                output[ss, rr, cc, 1] += rval*w*diffs2





class ParallelWelsch2ClassReg3D(SpatialReg):
    """
    Two-class welsch penalty. welsch tuning crossover is at delta .
    Classes are assumed to be interleaved (e.g. datashape is (nz, ny, nx, 2))

    """

    def init_reg(self):
        # JITTED version of huber penalty.
        welsch_sig = void(double[:,:,:,:], double[:], int64, double, double[:], int64, int64, int64)
        welsch_jitted = jit(welsch_sig, nopython=True)(self.welsch3d)

        # JITTED version of huber gradient.
        gwelsch_sig = void(double[:,:,:,:], double[:], int64, double, double[:,:,:,:], int64, int64)
        gwelsch_jitted = jit(gwelsch_sig, nopython=True)(self.gwelsch3d)


        # multi_* are our parallelized versions
        # this wraps the calculation in a multi threaded evaluator
        self.multi_gwelsch = self.make_grad_multithread(gwelsch_jitted, self.nthreads)
        self.multi_welsch = self.make_multithread(welsch_jitted, self.nthreads)


    def reg_func(self, x):
        return self.multi_welsch(x, self.kern_weights, self.kern_sz, self.delta)


    def reg_deriv(self, x, inout):
        self.multi_gwelsch(x, self.kern_weights, self.kern_sz, self.delta, inout)


    # static seems to be easier to jit()
    @staticmethod
    def welsch3d(imgs, weights, kernel_sz, delta, res, start, stop, tidx):
        threadstate = savethread()
        n_slices, n_rows, n_cols, n_channels = imgs.shape
        max_sz = weights.shape[0]

        assert(max_sz >= kernel_sz*kernel_sz)

        accum = 0

        ksq = delta**2

        for ss in range(start, stop):
            for rr in range(kernel_sz, n_rows - kernel_sz ):
                for cc in range(kernel_sz, n_cols - kernel_sz ):

                    lvals = imgs[ss, rr, cc, :]

                    # loop over kernel
                    #TODO: look at cost of visiting all the locations where
                    # weights are zero (e.g. corners of the domain)
                    # Maybe accelerate via pre-computing the indices?
                    # Surely checking weights isn't worth the cost of
                    # all the branching
                    #Alternativley, find bounds as a function of zpos and only check
                    # in outermost loop
                    for zz in range(-kernel_sz, kernel_sz + 1):
                        for hh in range(-kernel_sz, kernel_sz + 1):
                            for ww in range(-kernel_sz, kernel_sz + 1):

                                widx = hh*hh+ww*ww+zz*zz

                                imgvals = imgs[ss + zz, rr + hh, cc + ww, :]
                                zsq = 0

                                # This used to be a loop over channels, but an
                                # explicit unroll turned out to be a lot faster
                                # I guess numba isn't doing the right thing.
                                # Sadly, this means we need different functions
                                # specialized for # of channels
                                diffs = imgvals[0] - lvals[0]
                                zsq += diffs*diffs

                                diffs = imgvals[1] - lvals[1]
                                zsq += diffs*diffs

                                w = weights[widx]


                                tmp = 0.5*ksq * (1.0 - exp(-zsq/ksq))

                                accum += tmp * w
        res[tidx] = accum
        restorethread(threadstate)


    @staticmethod
    def gwelsch3d(imgs, weights, kernel_sz, delta, output, start, stop):
        threadstate = savethread()
        n_slices, n_rows, n_cols, n_channels = imgs.shape
        max_sz = weights.shape[0]

        assert(max_sz >= kernel_sz*kernel_sz)

        eps = 1e-8

        accum = 0
        ksq = delta**2

        for ss in range(start, stop):
            for rr in range(kernel_sz, n_rows - kernel_sz ):
                for cc in range(kernel_sz, n_cols - kernel_sz ):

                    lvals = imgs[ss, rr, cc, :]

                    # loop over kernel
                    for zz in range(-kernel_sz, kernel_sz + 1):

                        for hh in range(-kernel_sz, kernel_sz + 1):
                            for ww in range(-kernel_sz, kernel_sz + 1):
                                zsq = 0.0

                                widx = hh*hh+ww*ww+zz*zz

                                imgvals = imgs[ss + zz, rr + hh, cc + ww, :]

                                diffs1 = imgvals[0] - lvals[0]
                                zsq = diffs1*diffs1

                                diffs2 = imgvals[1] - lvals[1]
                                zsq += diffs2*diffs2

                                w = weights[widx]

                                z = sqrt(zsq)

                                rval =  2.0*exp(-zsq/ksq)


                                output[ss, rr, cc, 0] += rval*w*diffs1
                                output[ss, rr, cc, 1] += rval*w*diffs2

        restorethread(threadstate)

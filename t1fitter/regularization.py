
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


savethread = pythonapi.PyEval_SaveThread
savethread.argtypes = []
savethread.restype = c_void_p

restorethread = pythonapi.PyEval_RestoreThread
restorethread.argtypes = [c_void_p]
restorethread.restype = None


class Regularizer3D(HasTraits):
    """ Basic form of regularizers expose reg_func and reg_deriv


    """

    def __init__(self, nthreads=4):
        self.log = logging.get_logger('T1Fit')
        ne.set_num_threads(nthreads)


    def reg_func(self, x):
        return ne.evaluate('sum(x**2)')

    def reg_deriv(self, x):
        return ne.evaluate('2*x')


#Alias
TikhonovReg3D = Regularizer3D


class TikhonovDiffReg3D(Regularizer3D):
    """ Similar to basic L2 regularizer, but with distance from prior.

    """

    x0 = Array

    def __init__(self, x0, nthreads=4):
        self.x0 = x0
        ne.set_num_threads(nthreads)

    def reg_func(self, x):
        x0 = self.x0
        return ne.evaluate('sum((x - x0)**2)')


    def reg_deriv(self, x):
        x0 = self.x0
        grad = ne.evaluate('2*(x-x0)')
        return grad



class ParallelHuber2ClassReg3D(Regularizer3D):
    """
    Two-class huber penalty. Huber l1/l2 crossover is at 1.0, so scale the data.
    Classes are assumed to be interleaved (e.g. datashape is (nz, ny, nx, 2))

    """

    scratch = Array
    kern_sz = Int
    kern_weights = Array
    hub_scale = CFloat(3.0)

    def __init__(self, img_sz, nthreads=4, kern_radius=1, hub_scale=3.0):
        self.nthreads = nthreads
        self.scratch = np.zeros(img_sz)
        self.kern_sz = kern_radius
        self.hub_scale = hub_scale

        if kern_radius == 1:
            self.init_kern1()
        elif kern_radius == 2:
            self.init_kern2()
        else:
            self.init_bi_tv_kern()

        # Do jitting to set up multithreaded huber evaluator
        self.init_reg()


    def init_kern1(self):
        g = np.zeros(4)
        g[1] = 1.0/6.0

        self.kern_weights = g


    def init_kern2(self):

        kern2 = np.zeros(9)
        kern2[1] = 1.0/12.0
        kern2[2] = 1.0/24.0
        kern2[3] = 1/36.0;
        kern2[4] = 1/48.0;

        kern2 = kern2 / 1.34722;

        self.kern_weights = kern2


    def init_bi_tv_kern(self):
        g = np.zeros(64)
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

        self.kern_weights = g


    def init_reg(self):
        # JITTED version of huber penalty.
        huber_sig = void(double[:,:,:,:], double[:], int64, double, double[:], int64, int64, int64)
        huber_jitted = jit(huber_sig, nopython=True)(self.huber3d)

        # JITTED version of huber gradient.
        ghuber_sig = void(double[:,:,:,:], double[:], int64, double, double[:,:,:,:], int64, int64)
        ghuber_jitted = jit(ghuber_sig, nopython=True)(self.ghuber3d)

        '''
            multi_huber is our parallelized version
        '''
        self.multi_ghuber = self.make_grad_multithread(ghuber_jitted, self.nthreads)
        self.multi_huber = self.make_hub_multithread(huber_jitted, self.nthreads)


    def reg_func(self, x):
        return self.multi_huber(x, self.kern_weights, self.kern_sz, self.hub_scale)


    def reg_deriv(self, x, inout):
        self.multi_ghuber(x, self.kern_weights, self.kern_sz, self.hub_scale, inout)


    @staticmethod
    def huber3d(imgs, weights, kernel_sz, delta, res, start, stop, tidx):
        threadstate = savethread()
        n_slices, n_rows, n_cols, n_channels = imgs.shape
        max_sz = weights.shape[0]

        assert(max_sz >= kernel_sz*kernel_sz)


        accum = 0

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
                                # I guess numba isn't doing the right thing
                                # Sadly, this means we need different functions
                                # if we go to more classes
                                diffs = imgvals[0] - lvals[0]
                                zsq += diffs*diffs

                                diffs = imgvals[1] - lvals[1]
                                zsq += diffs*diffs

                                w = weights[widx]

                                tmp = 0

                                if zsq < delta:
                                    tmp = 0.5*zsq
                                else:
                                    tmp = delta*(np.sqrt(zsq)-0.5*delta)

                                accum += tmp * w
        res[tidx] = accum
        restorethread(threadstate)




    def make_hub_multithread(self, inner_func, numthreads):
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
                                zsq = 0
                                diffs = 0

                                widx = hh*hh+ww*ww+zz*zz

                                imgvals = imgs[ss + zz, rr + hh, cc + ww, :]

                                diffs1 = imgvals[0] - lvals[0]
                                zsq = diffs1*diffs1

                                diffs2 = imgvals[1] - lvals[1]
                                zsq += diffs2*diffs2

                                w = weights[widx]

                                #sum over channels
                                z = np.sqrt(zsq)

                                rsq = 0.0

                                if z >= delta:
                                    rsq = delta/z
                                    w *= rsq

                                output[ss, rr, cc, 0] += w*diffs1
                                output[ss, rr, cc, 1] += w*diffs2

        restorethread(threadstate)



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



class Huber2ClassReg3D(Regularizer3D):
    """ Implements a single threaded version of huber regularizer for testing """

    scratch = Array
    kern_sz = Int
    hub_scale = CFloat(3.0)
    kern_weights = Array

    def __init__(self, img_sz, kern_radius=1 ):

        self.kern_sz = kern_radius
        self.scratch = np.zeros(img_sz)

        if kern_radius == 1:
            self.init_kern1()
        elif kern_radius == 2:
            self.init_kern2()
        else:
            self.init_bi_tv_kern()


    def reg_func(self, x):
        return self.huber_single(x, self.kern_weights, self.kern_sz, self.hub_scale)


    def reg_deriv(self, x, inout):
        self.scratch *= 0.0;
        self.ghuber_single(x, self.kern_weights, self.kern_sz, self.hub_scale, self.scratch)
        return self.scratch



    def init_kern1(self):
        g = np.zeros(4)
        g[1] = 1.0/6.0

        self.kern_weights = g


    def init_kern2(self):

        kern2 = np.zeros(9)
        kern2[1] = 1.0/12.0
        kern2[2] = 1.0/24.0
        kern2[3] = 1/36.0;
        kern2[4] = 1/48.0;

        kern2 = kern2 / 1.34722;

        self.kern_weights = kern2


    def init_bi_tv_kern(self):
        g = np.zeros(64)
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

        self.kern_weights = g

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
                                diffs = imgvals[0] - lvals[0]
                                zsq += diffs*diffs

                                diffs = imgvals[1] - lvals[1]
                                zsq += diffs*diffs

                                w = weights[widx]

                                tmp = 0

                                if zsq < delta:
                                    tmp = 0.5*zsq
                                else:
                                     tmp = delta*(np.sqrt(zsq)-0.5*delta)

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

                                #sum over channels
                                z = np.sqrt(zsq)

                                rsq = 0.0

                                if z >= delta:
                                    rsq = delta/z
                                    w *= rsq

                                output[ss, rr, cc, 0] += w*diffs1
                                output[ss, rr, cc, 1] += w*diffs2

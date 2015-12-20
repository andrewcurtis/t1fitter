import sys

import numpy as np
import sh



l1s = [ 2e-4, 6e-4, 1e-3, 1.4e-3 ]
l2s = [ 5e-6, ]
ksz = [ 1, 2 ]
deltas = [ 0.3, 0.5, 0.7 ]

l1mode = ['huber','welsch']

simcombs = [ (l1, l2, k, h, l1m) for l1 in l1s for l2 in l2s for k in ksz for h in deltas for l1m in l1mode]




def main(rnum):

    cmdline = '--addvol /imaging/home/MRI/acurtis/t1fit/dec3dat/procfa14_bet_crop.nii.gz 14.0 10.0   --addvol /imaging/home/MRI/acurtis/t1fit/dec3dat/procfa3_align_bet_crop.nii.gz 3.0 10.0   --out /imaging/scratch/MRI/acurtis/tmp/welsch2/   --b1vol  /imaging/home/MRI/acurtis/t1fit/dec3dat/b1_mos.nii.gz   --mask /imaging/home/MRI/acurtis/t1fit/dec3dat/procfa14_bet_mask_crop.nii.gz     --fit_method nlreg --startmode zero   --smooth 25   --l2mode smooth_vfa --nthreads 6 --b1scale 1.12 --descriptive_names --l1lam {}  --l2lam {} --kern_radius {} --delta {} --l1mode {}'.format(*simcombs[rnum])

    print(cmdline)
    sh.t1fit(cmdline.split())



if __name__ == '__main__':
    main(int(sys.argv[1]))

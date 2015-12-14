import sys
from t1fitter import t1fit_cli

import numpy as np



l1s = [  4e-4, 6e-4, 8e-4,1e-3  ]
l2s = [   5e-6,  ]
ksz = [1,2,4 ]



simcombs = [ (l1, l2, k) for l1 in l1s for l2 in l2s for k in ksz ]




def main(rnum):

    cmdline = '--addvol /imaging/home/MRI/acurtis/t1fit/dec3dat/procfa14_bet_crop.nii.gz 14.0 10.0   --addvol /imaging/home/MRI/acurtis/t1fit/dec3dat/procfa3_align_bet_crop.nii.gz 3.0 10.0   --out /imaging/scratch/MRI/acurtis/tmp/   --b1vol  /imaging/home/MRI/acurtis/t1fit/dec3dat/b1_mos.nii.gz   --mask /imaging/home/MRI/acurtis/t1fit/dec3dat/procfa14_bet_mask_crop.nii.gz    --huber_scale 0.20 --fit_method nlreg --startmode zero   --smooth 25   --l2mode smooth_vfa --nthreads 12 --b1scale 1.12 --descriptive_names --l1lam {}  --l2lam {} --kern_radius {}'.format(*simcombs[rnum])

    print(cmdline)
    t1fit_cli(cmdline.split())



if __name__ == '__main__':
    main(int(sys.argv[1]))

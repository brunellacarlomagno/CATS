import numpy as np
import cv2
import proper
from cats.cats_simus  import *

def vortex(wfo, CAL, charge, f_lens, Debug_print):

    n = int(proper.prop_get_gridsize(wfo))
    ofst = 0 # no offset
    ramp_sign = 1 #sign of charge is positive
    #sampling = n
    ramp_oversamp = 11. # vortex is oversampled for a better discretization

    if charge!=0:
        if CAL==1: # create the vortex for a perfectly circular pupil
            if (Debug_print == True):
                print ("CAL:1, charge ", charge)
            writefield('zz_psf', wfo.wfarr) # write the pre-vortex field
            nramp = int(n*ramp_oversamp) #oversamp
            # create the vortex by creating a matrix (theta) representing the ramp (created by atan 2 gradually varying matrix, x and y)
            y1 = np.ones((nramp,), dtype=np.int)
            y2 = np.arange(0, nramp, 1.) - (nramp/2) - int(ramp_oversamp)/2
            y = np.outer(y2, y1)
            x = np.transpose(y)
            theta = np.arctan2(y,x)
            x = 0
            y = 0
            #vvc_tmp_complex = np.array(np.zeros((nramp,nramp)), dtype=complex)
            #vvc_tmp_complex.imag = ofst + ramp_sign*charge*theta
            #vvc_tmp = np.exp(vvc_tmp_complex)
            vvc_tmp = np.exp(1j*(ofst + ramp_sign*charge*theta))
            theta = 0
            vvc_real_resampled = cv2.resize(vvc_tmp.real, (0,0), fx=1/ramp_oversamp, fy=1/ramp_oversamp, interpolation=cv2.INTER_LINEAR) # scale the pupil to the pupil size of the simualtions
            vvc_imag_resampled = cv2.resize(vvc_tmp.imag, (0,0), fx=1/ramp_oversamp, fy=1/ramp_oversamp, interpolation=cv2.INTER_LINEAR) # scale the pupil to the pupil size of the simualtions
            vvc = np.array(vvc_real_resampled, dtype=complex)
            vvc.imag = vvc_imag_resampled
            vvcphase = np.arctan2(vvc.imag, vvc.real) # create the vortex phase
            vvc_complex = np.array(np.zeros((n,n)), dtype=complex)
            vvc_complex.imag = vvcphase
            vvc = np.exp(vvc_complex)
            vvc_tmp = 0.
            writefield('zz_vvc', vvc) # write the theoretical vortex field
            wfo0 = wfo
            proper.prop_multiply(wfo, vvc)
            proper.prop_propagate(wfo, f_lens, 'OAP2')
            proper.prop_lens(wfo, f_lens)
            proper.prop_propagate(wfo, f_lens, 'forward to Lyot Stop')
            proper.prop_circular_obscuration(wfo, 1., NORM=True) # null the amplitude iside the Lyot Stop
            proper.prop_propagate(wfo, -f_lens) # back-propagation
            proper.prop_lens(wfo, -f_lens)
            proper.prop_propagate(wfo, -f_lens)
            writefield('zz_perf', wfo.wfarr) # write the perfect-result vortex field
            wfo = wfo0
        else:
            if (Debug_print == True):
                print ("CAL:0, charge ", charge)
            vvc = readfield('zz_vvc') # read the theoretical vortex field
            vvc = proper.prop_shift_center(vvc)
            scale_psf = wfo._wfarr[0,0]
            psf_num = readfield('zz_psf') # read the pre-vortex field
            psf0 = psf_num[0,0]
            psf_num = psf_num/psf0*scale_psf
            perf_num = readfield('zz_perf') # read the perfect-result vortex field
            perf_num = perf_num/psf0*scale_psf
            wfo._wfarr = (wfo._wfarr - psf_num)*vvc + perf_num # the wavefront takes into account the real pupil with the perfect-result vortex field

    return

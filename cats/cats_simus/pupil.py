import numpy as np
import cv2
import proper
from astropy.io import fits
import os

def pupil(wfo, CAL, npupil, diam, r_obstr, spiders_width, spiders_angle, pupil_file, missing_segments_number, Debug, Debug_print):
    
    n = int(proper.prop_get_gridsize(wfo))
    
    if (missing_segments_number == 0):
        if (isinstance(pupil_file, (list, tuple, np.ndarray)) == True):
            pupil = pupil_file
            pupil_pixels = (pupil.shape)[0]## fits file size
            scaling_factor = float(npupil)/float(pupil_pixels) ## scaling factor between the fits file size and the pupil size of the simulation
            if (Debug_print==True):
                print ("scaling_factor: ", scaling_factor)
            pupil_scale = cv2.resize(pupil.astype(np.float32), (0,0), fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR) # scale the pupil to the pupil size of the simualtions
            if (Debug_print==True):
                print ("pupil_resample", pupil_scale.shape)
            pupil_large = np.zeros((n,n)) # define an array of n-0s, where to insert the pupuil
            if (Debug_print==True):
                print("n: ", n)
                print("npupil: ", npupil)
            pupil_large[int(n/2)+1-int(npupil/2)-1:int(n/2)+1+int(npupil/2),int(n/2)+1-int(npupil/2)-1:int(n/2)+1+int(npupil/2)] =pupil_scale # insert the scaled pupil into the 0s grid

        proper.prop_circular_aperture(wfo, diam/2) # create a wavefront with a circular pupil
    
        if CAL==0: # CAL=1 is for the back-propagation
            if (isinstance(pupil_file, (list, tuple, np.ndarray)) == True):
                proper.prop_multiply(wfo, pupil_large) # multiply the saved pupil
            else:
                proper.prop_circular_obscuration(wfo, r_obstr, NORM=True) # create a wavefront with a circular central obscuration
            if (spiders_width!=0):
                for iter in range(0,len(spiders_angle)):
                    proper.prop_rectangular_obscuration(wfo, spiders_width, 2*diam,ROTATION=spiders_angle[iter]) # define the spiders
                        
    else:
        PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))
        if (missing_segments_number == 1):
            pupil = fits.getdata(PACKAGE_PATH+'/ELT_2048_37m_11m_5mas_nospiders_1missing_cut.fits')
        if (missing_segments_number == 2):
            pupil = fits.getdata(PACKAGE_PATH+'/ELT_2048_37m_11m_5mas_nospiders_2missing_cut.fits')
        if (missing_segments_number == 4):
            pupil = fits.getdata(PACKAGE_PATH+'/ELT_2048_37m_11m_5mas_nospiders_4missing_cut.fits')
        if (missing_segments_number == 7):
            pupil = fits.getdata(PACKAGE_PATH+'/ELT_2048_37m_11m_5mas_nospiders_7missing_1_cut.fits')

        pupil_pixels = (pupil.shape)[0]## fits file size
        scaling_factor = float(npupil)/float(pupil_pixels) ## scaling factor between the fits file size and the pupil size of the simulation
        if (Debug_print==True):
            print ("scaling_factor: ", scaling_factor)
        pupil_scale = cv2.resize(pupil.astype(np.float32), (0,0), fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR) # scale the pupil to the pupil size of the simualtions
        if (Debug_print==True):
            print ("pupil_resample", pupil_scale.shape)
        pupil_large = np.zeros((n,n)) # define an array of n-0s, where to insert the pupuil
        if (Debug_print==True):
            print("n: ", n)
            print("npupil: ", npupil)
        pupil_large[int(n/2)+1-int(npupil/2)-1:int(n/2)+1+int(npupil/2),int(n/2)+1-int(npupil/2)-1:int(n/2)+1+int(npupil/2)] =pupil_scale # insert the scaled pupil into the 0s grid

            
        if CAL==0: # CAL=1 is for the back-propagation
            proper.prop_multiply(wfo, pupil_large) # multiply the saved pupil
            if (spiders_width!=0):
                for iter in range(0,len(spiders_angle)):
                    proper.prop_rectangular_obscuration(wfo, spiders_width, 2*diam,ROTATION=spiders_angle[iter]) # define the spiders



    return

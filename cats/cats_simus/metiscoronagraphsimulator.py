from __future__ import absolute_import

import numpy as np

import proper
import math

from astropy.io import fits
import os
import sys

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(PACKAGE_PATH)

def metiscoronagraphsimulator(n,lam, pixelsize, prefix, path, diam=37., r_obstr=0.3, f_lens=658.6, pupil_file=0,  spiders_width=0.60, spiders_angle=[0., 60., 120.], charge=0, LS_parameters=[0.0, 0.0, 0.0], amplitude_apodizer_file=0,phase_apodizer_file=0,LS_amplitude_apodizer_file=0,LS_phase_apodizer_file=0,  TILT=[0.0, 0.0], atm_screen=0., missing_segments_number=0, apodizer_misalignment=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], LS_misalignment=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], Island_Piston=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], NCPA=0., NoCoro_psf=0, Offaxis_psf=0, ELT_circ=True, Vortex=False, Back=False, RAVC=False, LS=False, Debug=True, Debug_print=True, Norm_max=True, Norm_flux1=False):
    """MCS.
        
        Parameters
        ----------
        n : integer
        Grid size
        
        lam : float in microns
        Wavelength
        
        pixelsize : float in mas
        Pixelsize in the detector plane
        
        prefix : string
        Prefix for the fits file writing
        
        path : string
        path where to save the outputs
        
        
        Returns
        -------
        None
        
        
        PASSVALUE
        ---------
        diam: 37. --> float, meters
        when both ELT_circ and other_pupil are False, the diameter has to be defined here
        
        r_obstr: 0.3 -->float, ratio
        when both ELT_circ and other_pupil are False, the central obstruction has to be defined here
        
        f_lens: 658.6 --> float, meters
        focal distance
        
        pupil_file: 0 --> float matrix
        it contains the pupil in case the 'other_pupil' is called
        
        spiders_width: 0.60 --> float, meters
        the spiders width has to be defined here, when the 'ELT_circ' is called (the fits file has no spiders) and when both ELT_circ and other_pupil are False and spiders are present
        
        spiders_angle: [0., 60., 120.] --> float, degrees
        the spiders rotation angle has to be defined here, when the 'ELT_circ' is called (the fits file has no spiders) and when both ELT_circ and other_pupil are False and spiders are present
        
        charge: 0 -->integer
        it indicates the charge of the vortex coronagraph ('Vortex' has to be called)
        
        LS_parameters : [0.0, 0.0, 0.0] --> float [percentage of the outer diamter, percentage of the outer diamter to add to the central obstruction/apodizer diameter, meters for the spiders]
            it indicates the values of the Lyot Stop ('LS' has to be called)
        
        amplitude_apodizer_file:0, float matrix
        it contains the pupil amplitude apodizer file in case the 'amplitude_apodizer' is called
        
        phase_apodizer_file:0, float matrix
        it contains the pupil phase apodizer file in case the 'phase_apodizer' is called
        
        LS_amplitude_apodizer_file:0, float matrix
        it contains the Lyot Stop amplitude apodizer file in case the 'LS_amplitude_apodizer' is called
        
        LS_phase_apodizer_file:0, float matrix
        it contains the yot Stop phase apodizer file in case the 'LS_phase_apodizer' is called
        
        TILT: [0.0, 0.0] --> float, lambda/D
        Tilt values : [xtilt, ytilt] !!! PROPER CHANGE THE DIRECTION BETWEEN XTILT AND YTILT !!!!
        
        atm_screen: 0 --> float matrix, microns
        it contains the atm phase screen
        
        missing_segments_number: 0 --> integer
        it contains the number of missing segments ('missing_segments' has to be called)
        
        apodizer_misalignment: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] --> float, percentage
        it indicated the pupil apodizer misalignment: the first 3 are for an amplitude apodizer misalignment ('pupil_amplitude_apodizer_misalignment' has to be called) and the last 3 are for a phase apodizer misalignment ('pupil_phase_apodizer_misalignment' has to be called)
        
        LS_misalignment: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] --> float, percentage
        it indicated the LS apodizer misalignment: the first 3 are for an amplitude apodizer misalignment ('LS_amplitude_apodizer_misalignment' has to be called) and the last 3 are for a phase apodizer misalignment ('LS_phase_apodizer_misalignment' has to be called)
        
        Island_Piston: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] --> float, microns
        it indicated the values for the petals differential pistons. Since only 'ELT_circ' is supported for now, it needs 6 values for the 6 petals
        
        NCPA: 0 --> float matrix, microns
        it contains the NCPA matrix in microns
        
        NoCoro_psf: 0 --> float matrix, non coronagraphic psf
        
        Offaxis_psf: 0 --> float matrix, off-axis psf
        
        
        
        kwargs
        ----------------
        ELT_circ : bool
        If the pupil is the circularised ELT pupil in the file "ELT_2048_37m_11m_5mas_nospiders_cut.fits"

        Vortex : bool
        If the coronagraph is a vortex coronagraph, it needs the passvalue 'charge', which contains the charge of the vortex
        
        Back : bool
        If the back-propagation for the vortex coronagraph is needed: the back rpopagation depends on the grid and the pupil size (only the external diamater).
        
        RAVC: bool
        If the coronagraph is a RAVC (to work properly, also the vortex parameter has to be True)
        
        LS: bool
        If the Lyot Stop is present, it needs the passvalue 'LS_parameters' to be completed with the proper values
        
        Debug: bool
        it writes all the fits file, with the prefix
        
        Debug_print: bool
        it print several values
        
        Norm_max: bool
        it normalizes the psfs with respect to the max of the non-coronagraphic psf
        
        Norm_flux1: bool
        it normalizes the psfs with respect to the max of the non-coronagraphic psf, where the total flux has been put to 1

        """

    if (ELT_circ == True):
        PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))
        pupil_file = fits.getdata(PACKAGE_PATH+'/ELT_2048_37m_11m_5mas_nospiders_cut.fits')
        diam = 37.

    beam_ratio = pixelsize*4.85e-9/(lam*1e-6/diam)
    npupil = math.ceil(n*beam_ratio) # compute the pupil size --> has to be ODD (proper puts the center in the up right pixel next to the grid center)
    if npupil % 2 == 0:
        npupil = npupil +1
        #if ("Debug_print" in kwargs and kwargs["Debug_print"]):
    if (Debug_print == True):
        print ("npupil: ", npupil)

    TILT=np.array(TILT)
    apodizer_misalignment=np.array(apodizer_misalignment)
    LS_misalignment=np.array(LS_misalignment)
    Island_Piston=np.array(Island_Piston)


    if (isinstance(NoCoro_psf, (list, tuple, np.ndarray)) != True):
## Non coronagraphic PSF --> simulate a non-coronagraphic psf
        (wfo_noCoro, sampling) = proper.prop_run('telescope', lam, n, PASSVALUE={'prefix':prefix, 'path':path, 'charge':0, 'CAL':0, 'diam':diam, 'spiders_width':spiders_width, 'spiders_angle':spiders_angle, 'beam_ratio': beam_ratio, 'f_lens':f_lens, 'npupil':npupil, 'r_obstr':r_obstr, 'pupil_file':pupil_file, 'phase_apodizer_file':0, 'amplitude_apodizer_file':0, 'TILT':[0.,0.],'LS':False,'RAVC':False, 'LS_phase_apodizer_file':0, 'LS_amplitude_apodizer_file':0,'LS_parameters':[0., 0., 0.], 'atm_screen':0, 'missing_segments_number':0, 'apodizer_misalignment':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'LS_misalignment':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'Island_Piston':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'NCPA':0, 'Debug_print':Debug_print,'Debug':Debug}, QUIET=True)
        NoCoro_psf = (abs(wfo_noCoro))**2
        fits.writeto(path+prefix+'_psf_noCoro_nonorm.fits', NoCoro_psf[int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio),int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio)], overwrite=True)
        #if ("Norm_max" in kwargs and kwargs["Norm_max"]):
        if (Norm_max == True):
            psf_noCoro_maxnorm = NoCoro_psf/np.max(NoCoro_psf)
            fits.writeto(path+prefix+'_psf_noCoro_maxnorm.fits', psf_noCoro_maxnorm[int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio),int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio)], overwrite=True)
        #if ("Norm_flux1" in kwargs and kwargs["Norm_flux1"]):
        if (Norm_flux1 == True):
            psf_noCoro_flux1norm = NoCoro_psf/sum(sum(NoCoro_psf))
            fits.writeto(path+prefix+'_psf_noCoro_flux1norm.fits', psf_noCoro_flux1norm[int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio),int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio)], overwrite=True)

    ## Coronagraphic PSF
    if (Vortex == True):
    
        if (Back == True):
        ## A/R --> simulate a perfect vortex by propagating a perfectly circular pupil through the vortex to the Lyot Stop, null the amplitude inside (as theory requires), then propagating back to the vortex level and save a "modified" vortex, to use in the future simulations
            (wfo_AR, sampling) = proper.prop_run('telescope', lam, n, PASSVALUE={'prefix':prefix, 'path':path, 'charge':charge, 'CAL':1, 'diam':diam, 'spiders_width':0, 'spiders_angle':[0., 0., 0.], 'beam_ratio': beam_ratio, 'f_lens':f_lens, 'npupil':npupil, 'r_obstr':0., 'pupil_file':0, 'phase_apodizer_file':0, 'amplitude_apodizer_file':0, 'TILT':[0.,0.],'LS':False,'RAVC':False, 'LS_phase_apodizer_file':0, 'LS_amplitude_apodizer_file':0,'LS_parameters':[0., 0., 0.], 'atm_screen':0, 'missing_segments_number':0, 'apodizer_misalignment':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'LS_misalignment':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'Island_Piston':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'NCPA':0, 'Debug_print':Debug_print,'Debug':Debug}, QUIET=True)
        
        if (isinstance(Offaxis_psf, (list, tuple, np.ndarray)) != True):
            ## No Vortex Si Mask --> simulate a non-coronagraphic-apodized psf: apodizer and Lyot Stop are present, but not the vortex --> as an off-axis psf
            (wfo_offaxis, sampling) = proper.prop_run('telescope', lam, n, PASSVALUE={'prefix':prefix, 'path':path, 'charge':0, 'CAL':0, 'diam':diam, 'spiders_width':spiders_width, 'spiders_angle':spiders_angle, 'beam_ratio': beam_ratio, 'f_lens':f_lens, 'npupil':npupil, 'r_obstr':r_obstr, 'pupil_file':pupil_file, 'phase_apodizer_file':phase_apodizer_file, 'amplitude_apodizer_file':amplitude_apodizer_file, 'TILT':[0.,0.],'LS':LS,'RAVC':RAVC, 'LS_phase_apodizer_file':LS_phase_apodizer_file, 'LS_amplitude_apodizer_file':LS_amplitude_apodizer_file,'LS_parameters':LS_parameters, 'atm_screen':0, 'missing_segments_number':missing_segments_number, 'apodizer_misalignment':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'LS_misalignment':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'Island_Piston':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'NCPA':0, 'Debug_print':Debug_print,'Debug':Debug}, QUIET=True)
            Offaxis_psf = (abs(wfo_offaxis))**2
            fits.writeto(path+prefix+'_psf_offaxis_nonorm.fits', Offaxis_psf[int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio),int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio)], overwrite=True)
            if (Norm_max == True):
                #if ("Norm_max" in kwargs and kwargs["Norm_max"]):
                psf_noVortex_Mask_maxnorm = Offaxis_psf/np.max(NoCoro_psf)
                fits.writeto(path+prefix+'_psf_offaxis_maxnorm.fits', psf_noVortex_Mask_maxnorm[int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio),int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio)], overwrite=True)
            if (Norm_flux1 == True):
                #if ("Norm_flux1" in kwargs and kwargs["Norm_flux1"]):
                psf_noVortex_Mask_flux1norm = Offaxis_psf/sum(sum(NoCoro_psf))
                fits.writeto(path+prefix+'_psf_offaxis_flux1norm.fits', psf_noVortex_Mask_flux1norm[int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio),int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio)], overwrite=True)

    atm_screen=np.array(atm_screen)
    NCPA=np.array(NCPA)

    if (atm_screen.ndim == 3) or (TILT.ndim == 2) or (LS_misalignment.ndim == 2) or (apodizer_misalignment.ndim == 2) or (Island_Piston.ndim == 2) or (NCPA.ndim == 3):
        print('Cube')
        
        if (atm_screen.ndim == 3):
            length_cube = atm_screen.shape[0]
        if (TILT.ndim == 2):
            length_cube = TILT.shape[0]
        if (LS_misalignment.ndim == 2):
            length_cube = LS_misalignment.shape[0]
        if (apodizer_misalignment.ndim == 2):
            length_cube = apodizer_misalignment.shape[0]
        if (Island_Piston.ndim == 2):
            length_cube = Island_Piston.shape[0]
        if (NCPA.ndim == 3):
            length_cube = NCPA.shape[0]
        
        psf_Coro = np.zeros((length_cube,n,n))

        
        for iter in range(0, length_cube):
            print('iter: ', iter)

            if ((isinstance(atm_screen, (list, tuple, np.ndarray)) == True)):
                if (atm_screen.ndim == 3):
                    atm_screen_iter = atm_screen[iter,:,:]
                else:
                    atm_screen_iter = atm_screen
            if (TILT.ndim == 2):
                TILT_iter = TILT[iter,:]
                print('TILT: ', TILT_iter)
            else:
                TILT_iter = TILT
            if (LS_misalignment.ndim == 2):
                LS_misalignment_iter =  LS_misalignment[iter,:]
            else:
                LS_misalignment_iter =  LS_misalignment
            if (apodizer_misalignment.ndim == 2):
                apodizer_misalignment_iter = apodizer_misalignment[iter,:]
            else:
                apodizer_misalignment_iter = apodizer_misalignment
            if (Island_Piston.ndim == 2):
                Island_Piston_iter = Island_Piston[iter,:]
            else:
                Island_Piston_iter = Island_Piston
            if (isinstance(NCPA, (list, tuple, np.ndarray)) == True):
                if (NCPA.ndim == 3):
                    NCPA_iter = NCPA[iter,:,:]
                else:
                    NCPA_iter = NCPA

            if (Vortex == True):
                ## Si Vortex Si Mask --> simulate the coronagraphic-apodized psf
                (wfo_Coro, sampling) = proper.prop_run('telescope', lam, n, PASSVALUE={'prefix':prefix,'path':path,  'charge':charge, 'CAL':0, 'diam':diam, 'spiders_width':spiders_width, 'spiders_angle':spiders_angle, 'beam_ratio': beam_ratio, 'f_lens':f_lens, 'npupil':npupil, 'r_obstr':r_obstr, 'pupil_file':pupil_file, 'phase_apodizer_file':phase_apodizer_file, 'amplitude_apodizer_file':amplitude_apodizer_file, 'TILT':TILT_iter,'LS':LS,'RAVC':RAVC, 'LS_phase_apodizer_file':LS_phase_apodizer_file, 'LS_amplitude_apodizer_file':LS_amplitude_apodizer_file,'LS_parameters':LS_parameters,  'atm_screen':atm_screen_iter, 'missing_segments_number':missing_segments_number, 'apodizer_misalignment':apodizer_misalignment_iter, 'LS_misalignment':LS_misalignment_iter, 'Island_Piston':Island_Piston_iter, 'NCPA':NCPA_iter,'Debug_print':Debug_print,'Debug':Debug}, QUIET=True)
                psf_Coro[iter,:,:] = (abs(wfo_Coro))**2
            else:
        
            ## Apodizer --> simulate a coronagraphic psf with an apodizer (no Vortex)
                (wfo_apodizer, sampling) = proper.prop_run('telescope', lam, n, PASSVALUE={'prefix':prefix,'path':path,  'charge':0, 'CAL':0, 'diam':diam, 'spiders_width':spiders_width, 'spiders_angle':spiders_angle, 'beam_ratio': beam_ratio, 'f_lens':f_lens, 'npupil':npupil, 'r_obstr':r_obstr, 'pupil_file':pupil_file, 'phase_apodizer_file':phase_apodizer_file, 'amplitude_apodizer_file':amplitude_apodizer_file, 'TILT':TILT_iter,'LS':LS,'RAVC':False, 'LS_phase_apodizer_file':LS_phase_apodizer_file, 'LS_amplitude_apodizer_file':LS_amplitude_apodizer_file,'LS_parameters':LS_parameters, 'atm_screen':atm_screen_iter, 'missing_segments_number':missing_segments_number, 'apodizer_misalignment':apodizer_misalignment_iter, 'LS_misalignment':LS_misalignment_iter, 'Island_Piston':Island_Piston_iter,'NCPA':NCPA_iter, 'Debug_print':Debug_print,'Debug':Debug}, QUIET=True)
                psf_Coro[iter,:,:]  = (abs(wfo_apodizer))**2
        
        fits.writeto(path+prefix+'_psf_cube_Coro_nonorm.fits', psf_Coro[:,int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio),int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio)], overwrite=True)
        if (Norm_max == True):
            if (Vortex == True):
                psf_Coro_maxnorm = psf_Coro/np.max(Offaxis_psf)
            else:
                psf_Coro_maxnorm = psf_Coro/np.max(NoCoro_psf)
        fits.writeto(path+prefix+'_psf_cube_Coro_maxnorm.fits', psf_Coro_maxnorm[:,int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio),int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio)], overwrite=True)
        if (Norm_flux1 == True):
            psf_Coro_flux1norm = psf_Coro/sum(sum(NoCoro_psf))
            fits.writeto(path+prefix+'_psf_cube_Coro_flux1norm.fits', psf_Coro_flux1norm[:,int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio),int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio)], overwrite=True)

        
    else:
        print('No Cube')
        if (Vortex == True):
            print('VC') 
            print('ATM: ', int(isinstance(atm_screen, (list, tuple, np.ndarray)) == True))
            print('atm screen: ', atm_screen.shape)
        ## Si Vortex Si Mask --> simulate the coronagraphic-apodized psf
            (wfo_Coro, sampling) = proper.prop_run('telescope', lam, n, PASSVALUE={'prefix':prefix,'path':path,  'charge':charge, 'CAL':0, 'diam':diam, 'spiders_width':spiders_width, 'spiders_angle':spiders_angle, 'beam_ratio': beam_ratio, 'f_lens':f_lens, 'npupil':npupil, 'r_obstr':r_obstr, 'pupil_file':pupil_file, 'phase_apodizer_file':phase_apodizer_file, 'amplitude_apodizer_file':amplitude_apodizer_file, 'TILT':TILT,'LS':LS,'RAVC':RAVC, 'LS_phase_apodizer_file':LS_phase_apodizer_file, 'LS_amplitude_apodizer_file':LS_amplitude_apodizer_file,'LS_parameters':LS_parameters,  'atm_screen':atm_screen, 'missing_segments_number':missing_segments_number, 'apodizer_misalignment':apodizer_misalignment, 'LS_misalignment':LS_misalignment, 'Island_Piston':Island_Piston, 'NCPA':NCPA,'Debug_print':Debug_print,'Debug':Debug}, QUIET=True)
            psf_Coro = (abs(wfo_Coro))**2
        
        else:
            print('Apodizer')  
            print('ATM: ', int(isinstance(atm_screen, (list, tuple, np.ndarray)) == True))
            print('atm screen: ', atm_screen.shape)
            ## Apodizer --> simulate a coronagraphic psf with an apodizer (no Vortex)
            (wfo_apodizer, sampling) = proper.prop_run('telescope', lam, n, PASSVALUE={'prefix':prefix,'path':path,  'charge':0, 'CAL':0, 'diam':diam, 'spiders_width':spiders_width, 'spiders_angle':spiders_angle, 'beam_ratio': beam_ratio, 'f_lens':f_lens, 'npupil':npupil, 'r_obstr':r_obstr, 'pupil_file':pupil_file, 'phase_apodizer_file':phase_apodizer_file, 'amplitude_apodizer_file':amplitude_apodizer_file, 'TILT':TILT,'LS':LS,'RAVC':False, 'LS_phase_apodizer_file':LS_phase_apodizer_file, 'LS_amplitude_apodizer_file':LS_amplitude_apodizer_file,'LS_parameters':LS_parameters, 'atm_screen':atm_screen, 'missing_segments_number':missing_segments_number, 'apodizer_misalignment':apodizer_misalignment, 'LS_misalignment':LS_misalignment, 'Island_Piston':Island_Piston,'NCPA':NCPA, 'Debug_print':Debug_print,'Debug':Debug}, QUIET=True)
            psf_Coro = (abs(wfo_apodizer))**2

        fits.writeto(path+prefix+'_psf_Coro_nonorm.fits', psf_Coro[int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio),int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio)], overwrite=True)
        if (Norm_max == True):
            if (Vortex == True):
                psf_Coro_maxnorm = psf_Coro/np.max(Offaxis_psf)
            else:
                psf_Coro_maxnorm = psf_Coro/np.max(NoCoro_psf)
            fits.writeto(path+prefix+'_psf_Coro_maxnorm.fits', psf_Coro_maxnorm[int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio),int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio)], overwrite=True)
        if (Norm_flux1 == True):
            psf_Coro_flux1norm = psf_Coro/sum(sum(NoCoro_psf))
            fits.writeto(path+prefix+'_psf_Coro_flux1norm.fits', psf_Coro_flux1norm[int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio),int(n/2)-math.ceil(50./beam_ratio):int(n/2)+math.ceil(50./beam_ratio)], overwrite=True)



    return

if __name__ == '__main__':
    metiscoronagraphsimulator()

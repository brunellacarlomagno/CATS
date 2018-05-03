#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:21:04 2018

@author: brunella
"""

from __future__ import absolute_import
from astropy.io import fits
import cats

path = '/your_path/' 

n = 1024
lam = 3.8
pixelsize = 5.
charge=2
LS_parameters = [0.98, 0.03, 1.1]

cats.cats_simus.metiscoronagraphsimulator(n,lam, pixelsize, 'VC2_3.8microns', path, charge=charge, LS_parameters=LS_parameters, atm_screen=0,ELT_circ=True, Vortex = True, Back=True,LS=True, Debug=True, Debug_print=True, Norm_max=True)

NoCoro_psf = fits.getdata(path+'VC2_3.8microns_psf_noCoro_nonorm.fits')

cats.cats_simus.metiscoronagraphsimulator(n,lam, pixelsize, 'RAVC2_3.8microns', path, charge=charge, LS_parameters=LS_parameters, NoCoro_psf = NoCoro_psf,  ELT_circ=True, Vortex = True, RAVC=True, LS=True, Debug=True, Debug_print=False, Norm_max=True)

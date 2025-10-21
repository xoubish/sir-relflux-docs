#
# Copyright (C) 2015-2020 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General
# Public License as published by the Free Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#

#
# File RelativeFluxCorrectionCore.py
#
# Created on: July 14, 2020
#
# Author: Marco Fumana, Shooby
#


import ElementsKernel.Logging as log


import os
from typing import Tuple

import numpy as np
import astropy.io.fits as fits
from scipy.interpolate import RegularGridInterpolator as RGI

from SIR_InstrumentModels.DetectorModel import DetectorModel
from SIR_SpectraExtractionBinding import SpectrumDataSet
from SIR_H5SpectraExtractionBinding import H5ExtractedSpectrum
from SIR_SpectraExtractionBinding import SPECTRUM_MASK


logger = log.getLogger("LargeScaleFlatCore")

def normalfov(box, fov):
    """gets 
    - the detector model which has the corners of the FOV in mm 
    - the fov in mm
    returns:
    - the focal plane coordinates in normalized [-1,1] range """
    xslope = 2.0 / (box[1][0] - box[0][0])
    xint = (1 + (box[1][0] / box[0][0])) / (1.0 - (box[1][0] / box[0][0]))
    x = xslope * fov[0] + xint
        
    yslope = 2.0 / (box[1][1] - box[0][1])
    yint = (1 + (box[1][1] / box[0][1])) / (1.0 - (box[1][1] / box[0][1]))
    y = yslope * fov[1] + yint
    return x, y

class RelativeFluxCorrectionCore():
    SPEC1D_RELFLUX_LABEL = '1D REL'

    """Core of the Large Scale Flat correction"""
    def __init__(self, flatcube, flatquality):
        self.flatcube = flatcube
        self.flatquality = flatquality

        # Initialize interpolators as None; they will be set in setup_interpolators
        self._fn = None
        self._fnq = None

    def setup_interpolators(self, wstart, wend):

        XX = np.linspace(-1., 1., np.shape(self.flatcube)[2])
        YY = np.linspace(-1., 1., np.shape(self.flatcube)[1])
        ZZ = np.linspace(wstart, wend, np.shape(self.flatcube)[0])
        self.flatcube[np.isnan(self.flatcube)] = 0.0
        self.flatquality[np.isnan(self.flatquality)] = 1.0
        self._fn = RGI((ZZ, YY, XX), self.flatcube)
        self._fnq = RGI((ZZ, YY, XX), self.flatquality)

    def compute_correction(self, wavelengths: np.ndarray, location: 'LocationSpectrum'):
        if self._fn is None:
            self.setup_interpolators(wavelengths.min(), wavelengths.max())

        middle_row = location.getApertureSize() // 2
        trace = np.array([location.computePosition(float(l), row=middle_row).getPosition() for l in wavelengths])
        detector_model = location.getDetectorModel()
        detector_box = detector_model.getEnvelopeBox()
        det_id = location.getDetectorID()

        x_fov, y_fov = normalfov(detector_box, detector_model.getFOVPositions(trace, det_id).T)
        in_fov = (x_fov > -1) & (x_fov < 1) & (y_fov > -1) & (y_fov < 1)
        points = np.array([wavelengths, y_fov, x_fov])

        correction_mag = np.ones_like(wavelengths) * 10.0
        correction_mag[in_fov] = self._fn(points.T[in_fov])
        correction_factor = np.ones_like(correction_mag)
        correction_factor[in_fov] = 10.0 ** -(correction_mag[in_fov] / -2.5)

        return correction_factor, correction_mag, in_fov

    def get_correction(self, spectrum: H5ExtractedSpectrum) -> Tuple[np.ndarray]:
        """
        Gets a single corrected spectrum.

        Parameters
        ----------

        spectrum: H5ExtractedSpectrum
            The uncorrected spectrum
        
        Returns
        -------

        corrected_flux, corrected_variance, magnitude_correction

        The corrected flux, variance, and a magnitude correction as a tuple of 1D NumPy arrays.
        """
        dataset = spectrum.getSpectrumDataSet(H5ExtractedSpectrum.SPEC1D_LABEL)
        corrected_flux = dataset.getActualData()
        corrected_var = dataset.getActualVariance()

        correction, correction_mag, in_fov = self.compute_correction(spectrum.getWavelength(),
                                                                     spectrum.getLocationSpectrum())

        corrected_flux[in_fov] *= correction[in_fov]
        corrected_var[in_fov] *= correction[in_fov] ** 2

        return corrected_flux, corrected_var, correction_mag

    def invoke(self, exspectra, overwrite):
        """
        Applying the correction to all spectra in exspectra collection,
        looping over each spectrum and over wavelengths in each
        """
        corrections = []

        for spectrum in exspectra:

            corrected_flux, corrected_var, correction_mag = self.get_correction(spectrum)

            dataset = spectrum.getSpectrumDataSet(H5ExtractedSpectrum.SPEC1D_LABEL)
            preflux = dataset.getActualData()
            corrected_QF = dataset.getQuality() # we don't change quality
            cov_tolerance = dataset.getCovarianceContainer().getTolerance()

            maskk = dataset.getMask()
            sel = (correction_mag == 10)
            mask_arr = np.zeros_like(corrected_flux[sel])
            mask_arr[(correction_mag[sel]) > 0.5] = SPECTRUM_MASK["SUSP_RFX"]
            mask_arr[(correction_mag[sel]) < -0.5] = SPECTRUM_MASK["SUSP_RFX"]
            maskk[sel] = maskk[sel] + mask_arr

            dataset_1d = SpectrumDataSet(cov_tolerance)
            timestamp = spectrum.getSpectrumDataSet(H5ExtractedSpectrum.SPEC1D_LABEL).getExposureTime()
            dataset_1d.setExposureTime(timestamp)
            
            lsf_sigma = dataset.getLSFSigma()
            dataset_1d.setLSFSigma(lsf_sigma)
            dataset_1d.setRawData((corrected_flux).astype(np.float32), (corrected_QF).astype(np.float32), True)
            dataset_1d.setRawVariance(corrected_var.astype(np.float32))
            dataset_1d.setMask(maskk)
            spectrum.addSpectrumDataSet(RelativeFluxCorrectionCore.SPEC1D_RELFLUX_LABEL, dataset_1d, replace=overwrite)

            ### For the red/green flag: collecting the correction factors of 
            ### non-masked and ok quality pixels into one array, red flag in all 
            ### spectra if the mean and std corrections are too large
    
            u = (maskk == 0) & (corrected_QF > 0.5)
            corrections.append(np.array(correction_mag[u]))

        if (len(corrections) > 0): #JX There are some data in the corrections array
            correction_array = np.concatenate(corrections, axis=0)
            if (np.abs(np.mean(correction_array)) < 0.3) & (np.std(correction_array) < 0.3):
                flagg = 0 ## green
            else:
                flagg = 1 ## red
        else:
           flagg = 0 ## green
            
        #JX
        return flagg

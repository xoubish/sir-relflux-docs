
# Overview

SIR RelativeFlux handles **relative flux scaling** in the SIR pipeline.

- **Calibration (CAL-RFX):** Builds the wavelength- and field-dependent relative response model from self-calibration observations.
- **Correction (SCI-RFX):** Applies that model to science spectra for consistent flux scales across detectors and epochs.

This documentation covers:
- The Calibration and Correction algorithms
- Their data products and validation steps
- The software API

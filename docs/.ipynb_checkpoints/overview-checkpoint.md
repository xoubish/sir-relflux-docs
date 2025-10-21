# Overview

## Purpose
The **SIR RelativeFlux** module provides the *relative spectrophotometric calibration* for Euclid slitless spectroscopy.  
Its goal is to ensure that all extracted spectra from the NISP instrument share a **consistent relative flux scale** across
time, detector, and wavelength — before absolute flux calibration is applied.

RelativeFlux is divided into two processing components:

- **Calibration (CAL-RFX):** Derives the instrument's wavelength- and position-dependent *relative response model*  
  using repeated “self-calibration” observations and stable reference sources.
- **Correction (SCI-RFX):** Applies the derived model to all science spectra, bringing them to a common
  instrumental scale for later absolute calibration and combination.

These two steps correspond to **SCI-RFX** and **CAL-RFX** processing stages in the SIR pipeline, as described in the
_SIR Software Design Document (SDD)_.

---

## Context in the SIR Pipeline
Relative flux scaling sits between **flat-field/extraction** and **absolute flux calibration (AFX)**:

- **Inputs:** extracted 1D spectra, source metadata, and calibration field observations.  
- **Outputs:** corrected spectra and the relative response model product.  

Raw Frames
    ↓
Detector-Level Calibrations (DARK, FLAT)
    ↓
Extraction (SCI-EXT)
    ↓
Relative Flux Calibration (CAL-RFX → SCI-RFX)
    ↓
Absolute Flux Calibration (AFX)
    ↓
Combination / Merging (COM)
---

## Motivation
Despite careful flat-fielding, residual **wavelength- and detector-dependent sensitivity variations** remain.
These arise from:
- differences in optical throughput between detectors,
- time-variable response of the NISP grisms, and
- imperfect large-scale flat correction.

By leveraging repeat observations of bright, stable sources, SIR RelativeFlux determines a smooth
two-dimensional model of these effects as a function of wavelength and detector position.  
This model brings all spectra to a uniform internal scale.

---

## Processing Summary

| Component | Task | Product | Description |
|------------|------|----------|-------------|
| **CAL-RFX** | Build relative response model | `RELFLUX_MODEL` | Fit smooth surface over λ × detector using self-cal data |
| **SCI-RFX** | Apply model to science spectra | `RELFLUX_CORRECTED_SPECTRA` | Scale spectra by model response and propagate uncertainties |

Both steps generate quality-control (QC) metrics stored alongside the data products.

---

## Data Products
The calibration produces an intermediate *Relative Response Model* containing:
- Response factors vs. wavelength and detector ID
- Validity time range
- Provenance of input exposures
- Versioning information

The correction step produces:
- Corrected 1D spectra
- Flags for missing or extrapolated regions
- QC statistics such as detector-to-detector repeatability

See the detailed [Calibration Data Product](calibration/product.md) section for format and metadata definitions.

---

## Validation and Performance
RelativeFlux calibration quality is monitored through:
- Repeatability scatter across calibration sources
- Consistency across detectors and epochs
- Temporal stability of response
- Comparison with external photometric standards (for trend checks)

Performance metrics and plots are presented in the [Validation](calibration/validation.md) sections.

---

## Summary
The **SIR RelativeFlux** stage is critical to ensure that spectral features and continuum levels
are comparable across the entire NISP field and over time.
It provides the bridge between instrumental calibration and scientific flux measurements,
delivering both the calibration product and the corrected spectra required for the Euclid
Spectroscopic Pipeline.
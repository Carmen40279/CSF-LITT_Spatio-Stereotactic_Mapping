# ThermoSEEG RC — research prototype

ThermoSEEG is a **2D research and educational prototype** for image-mask geometry, threshold sensitivity, finite candidate search, geometric SEEG contact placement, and a simplified synthetic thermal demonstration. It is not clinically validated and must not be used for clinical decisions or surgical guidance.

## Runtime and installation

Use 64-bit Python 3.11–3.13. The pinned environment was installed and tested with Python 3.13.17. On Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --require-hashes -r requirements-lock.txt
streamlit run ThermoSEEG_RC_UI.py
```

Direct dependencies are pinned in `requirements.txt`; `requirements-lock.txt` pins the resolved dependency set with hashes for reproducible installs. Recompile and review the lock, then rerun the suite after changing direct pins.

## Supported inputs

- JPG, JPEG, PNG raster slices.
- Single-frame, single-channel, monochrome DICOM MR or CT slices. MR is the intended input; CT is decoded only, and requires explicit acknowledgement because the ROI heuristic assumes MRI-like contrast and is not validated for CT.
- DICOM Modality LUT or Rescale Slope/Intercept is applied before VOI LUT/window when present. The resulting display values are then scaled per slice to 8-bit for the existing segmentation pipeline; they are not quantitative tissue intensities; original HU or MR signal values are not preserved.
- Uploads are limited to 50 MB, 4096 pixels per image side, and 16 million pixels before full pixel decoding. Color, multi-frame, non-MR/CT, empty, constant-intensity, malformed, or unsupported DICOM data is rejected with an error message. CT analysis requires acknowledgement that MRI-oriented ROI heuristics are unvalidated for CT.
- Pixel Spacing is read as `[row, column]` and reported as horizontal/vertical spacing. Orientation and position are validated and exported as metadata but are not used to construct 3D coordinates.

## Analysis outline

1. Decode image data and create an 8-bit display representation.
2. Build a heuristic foreground mask using Otsu thresholding, morphology, and largest connected component. This is not a brain segmentation; morphological closing can bridge small gaps. No CSF or tissue classification is performed.
3. Extract high-signal regions by the configured intensity threshold. This is not vessel or functional-anatomy segmentation.
4. Evaluate a 2D raster trajectory, pixel-based risk clearance, original ROI containment, and a separately reported tolerated ROI.
5. Vary only the risk threshold in three sensitivity cases while holding the ROI fixed. This stability measure does not measure accuracy, mask quality, or data reliability.
6. In LITT mode, render an uncalibrated static Gaussian demonstration. Duration scales synthetic CEM43 only. If the numerical guard exceeds 100 °C, CEM43 and dose area are suppressed; the displayed temperature field is clipped at the documented display ceiling.

## Units and interpretation

- Path length and minimum Euclidean clearance in millimeters use horizontal and vertical Pixel Spacing when available or an explicitly entered isotropic manual spacing.
- The operative clearance thresholds, risk dilation, candidate score, contact placement and thermal kernels remain pixel-based experimental settings. Therefore metric distances do not imply calibrated physical safety thresholds.
- SEEG contact fit is geometric only. The displayed contact span in millimeters is measured between rounded contact coordinates.
- CEM43 inputs derive from an unvalidated synthetic temperature field. The 240-minute reference threshold is not universal and has no patient-specific interpretation here.
- “Heuristic checks passed” is a software status only. It does not mean safe, valid anatomy, or clinical suitability.

## Exports and privacy

The app offers JSON, annotated PNG, and PDF research reports. JSON records the image SHA-256, image dimensions, processing settings, software versions, timestamps, spacing source, sensitivity scenarios, candidate-search audit, and known scope limitations. Patient identifiers are not added to exports. Uploaded bytes are not intentionally saved to disk, but image arrays can remain in process memory and the Streamlit data cache for up to five minutes (and two processed images or four distance maps) until eviction or process restart; operate only under the team's data-governance rules.

The JSON is a run record and its `reproducibility.configuration` section can be loaded from the sidebar. Import restores supported slider/checkbox values and trajectory points only when the uploaded image SHA-256 matches; mismatched image points are not restored. Reproduction still requires the same image bytes, compatible pinned library environment, and software version.

## Automated tests

```powershell
python -m unittest discover -s tests -v
```

The suite covers core geometry, mask extraction, target validation, contact placement, thermal dose masking and numerical-guard suppression, exported-config edge cases, plus generated synthetic DICOM ingestion. It is not a clinical validation study. Full Streamlit interaction, PDF visual rendering, large-image performance, and broad real-world DICOM vendor compatibility still require environment-backed testing.

## Scientific and engineering limitations

- One 2D slice only; no 3D spatial reconstruction or inter-slice anatomy.
- ROI and risk masks are heuristic intensity masks, not anatomy.
- No vessel, CSF, eloquent cortex, tissue property, perfusion, or patient-specific thermal model.
- No transient bioheat simulation, empirical calibration, external reference comparison, or clinical validation.
- Sensitivity has only three threshold cases and a fixed ROI.
- Auto-search checks a finite sampled set with local pixel refinement. Its score is an experimental ranking rule, not a global optimum. Exported counters distinguish proposed candidates from full path evaluations and unique evaluated coordinates.
- DICOM spatial tags are recorded but only Pixel Spacing contributes to the 2D metric calculations. Presentation LUT and specialized multi-frame/colored objects are not supported.
- Python/package versions are captured per export. Determinism is not claimed: JSON records `determinism.claim = not_established`; neither repeat-run nor cross-environment reproducibility has been established. SHA-256 is used for full image identities, session keys, and signatures; output filenames show its first 16 hex characters. Local and UTC timestamps derive from one instant. Clearance thresholds are experimental pixel values, not clinical safety limits.

Experimental calibration, anatomical ground truth, independent validation data, and expert review are responsibilities of the research team and are not claimed by this software.


# OZITelemetry

## Overview

`OZITele` is a Python class dedicated to the analysis of OZIRIIS telemetry data.

It allows you to:
- load a saved telemetry `.npy` file,
- extract the two Zernike Wavefront Sensor (ZWFS) image streams,
- reconstruct the wavefront phase from the two ZWFS channels,
- convert reconstructed phase maps into OPD maps,
- project OPDs onto actuator influence functions or modal bases,
- compute PSDs from reconstructed OPDs and command vectors.

The main class is implemented in `OZITelemetry.py` and is designed for offline analysis of telemetry cubes.

## Main features

The `OZITele` class provides the following public methods:
- `compute_projectors()`
- `extract_Zimages()`
- `reconstruct_phase(...)`
- `reconstruct_all_phase(...)`
- `project_OPDs()`
- `PSD_IFs(...)`
- `PSD_modal(...)`
- `compute_all_PSD(...)`
- `PSD_cmd_IFs(...)`
- `PSD_cmd_modal(...)`

Example files are provided in:
- `OZITele_spyder_tutorial.py`
- `OZITele_jupyter_tutorial.ipynb`

## Requirements

This project relies on:
- `numpy`
- `matplotlib`
- `scipy`
- `scikit-image`
- `astropy`
- `tqdm`
- `tkinter`
- `OOPAO`

## Important OOPAO modifications

This project depends on custom ZWFS classes that must be available inside your local `OOPAO` package.

### 1. Add `ZWFS.py` and `ZWFS2.py` to the OOPAO package

You must copy:
- `ZWFS.py`
- `ZWFS2.py`

into your local `OOPAO` package directory.

These files must then be importable as:
- `OOPAO.ZWFS`
- `OOPAO.ZWFS2`

Without this step, `OZITelemetry.py` will not import correctly because `OZITele` explicitly imports these classes.

### 2. Modify `OOPAO.Telescope.__mul__`

You must modify the `__mul__` method in `OOPAO.Telescope` by adding at the corresponding line:

```python
obj.tag == 'ZWFS'
```

In practice, you need to add a condition for `obj.tag == 'ZWFS'` in the relevant line of `OOPAO.Telescope.__mul__`.

Without this modification, propagation from a telescope object to the ZWFS objects will fail.

## Typical usage

```python
from OZITelemetry import OZITele

tele = OZITele(tele_path="my_telemetry_file.npy", is_onsky=True)

tele.reconstruct_all_phase(method="atan", iteration=10, damping=0.5)
tele.project_OPDs()
tele.compute_all_PSD(npsg=1024)
```

## Notes

- If `tele_path=None`, the class opens a file selection dialog using `tkinter`.
- This is generally more reliable in Spyder or a standard Python session than in JupyterLab.
- In JupyterLab, it is usually better to pass the telemetry path explicitly.

## Repository content

- `OZITelemetry.py`: main telemetry analysis class
- `Pupil_selection.py`: pupil detection and mask extraction utilities
- `plot_functions.py`: plotting utilities
- `ZWFS.py`: custom ZWFS class for OOPAO
- `ZWFS2.py`: dual-ZWFS class for OOPAO
- `OZITele_spyder_tutorial.py`: Spyder tutorial script
- `OZITele_jupyter_tutorial.ipynb`: Jupyter tutorial notebook

## Status

This repository is intended for local scientific use and depends on a modified OOPAO installation.

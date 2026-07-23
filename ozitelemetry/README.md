<<<<<<< HEAD
# OOPAO

Object-Oriented Python Adaptive Optics (OOPAO) is a project under development to propose a python-based tool to perform end-to-end AO simulations.
This code is inspired from the OOMAO architecture: https://github.com/cmcorreia/LAM-Public developped by C. Correia and R. Conan (https://doi.org/10.1117/12.2054470). 
The project was initially intended for personal use in the frame of an ESO project. It is now open to any interested user. 

## DOCUMENTATION 

The OOPAO documentation is here:  https://cheritier.github.io/OOPAO/index.html

## FUNCTIONALITIES

	_ Atmosphere: 		Multi-layers with infinitely and non-stationary phase screens, conditions can be updated on the fly if required, scintillation can be simulated
	_ Telescope: 		Default circular pupil or user defined, with/without spiders
	_ Deformable Mirror:	Gaussian Influence Functions (default) or user defined, cartesian coordinates (default) or user defined
	_ WFS: 			Pyramid, SH-WFS (diffractive and geometric), Bi-O Edge
	_ Source: 		NGS or LGS
	_ Control Basis: 	KL modal basis, Zernike Polynomials

## MODULES REQUIRED
The code is written for Python 3 (version 3.8.8) and requires the following modules:
```
joblib          => paralleling computing
scikit-image    => 2D interpolations
astropy         => handling of fits files
pyFFTW          => optimization of the FFT  
mpmath          => arithmetic with arbitrary precision
jsonpickle      => json files encoding
aotools         => zernike modes and functionalities for atmosphere computation
numba           => required in aotools
numexpr 	=> optimized maths operations
psutil		=> access system info
mpmath		=> real and complex floating-point arithmetic with arbitrary precision
tqdm 		=> loading bar
```
If GPU computation is available:
```
cupy => GPU computation of the PWFS code (Not required)
```

## INSTALLATION 

### (Recommended) Creating a virtual environment

It is always recommended that you use a virtual environment. First create it:

```
python -m venv venv

# or

python3 -m venv venv

```

And finally activate it:

```
# Unix
source ./venv/bin/activate

# or

# Windows PowerShell
.\venv\Scripts\activate
```

After the environment is set up and activated, this package can then be easily installed. Anytime you wish to use this
package, you should activate the respective environment.

### Using `pip`

First clone the repository:

```
https://github.com/cheritier/OOPAO.git
```

And then install the package using `pip`:

```
python -m pip install -e OOPAO

# or 

python3 -m pip install -e OOPAO
```

To include CUDA acceleration, just specify the `[CUDA]` branch:

```
python -m pip install -e OOPAO[CUDA]

# or 

python3 -m pip install -e OOPAO[CUDA]
```

If you experience errors during the installation of one of the required scientific packages (e.g. `numpy`, `cupy`), 
please consider using a `conda` environment and installing these using `conda install`.


## CONTRIBUTORS
Main developer and maintainer: Cédric Taïssir Héritier

Main contributors: 
 - João Aveiro
 - Byron Engler
 - Arseniy Kuznetsov
 - Rafael Machado Salgueiro
 - Arnaud Striffling
 - Christophe Vérinaud
 - Jonathan Dray
 - João Monteiro
 - Matteo Pasinetti
 - Francisco Oyarzùn

## CITING OOPAO
If you use OOPAO for your own research, we kindly ask you to cite the OOPAO AO4ELT7 proceeding (Heritier et al. 2023).
See https://hal.science/AO4ELT7/hal-04402878v1.

## ACKNOWLEDGEMENTS
This tool has been developped during the Engineering & Research Technology Fellowship of C. Héritier funded by ESO. 
Some functionalities of the code make use of the aotools package developped by M. J. Townson et al (2019). See https://doi.org/10.1364/OE.27.031316.


## LICENSE
This project is licensed under the terms of the GPL license.
=======
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
>>>>>>> ozi/main

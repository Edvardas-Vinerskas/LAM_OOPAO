# -*- coding: utf-8 -*-
"""
Spyder tutorial for the OZITele class.

This script is meant to be opened in Spyder and executed section by section
with the ``#%%`` cells.

It demonstrates the use of all public methods of OZITele:
    - __init__
    - compute_projectors
    - extract_Zimages
    - reconstruct_phase
    - reconstruct_all_phase
    - project_OPDs
    - PSD_IFs
    - PSD_modal
    - compute_all_PSD
    - PSD_cmd_IFs
    - PSD_cmd_modal

Notes
-----
1. In Spyder, using ``tele_path=None`` should allow the Tk file dialog to open.
2. In notebooks, passing an explicit path is usually more reliable.
3. Some PSD methods store the tuple returned by ``_psd`` directly as
   ``(freq, psd)``. This tutorial handles that explicitly.
"""

#%% Imports
from OZITelemetry import OZITele
import numpy as np
import matplotlib.pyplot as plt


#%% User settings
# Choose one of the two modes below.

USE_FILE_DIALOG = True

# If USE_FILE_DIALOG is False, set your file path here.
TELE_PATH = r"C:\path\to\your\telemetry_file.npy"

# Typical class options
IS_ONSKY = True
USE_CNN_FALLBACK = False

# Plot settings
FRAME_TO_DISPLAY = 0
MODE_TO_DISPLAY = 0
NPERSEG = 1024


#%% Create the OZITele object
if USE_FILE_DIALOG:
    tele = OZITele(tele_path=None, is_onsky=IS_ONSKY, CNN=USE_CNN_FALLBACK)
else:
    tele = OZITele(tele_path=TELE_PATH, is_onsky=IS_ONSKY, CNN=USE_CNN_FALLBACK)

print("tele.tele_path:", tele.tele_path)
print("tele.img shape:", tele.img.shape)
print("tele.rec_cmd shape:", tele.rec_cmd.shape)
print("tele.time shape:", tele.time.shape)
print("tele.is_onsky:", tele.is_onsky)


#%% Inspect the main attributes created at initialization
print("off_mask shape         :", tele.off_mask.shape)
print("img_raw shape          :", tele.img_raw.shape)
print("dark shape             :", tele.dark.shape)
print("img_ZWFS1 shape        :", tele.img_ZWFS1.shape)
print("img_ZWFS2 shape        :", tele.img_ZWFS2.shape)
print("proj_M2C shape         :", tele.proj_M2C.shape)
print("proj_IF shape          :", tele.proj_IF.shape)
print("has_recontructed_phase :", tele.has_recontructed_phase)
print("has_projected_phase    :", tele.has_projected_phase)


#%% Public method: compute_projectors
# This is already called in __init__, but we call it again here to show usage.
tele.compute_projectors()

print("After compute_projectors():")
print("proj_M2C shape:", tele.proj_M2C.shape)
print("proj_IF shape :", tele.proj_IF.shape)


#%% Public method: extract_Zimages
# This is already called in __init__, but we call it again here to show usage.
tele.extract_Zimages()

print("After extract_Zimages():")
print("img_ZWFS1 shape:", tele.img_ZWFS1.shape)
print("img_ZWFS2 shape:", tele.img_ZWFS2.shape)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(tele.img_ZWFS1[FRAME_TO_DISPLAY])
plt.title(f"ZWFS1 frame {FRAME_TO_DISPLAY}")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(tele.img_ZWFS2[FRAME_TO_DISPLAY])
plt.title(f"ZWFS2 frame {FRAME_TO_DISPLAY}")
plt.colorbar()
plt.tight_layout()
plt.show()


#%% Public method: reconstruct_phase
# Reconstruct a single phase map from one pair of ZWFS images.
phase_one = tele.reconstruct_phase(
    tele.img_ZWFS1[FRAME_TO_DISPLAY],
    tele.img_ZWFS2[FRAME_TO_DISPLAY],
    method='atan',
    damping=0.5,
    iteration=10,
)

print("phase_one shape:", phase_one.shape)
print("phase_one min/max:", np.nanmin(phase_one), np.nanmax(phase_one))

plt.figure(figsize=(5, 5))
plt.imshow(phase_one)
plt.title(f"Single reconstructed phase, frame {FRAME_TO_DISPLAY}")
plt.colorbar()
plt.tight_layout()
plt.show()


#%% Public method: reconstruct_all_phase
# Reconstruct the full sequence and create tele.phase and tele.OPDs.
tele.reconstruct_all_phase(method='atan', iteration=10, damping=0.5)

print("After reconstruct_all_phase():")
print("phase shape:", tele.phase.shape)
print("OPDs shape :", tele.OPDs.shape)
print("has_recontructed_phase:", tele.has_recontructed_phase)

plt.figure(figsize=(5, 5))
plt.imshow(tele.OPDs[FRAME_TO_DISPLAY])
plt.title(f"OPD map [nm], frame {FRAME_TO_DISPLAY}")
plt.colorbar()
plt.tight_layout()
plt.show()


#%% Public method: project_OPDs
# Project reconstructed OPDs onto IFs and modal coefficients.
tele.project_OPDs()

print("After project_OPDs():")
print("OPDs_on_IFs shape  :", tele.OPDs_on_IFs.shape)
print("OPDs_on_modes shape:", tele.OPDs_on_modes.shape)
print("has_projected_phase:", tele.has_projected_phase)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(tele.OPDs_on_IFs[:, MODE_TO_DISPLAY])
plt.title(f"Projected OPD on IF #{MODE_TO_DISPLAY}")
plt.xlabel("Frame")
plt.ylabel("Amplitude")

plt.subplot(1, 2, 2)
plt.plot(tele.OPDs_on_modes[:, MODE_TO_DISPLAY])
plt.title(f"Projected OPD on mode #{MODE_TO_DISPLAY}")
plt.xlabel("Frame")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()


#%% Public method: PSD_IFs
tele.PSD_IFs(npsg=NPERSEG)

# In the current class version, PSD_IFs stores the tuple (freq, psd) in psd_IFs.
# So we unpack it explicitly.
freq_IFs, psd_IFs = tele.psd_IFs

print("After PSD_IFs():")
print("freq_IFs shape:", freq_IFs.shape)
print("psd_IFs shape :", psd_IFs.shape)

plt.figure(figsize=(6, 4))
plt.loglog(freq_IFs[1:], psd_IFs[1:, MODE_TO_DISPLAY])
plt.title(f"PSD of projected IF #{MODE_TO_DISPLAY}")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD")
plt.tight_layout()
plt.show()


#%% Public method: PSD_modal
tele.PSD_modal(npsg=NPERSEG)

freq_modal, psd_modal = tele.psd_modal

print("After PSD_modal():")
print("freq_modal shape:", freq_modal.shape)
print("psd_modal shape :", psd_modal.shape)

plt.figure(figsize=(6, 4))
plt.loglog(freq_modal[1:], psd_modal[1:, MODE_TO_DISPLAY])
plt.title(f"PSD of projected mode #{MODE_TO_DISPLAY}")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD")
plt.tight_layout()
plt.show()


#%% Public method: PSD_cmd_IFs
tele.PSD_cmd_IFs(npsg=NPERSEG)

freq_cmd_IFs, psd_cmd_IFs = tele.psd_cmd_IFs

print("After PSD_cmd_IFs():")
print("freq_cmd_IFs shape:", freq_cmd_IFs.shape)
print("psd_cmd_IFs shape :", psd_cmd_IFs.shape)

plt.figure(figsize=(6, 4))
plt.loglog(freq_cmd_IFs[1:], psd_cmd_IFs[1:, MODE_TO_DISPLAY])
plt.title(f"PSD of command IF #{MODE_TO_DISPLAY}")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD")
plt.tight_layout()
plt.show()


#%% Public method: PSD_cmd_modal
tele.PSD_cmd_modal(npsg=NPERSEG)

freq_cmd_modal, psd_cmd_modal = tele.psd_cmd_modal

print("After PSD_cmd_modal():")
print("freq_cmd_modal shape:", freq_cmd_modal.shape)
print("psd_cmd_modal shape :", psd_cmd_modal.shape)

plt.figure(figsize=(6, 4))
plt.loglog(freq_cmd_modal[1:], psd_cmd_modal[1:, MODE_TO_DISPLAY])
plt.title(f"PSD of command mode #{MODE_TO_DISPLAY}")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD")
plt.tight_layout()
plt.show()


#%% Public method: compute_all_PSD
# This recomputes every PSD product in one call.
tele.compute_all_PSD(npsg=NPERSEG)

print("After compute_all_PSD():")
print("tele.psd_IFs type      :", type(tele.psd_IFs))
print("tele.psd_modal type    :", type(tele.psd_modal))
print("tele.psd_cmd_IFs type  :", type(tele.psd_cmd_IFs))
print("tele.psd_cmd_modal type:", type(tele.psd_cmd_modal))

print("All PSD products were recomputed.")


#%% Final quick summary
print("\nSummary of useful outputs:")
print("tele.phase            -> reconstructed phase cube")
print("tele.OPDs             -> OPD cube in nm")
print("tele.OPDs_on_IFs      -> OPD projections on IFs")
print("tele.OPDs_on_modes    -> OPD projections on modes")
print("tele.psd_IFs          -> (freq, psd) for projected IFs")
print("tele.psd_modal        -> (freq, psd) for projected modes")
print("tele.psd_cmd_IFs      -> (freq, psd) for actuator commands")
print("tele.psd_cmd_modal    -> (freq, psd) for modal commands")

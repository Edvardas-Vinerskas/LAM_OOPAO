# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 16:28:46 2026

@author: mmotte
"""

import numpy as np  # Import NumPy for numerical operations

from scipy.ndimage import binary_fill_holes  # For image post-processing 
from skimage import measure  # For image measurements 
from plot_functions import *
from scipy.signal import welch
from scipy.interpolate import interp1d
from scipy.sparse import coo_matrix
from skimage.transform import resize
from matplotlib.widgets import Slider, TextBox, Button
import matplotlib.pyplot as plt
import logging  # For logging messages
import tqdm  # Progress bar
import sys
from astropy.io import fits
from pathlib import Path


def reference_intensities(off_mask_image, crop=0.25, use_tuner=True):
    if use_tuner:
        plt.close('all')
        tuner = CropMaskTuner(off_mask_image, initial_crop=crop)
        result = tuner.show()

        if result is None:
            raise RuntimeError("CropMaskTuner fermé sans confirmation.")

        crop, submasks, global_masks, positions, pupilles = result
        return positions, pupilles, submasks, global_masks

    submasks, pupilles, global_masks, positions, global_sum, error = compute_pupil_masks(
        off_mask_image, crop
    )

    if error is not None:
        raise ValueError(f"Impossible de calculer les masques de pupille : {error}")

    return positions, pupilles, submasks, global_masks
def pad_to_square(arr: np.ndarray):
    """
    Pad un array 2D ou 3D en le centrant pour le rendre carré sur ses 2 dernières dimensions.
    - Pour (M, N): pad sur lignes/colonnes pour obtenir (S, S).
    - Pour (L, M, N): pad sur axes M et N (L inchangé).

    Retour:
        padded : array paddé
        padded_cr : [rows_bottom, cols_left, rows_top, cols_right]
                     (avec la même convention que la version d'origine)
    """
    if arr.ndim == 2:
        M, N = arr.shape
    elif arr.ndim == 3:
        L, M, N = arr.shape
    else:
        raise ValueError("Seuls les arrays 2D ou 3D sont gérés.")

    size = max(M, N)
    pad_top = (size - M) // 2
    pad_bottom = size - M - pad_top
    pad_left = (size - N) // 2
    pad_right = size - N - pad_left

    # Padding selon dimension
    if arr.ndim == 2:
        pad_width = [(pad_top, pad_bottom), (pad_left, pad_right)]
    else:  # 3D
        pad_width = [(0, 0), (pad_top, pad_bottom), (pad_left, pad_right)]

    # Détermination du remplissage constant selon le type
    if np.issubdtype(arr.dtype, np.bool_):
        const_value = False
    elif np.issubdtype(arr.dtype, np.integer):
        const_value = 0
    elif np.issubdtype(arr.dtype, np.floating):
        const_value = 0.0
    else:
        raise TypeError(f"Type {arr.dtype} non supporté pour le padding constant.")

    padded = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=const_value)

    # Conserve la même convention que ton code précédent
    padded_cr = [-pad_bottom, -pad_left, pad_top, pad_right]

    return padded, padded_cr

def compute_pupil_masks(off_mask_image, crop):
    if off_mask_image.ndim != 2:
        return None, None, None, None, None, "off_mask_image must be 2D"

    img = off_mask_image
    vmax = np.nanmax(img)
    if not np.isfinite(vmax) or vmax <= 0:
        return None, None, None, None, None, "Invalid image max (<=0 or NaN)"

    crop = float(np.clip(crop, 0.0, 1.0))
    thr = crop * vmax

    mask = img >= thr
    labels = measure.label(mask, connectivity=2)
    regions = measure.regionprops(labels)

    if len(regions) < 2:
        return None, None, None, None, None, f"Only {len(regions)} region(s) detected"

    candidates = sorted(regions, key=lambda r: r.area, reverse=True)[:2]

    # 2) ordre FIXE : row minimal (pupille la plus haute) -> index 0
    regions_sorted = sorted(
        candidates,
        key=lambda r: r.centroid[0]   # tri croissant automatique
    )
    submasks = []
    pupilles_masked = []
    global_masks = []
    positions = []

    for r in regions_sorted:
        minr, minc, maxr, maxc = r.bbox

        sub_image = img[minr:maxr, minc:maxc]
        sub_mask = mask[minr:maxr, minc:maxc]

        lab_sub = measure.label(sub_mask, connectivity=2)
        sub_regions = measure.regionprops(lab_sub)

        if not sub_regions:
            return None, None, None, None, None, "Empty sub-region"

        largest_label = max(sub_regions, key=lambda rr: rr.area).label
        pup_clean = (lab_sub == largest_label)
        sub_mask_clean = binary_fill_holes(pup_clean).astype(bool)

        # image masquée
        sub_image_masked = sub_image * sub_mask_clean

        # global mask
        g = np.zeros(img.shape, dtype=bool)
        g[minr:maxr, minc:maxc] = sub_mask_clean

        # padding carré
        sub_mask_sq, cr = pad_to_square(sub_mask_clean)
        sub_image_sq, _ = pad_to_square(sub_image_masked)

        pos = np.array([minr, minc, maxr, maxc], dtype=int) 

        submasks.append(sub_mask_clean)
        pupilles_masked.append(sub_image_masked)
        global_masks.append(g)
        positions.append(pos)

    global_mask_sum = global_masks[0] | global_masks[1]

    return submasks, pupilles_masked, global_masks, positions, global_mask_sum, None


class CropMaskTuner:
    """
    GUI interactive pour ajuster le seuil `crop` et visualiser les 2 pupilles détectées.

    Comportement :
    - recalcul des masques seulement quand nécessaire ;
    - pendant le déplacement du slider, on met à jour la textbox sans relancer
      tout le pipeline à chaque micro-mouvement ;
    - les pupilles sont paddées au carré uniquement pour l'affichage, afin de
      garder une interface fluide et stable ;
    - si l'utilisateur confirme, show() retourne :
        (crop, submasks, global_masks, positions, pupilles_masked)
      sinon None.
    """

    def __init__(self, off_mask_image, initial_crop=0.25):
        self.img = np.asarray(off_mask_image)
        self.crop = float(np.clip(initial_crop, 0.0, 1.0))

        self.submasks = None
        self.pupilles_masked = None
        self.global_masks = None
        self.positions = None
        self.global_sum = None
        self.error = None

        self._confirmed = False
        self._result = None
        self._updating = False
        self._slider_dragging = False

        self._build_ui()
        self._recompute_and_refresh()

    def _build_ui(self):
        self.fig = plt.figure(figsize=(12, 6))
        try:
            self.fig.canvas.manager.set_window_title("Crop tuner - pupil masks")
        except Exception:
            pass

        gs = self.fig.add_gridspec(2, 3, height_ratios=[12, 2])

        self.ax_p1 = self.fig.add_subplot(gs[0, 0])
        self.ax_p2 = self.fig.add_subplot(gs[0, 1])
        self.ax_full = self.fig.add_subplot(gs[0, 2])

        ax_slider_bg = self.fig.add_subplot(gs[1, 0:2])
        ax_text_bg = self.fig.add_subplot(gs[1, 2])
        ax_slider_bg.set_axis_off()
        ax_text_bg.set_axis_off()

        slider_ax = self.fig.add_axes([0.08, 0.08, 0.55, 0.04])
        self.slider = Slider(
            slider_ax,
            "crop",
            0.0,
            1.0,
            valinit=self.crop,
            valstep=0.001
        )

        text_ax = self.fig.add_axes([0.70, 0.06, 0.10, 0.06])
        self.textbox = TextBox(text_ax, "crop", initial=f"{self.crop:.3f}")

        btn_ax = self.fig.add_axes([0.82, 0.06, 0.12, 0.06])
        self.btn_confirm = Button(btn_ax, "Confirm")

        self.ax_status = self.fig.add_axes([0.08, 0.01, 0.85, 0.04])
        self.ax_status.axis("off")
        self.status_text = self.ax_status.text(0.0, 0.5, "", va="center")

        for ax in (self.ax_p1, self.ax_p2, self.ax_full):
            ax.set_xticks([])
            ax.set_yticks([])

        self.ax_p1.set_title("Pupil 1")
        self.ax_p2.set_title("Pupil 2")
        self.ax_full.set_title("Complete image + global overlay")

        vmin = np.nanmin(self.img)
        vmax = np.nanmax(self.img)

        # Images initiales factices ; elles seront mises à jour ensuite avec set_data
        dummy = np.zeros((10, 10), dtype=float)

        self.im_p1 = self.ax_p1.imshow(dummy, vmin=vmin, vmax=vmax, interpolation="nearest")
        self.im_p2 = self.ax_p2.imshow(dummy, vmin=vmin, vmax=vmax, interpolation="nearest")

        self.im_full = self.ax_full.imshow(self.img, interpolation="nearest")

        dummy_overlay = np.ma.masked_where(
            np.ones_like(self.img, dtype=bool),
            np.zeros_like(self.img, dtype=float)
        )
        self.im_overlay = self.ax_full.imshow(
            dummy_overlay,
            alpha=0.35,
            interpolation="nearest"
        )

        self.txt_p1 = self.ax_p1.text(
            0.5, 0.5, "",
            ha="center", va="center",
            transform=self.ax_p1.transAxes
        )
        self.txt_p2 = self.ax_p2.text(
            0.5, 0.5, "",
            ha="center", va="center",
            transform=self.ax_p2.transAxes
        )
        self.txt_full = self.ax_full.text(
            0.5, 0.95, "",
            ha="center", va="top",
            transform=self.ax_full.transAxes,
            color="red"
        )

        self.slider.on_changed(self._on_slider)
        self.textbox.on_submit(self._on_text)
        self.btn_confirm.on_clicked(self._on_confirm)

        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)

    def _set_status(self, msg, ok=True):
        self.status_text.set_text(msg)
        self.status_text.set_color("black" if ok else "red")

    def _on_mouse_press(self, event):
        if event.inaxes == self.slider.ax:
            self._slider_dragging = True

    def _on_mouse_release(self, event):
        if self._slider_dragging:
            self._slider_dragging = False
            self._recompute_and_refresh()

    def _recompute_and_refresh(self):
        (
            self.submasks,
            self.pupilles_masked,
            self.global_masks,
            self.positions,
            self.global_sum,
            self.error
        ) = compute_pupil_masks(self.img, self.crop)

        ok = self.error is None

        self.btn_confirm.ax.set_visible(ok)
        self.btn_confirm.label.set_visible(ok)

        if ok:
            self._set_status(f"OK — 2 detected regions. crop={self.crop:.3f}", ok=True)
        else:
            self._set_status(
                f"Error — {self.error}. Adjust threshold. (crop={self.crop:.3f})",
                ok=False
            )

        self._refresh_views(ok)
        self.fig.canvas.draw_idle()

    def _refresh_views(self, ok):
        vmin = np.nanmin(self.img)
        vmax = np.nanmax(self.img)

        self.ax_p1.set_title("Pupil 1")
        self.ax_p2.set_title("Pupil 2")
        self.ax_full.set_title("Complete image + global overlay")

        if ok:
            # Padding seulement pour l'affichage
            p1_disp, _ = pad_to_square(self.pupilles_masked[0])
            p2_disp, _ = pad_to_square(self.pupilles_masked[1])

            self.im_p1.set_data(p1_disp)
            self.im_p2.set_data(p2_disp)
            self.im_p1.set_clim(vmin, vmax)
            self.im_p2.set_clim(vmin, vmax)

            overlay = np.ma.masked_where(~self.global_sum, self.global_sum.astype(float))
            self.im_overlay.set_data(overlay)

            self.txt_p1.set_text("")
            self.txt_p2.set_text("")
            self.txt_full.set_text("")
        else:
            dummy = np.zeros((10, 10), dtype=float)
            self.im_p1.set_data(dummy)
            self.im_p2.set_data(dummy)
            self.im_p1.set_clim(vmin, vmax)
            self.im_p2.set_clim(vmin, vmax)

            empty_overlay = np.ma.masked_where(
                np.ones_like(self.img, dtype=bool),
                np.zeros_like(self.img, dtype=float)
            )
            self.im_overlay.set_data(empty_overlay)

            self.txt_p1.set_text("N/A")
            self.txt_p2.set_text("N/A")
            self.txt_full.set_text("Invalid masks (<2 regions)")

    def _on_slider(self, val):
        if self._updating:
            return

        self._updating = True
        try:
            self.crop = float(np.clip(val, 0.0, 1.0))
            self.textbox.set_val(f"{self.crop:.3f}")
        finally:
            self._updating = False

        # Ne recalcule pas en boucle pendant le drag
        if not self._slider_dragging:
            self._recompute_and_refresh()

    def _on_text(self, text):
        if self._updating:
            return

        self._updating = True
        try:
            try:
                v = float(text)
            except ValueError:
                self._set_status(f"Invalid value: '{text}'", ok=False)
                return

            self.crop = float(np.clip(v, 0.0, 1.0))
            self.slider.set_val(self.crop)
            self.textbox.set_val(f"{self.crop:.3f}")
        finally:
            self._updating = False

        self._recompute_and_refresh()

    def _on_confirm(self, event):
        if self.error is not None:
            self._set_status("Invalid mask, impossible to confirm.", ok=False)
            return

        self._confirmed = True
        self._result = (
            self.crop,
            self.submasks,
            self.global_masks,
            self.positions,
            self.pupilles_masked
        )
        plt.close(self.fig)

    def _on_close(self, event):
        if not self._confirmed:
            self._result = None

    def show(self):
        try:
            self.fig.canvas.draw_idle()
            self.fig.show()
        except Exception:
            pass

        while plt.fignum_exists(self.fig.number) and self._result is None:
            plt.pause(0.05)

        return self._result
#%%
def load_data_papy(address_fits, return_normalisation = False, averaging = True):
    
    with fits.open(address_fits) as hdul:
        image = np.array(hdul[0].data)
    
    treated_image = image[:-1].astype(np.float32)
    count = treated_image[:,0,0].copy()
    treated_image[...,0,:] = 0
    treated_image[...,:,0] = 0
    bckg = image[-1].astype(np.float32)
    bckg[...,0,:] = 0
    bckg[...,:,0] = 0
    if averaging:
        treated_image = treated_image.mean(axis=0)-bckg
    
        normalisation = treated_image.sum()
        treated_image/=normalisation
    else:
        treated_image = treated_image - bckg
        normalisation = treated_image.sum(axis = (1,2))
        treated_image/=normalisation[:,None,None]
    if return_normalisation:
        return treated_image, normalisation, count
    else:
        return treated_image, count
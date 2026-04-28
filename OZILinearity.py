# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:41:12 2026

@author: mmotte

OZILin.py

"""

import os
import sys
import logging
import tkinter as tk
from tkinter import filedialog

import numpy as np
import tqdm
from skimage.transform import resize
from joblib import Parallel, delayed

from Pupil_selection import reference_intensities
from OOPAO.Zernike import Zernike

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

HERE = os.path.dirname(os.path.abspath(__file__))

if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    from parallel_utils import _reconstruct_phase_worker, _import_oopao_symbols
except ImportError as exc:
    raise ImportError(
        "Impossible d'importer parallel_utils. "
    ) from exc


class OZILin:
    """
    Analyse de fichiers .npz issus de mesures de linéarité OZIRIIS.

    Cette classe reprend uniquement les parties utiles de OZITele :
    - chargement des données de linéarité,
    - extraction des deux ZWFS,
    - initialisation OOPAO,
    - calcul des projecteurs,
    - reconstruction de phase,
    - conversion phase -> OPD,
    - projection des OPD sur les modes ou les fonctions d'influence.

    Elle ne contient volontairement aucune méthode temporelle ou PSD.
    """

    def __init__(
        self,
        tele_path: str = None,
        repetition_index: int = 0,
        is_onsky: bool = False,
        narrow_band: bool = False,
        extract_values: bool = True,
        subtract_dark: bool = False,
        normalize_images: bool = True,
        zero_first_row: bool = True,
        pupil_key: str = "pupil",
      
        command_indices = None,
        psf_sampling: float = 2.5354838709677416,
    ):
        """
        Parameters
        ----------
        tele_path : str, optional
            Chemin vers le fichier .npz de linéarité.
            Si None, ouvre une fenêtre de sélection.

        repetition_index : int
            Index de répétition à extraire dans data["linearity_images"].

        is_onsky : bool
            Utilisé uniquement pour garder une logique proche de OZITele
            lors de l'initialisation des sources et ZWFS.

        narrow_band : bool
            Si True, force une source bande étroite comme dans OZITele.

        extract_values : bool
            Si True, extrait les images ZWFS, initialise OOPAO et calcule
            les projecteurs.

        subtract_dark : bool
            Par défaut False, car les images de linéarité sont supposées
            avoir déjà le fond soustrait.
            Si True, soustrait data["cred_dark"].

        normalize_images : bool
            Si True, normalise chaque image par sa somme.
            C'est cohérent avec OZITele, mais peut être désactivé pour une
            analyse photométrique brute.

        zero_first_row : bool
            Si True, met à zéro la première ligne caméra, comme dans OZITele.
            Par défaut False, car ce comportement est spécifique aux données
            de télémétrie et ne doit pas être imposé silencieusement.

        pupil_key : str
            Clé utilisée pour définir le masque pupille/global.
            Par défaut "pupil", suivant ton code minimal.
            Peut être remplacé par "validMask" si c'est ce masque qui
            correspond réellement aux pixels valides des deux ZWFS.

        psf_sampling : float
            Sampling utilisé par le détecteur OOPAO.
        """
        self.tag = "ozilin"
        self.is_onsky = bool(is_onsky)
        self.is_nb = bool(narrow_band)
        self.repetition_index = int(repetition_index)
        self.subtract_dark = bool(subtract_dark)
        self.normalize_images = bool(normalize_images)
        self.zero_first_row = bool(zero_first_row)
        self.psf_sampling = float(psf_sampling)
        self.command_indices_input = command_indices
        self.has_reconstructed_phase = False
        self.has_projected_phase = False

        if tele_path is None:
            tele_path = self._choose_file()
            if not tele_path:
                raise ValueError("Aucun fichier sélectionné.")

        self.tele_path = tele_path

        self._load_npz(pupil_key=pupil_key)
        self._prepare_images()

        if extract_values:
            self._prepare_pupils()
            self._initialise_OOPAO_objects()
            self.compute_projectors()
            self.extract_Zimages()
            self.compute_reconstructor()
            self._compute_synth_IM()

    # ------------------------------------------------------------------
    # Chargement et validation
    # ------------------------------------------------------------------
    
        
    def _load_npz(self, pupil_key: str = "pupil"):
        """
        Charge le fichier .npz et vérifie les clés et dimensions attendues.
        """
        data = np.load(self.tele_path, allow_pickle=True)
        self.data_keys = list(data.keys())

        required_keys = [
            "linearity_images",
            "m2c",
            "stroke",
            "pupil",
            "ref",
            "intmat",
        ]

        missing = [key for key in required_keys if key not in data]
        if missing:
            raise KeyError(
                f"Clés manquantes dans le fichier .npz : {missing}. "
                f"Clés disponibles : {self.data_keys}"
            )

        if pupil_key not in data:
            raise KeyError(
                f"pupil_key='{pupil_key}' absent du fichier. "
                f"Clés disponibles : {self.data_keys}"
            )

        linearity_images = np.asarray(data["linearity_images"])

        if linearity_images.ndim != 5:
            raise ValueError(
                "data['linearity_images'] doit avoir 5 dimensions : "
                "(n_repetitions, n_commands, n_strokes, ny, nx). "
                f"Shape reçue : {linearity_images.shape}"
            )

        n_rep, n_cmd_total, n_stroke, ny, nx = linearity_images.shape

        self.command_indices = self._parse_command_indices(
            self.command_indices_input,
            n_cmd_total,
        )
        n_cmd = self.command_indices.size
        if not (0 <= self.repetition_index < n_rep):
            raise IndexError(
                f"repetition_index={self.repetition_index} invalide. "
                f"Le fichier contient {n_rep} répétitions, donc l'index doit "
                f"être entre 0 et {n_rep - 1}."
            )

        self.linearity_shape = {
            "n_repetitions": n_rep,
            "n_commands": n_cmd,
            "n_strokes": n_stroke,
            "ny": ny,
            "nx": nx,
        }

        # Hypothèse explicite :
        # data["linearity_images"][repetition_index] est interprété comme :
        #     (n_commands, n_strokes, ny, nx)
        # où n_commands peut désigner des modes ou des actionneurs selon la
        # matrice m2c utilisée pendant l'acquisition.
        self.img = linearity_images[self.repetition_index, self.command_indices, :, :, :].astype(np.float32)

        self.M2C = np.asarray(data["m2c"], dtype=np.float32)
        if self.M2C.ndim != 2:
            raise ValueError(
                f"data['m2c'] doit être une matrice 2D. Shape reçue : {self.M2C.shape}"
            )

        self.C2M = np.linalg.pinv(self.M2C).astype(np.float32)

        self.stroke = np.asarray(data["stroke"], dtype=np.float32)
        self.intmat = np.asarray(data["intmat"], dtype=np.float32)
        self.ref = np.asarray(data["ref"], dtype=np.float32)

        self.off_mask = np.asarray(data[pupil_key], dtype=np.float32)

        if self.off_mask.ndim != 2:
            raise ValueError(
                f"data['{pupil_key}'] doit être un masque 2D. "
                f"Shape reçue : {self.off_mask.shape}"
            )

        self.validMask = None
        if "validMask" in data:
            self.validMask = np.asarray(data["validMask"], dtype=np.float32)

        self.dark = None
        if "cred_dark" in data:
            self.dark = np.asarray(data["cred_dark"], dtype=np.float32)

        if self.subtract_dark and self.dark is None:
            raise KeyError(
                "subtract_dark=True demandé, mais la clé 'cred_dark' est absente."
            )

        self.n_commands = n_cmd
        self.n_strokes = n_stroke
        self.image_shape = (ny, nx)

        logger.info(
            "Fichier chargé : %s | linearity_images[%d].shape = %s",
            self.tele_path,
            self.repetition_index,
            self.img.shape,
        )

    def _prepare_images(self):
        """
        Prépare self.img à partir du cube de linéarité sélectionné.

        self.img.shape :
            (n_commands, n_strokes, ny, nx)

        self.img.shape :
            même shape.

        self.img_flat.shape :
            (n_commands * n_strokes, ny, nx)

        L'aplatissement est uniquement technique. Il ne crée pas une notion
        de temps. C'est juste une liste d'images indépendantes à reconstruire.
        """
        

        if self.subtract_dark:
            self._subtract_dark()

        if self.zero_first_row:
            self.img[..., :1, :] = 0.0

        if self.normalize_images:
            self._normalize_images()

        self.img_flat = self.img.reshape(
            self.n_commands * self.n_strokes,
            self.image_shape[0],
            self.image_shape[1],
        )

    def _subtract_dark(self):
        """
        Soustrait explicitement cred_dark si demandé.

        Par défaut, cette méthode n'est pas appelée, car les images de
        linéarité sont supposées avoir déjà le fond soustrait.
        """
        dark = np.asarray(self.dark, dtype=np.float32)

        if dark.shape == self.image_shape:
            self.img -= dark[None, None, :, :]
        elif dark.shape == self.img.shape:
            self.img -= dark
        else:
            raise ValueError(
                "Shape incompatible pour cred_dark. "
                f"dark.shape={dark.shape}, image_shape={self.image_shape}, "
                f"img.shape={self.img.shape}"
            )

    def _normalize_images(self):
        """
        Normalise chaque image par sa somme.

        Protège contre les images de somme nulle ou non finie.
        """
        sums = self.img.sum(axis=(-2, -1), keepdims=True)

        bad = (~np.isfinite(sums)) | (np.abs(sums) < 1e-30)
        if np.any(bad):
            logger.warning(
                "%d images ont une somme nulle ou non finie. "
                "Elles ne seront pas normalisées correctement.",
                int(np.sum(bad)),
            )

        sums_safe = np.where(bad, 1.0, sums)
        self.img = self.img / sums_safe

    # ------------------------------------------------------------------
    # Préparation pupilles et ZWFS
    # ------------------------------------------------------------------
    
    def _prepare_pupils(self):
        """
        Sélectionne les deux pupilles ZWFS à partir du masque fourni.

        Cette méthode reprend la logique de OZITele :
        - reference_intensities donne les régions associées aux deux ZWFS,
        - la deuxième pupille est redimensionnée sur la première,
        - les deux masques sont paddés en carré.

        Hypothèse importante :
            self.off_mask doit représenter un masque compatible avec
            reference_intensities, c'est-à-dire une image caméra globale
            contenant les régions valides des deux ZWFS.
        """
        (
            self.initial_positions,
            self.initial_pupils,
            self.initial_submasks,
            self.global_masks,
        ) = reference_intensities(self.off_mask)

        if len(self.initial_pupils) < 2:
            raise RuntimeError(
                "reference_intensities n'a pas retourné deux pupilles. "
                "Vérifie que pupil_key pointe vers le bon masque caméra."
            )

        self.submasks = [None, None]
        self.pupils = [None, None]

        pupil2 = self._rescale_matrix(
            self.initial_pupils[1],
            self.initial_pupils[0].shape[0],
            self.initial_pupils[0].shape[1],
        )

        self.pupils[0], _ = self._pad_to_square(self.initial_pupils[0])
        self.pupils[1], _ = self._pad_to_square(pupil2)

        # Même convention que OZITele :
        # on force les deux submasks au support de la pupille 1.
        self.submasks[0], _ = self._pad_to_square(self.initial_pupils[0])
        self.submasks[1] = self.submasks[0].copy()

        self.pupils[0] = self.pupils[0].astype(np.float32)
        self.pupils[1] = self.pupils[1].astype(np.float32)
        self.submasks[0] = self.submasks[0].astype(bool)
        self.submasks[1] = self.submasks[1].astype(bool)

    def extract_Zimages(self):
        """
        Extrait les images correspondant aux deux ZWFS.

        Entrée :
            self.img_flat.shape = (n_commands * n_strokes, ny, nx)

        Sorties :
            self.img_ZWFS1_flat.shape = (n_images, npix, npix)
            self.img_ZWFS2_flat.shape = (n_images, npix, npix)

            self.img_ZWFS1.shape = (n_commands, n_strokes, npix, npix)
            self.img_ZWFS2.shape = (n_commands, n_strokes, npix, npix)
        """
        n_images = self.img_flat.shape[0]

        z1_shape = self.initial_submasks[0].shape
        z2_shape = self.initial_submasks[1].shape

        images_z1 = np.zeros((n_images, *z1_shape), dtype=np.float32)
        images_z2 = np.zeros((n_images, *z2_shape), dtype=np.float32)

        try:
            images_z1[:, self.initial_submasks[0]] = self.img_flat[:, self.global_masks[0]]
            images_z2[:, self.initial_submasks[1]] = self.img_flat[:, self.global_masks[1]]
        except IndexError as exc:
            raise IndexError(
                "Erreur pendant l'extraction des ZWFS. "
                "Les masques issus de reference_intensities ne sont pas compatibles "
                "avec la shape des images. "
                f"img_flat.shape={self.img_flat.shape}, "
                f"off_mask.shape={self.off_mask.shape}"
            ) from exc

        logger.info("Extraction des images des deux ZWFS.")

        img_ZWFS2 = []
        for i in tqdm.tqdm(range(n_images), desc="Rescale ZWFS2"):
            img_ZWFS2.append(
                self._rescale_matrix(
                    images_z2[i],
                    self.pupils[0].shape[0],
                    self.pupils[0].shape[1],
                )
            )

        self.img_ZWFS2_flat = np.asarray(img_ZWFS2, dtype=np.float32)
        self.img_ZWFS2_flat, _ = self._pad_to_square(self.img_ZWFS2_flat)

        self.img_ZWFS1_flat, _ = self._pad_to_square(images_z1)
        self.img_ZWFS1_flat = self.img_ZWFS1_flat.astype(np.float32)

        spatial_shape = self.img_ZWFS1_flat.shape[-2:]

        self.img_ZWFS1 = self.img_ZWFS1_flat.reshape(
            self.n_commands,
            self.n_strokes,
            *spatial_shape,
        )

        self.img_ZWFS2 = self.img_ZWFS2_flat.reshape(
            self.n_commands,
            self.n_strokes,
            *spatial_shape,
        )

    # ------------------------------------------------------------------
    # OOPAO
    # ------------------------------------------------------------------

    def _initialise_OOPAO_objects(self):
        """
        Initialise les objets OOPAO nécessaires à la reconstruction.

        Cette méthode reprend la logique de OZITele, mais sans dépendance
        temporelle.
        """
        (
            Source,
            Telescope,
            ZWFS,
            ZWFS2,
            DeformableMirror,
            MisRegistration,
            Detector,
        ) = _import_oopao_symbols()

        if self.is_onsky and not self.is_nb:
            wavelength = 1.6e-6
            bandwidth = 0.2e-6
        else:
            wavelength = 1.550e-6
            bandwidth = 0.0

        self.src1 = Source(optBand="H", magnitude=-2.5)
        self.src1.wavelength = wavelength
        self.src1.bandwidth = bandwidth

        self.src2 = Source(optBand="H", magnitude=-2.5)
        self.src2.wavelength = wavelength
        self.src2.bandwidth = bandwidth

        self.tel1 = Telescope(
            self.submasks[0].shape[0],
            1.52,
            pupil=self.submasks[0],
        )

        self.tel1.pupilReflectivity = np.sqrt(self.pupils[0])
        self.tel1.pupilReflectivity[~np.isfinite(self.tel1.pupilReflectivity)] = 0.0
        self.src1 * self.tel1

        self.tel2 = Telescope(
            self.submasks[1].shape[0],
            1.52,
            pupil=self.submasks[1],
        )

        self.tel2.pupilReflectivity = np.sqrt(self.pupils[1])
        self.tel2.pupilReflectivity[~np.isfinite(self.tel2.pupilReflectivity)] = 0.0
        self.src2 * self.tel2

        self.vzwfs = self._build_vzwfs_class()
        self.zwfs1 = self.vzwfs.zwfs1
        self.zwfs2 = self.vzwfs.zwfs2

        self.cam = Detector(psf_sampling=self.psf_sampling)

        misreg_path = os.path.join(HERE, "dm_second_stage_misreg_dict.npy")
        if not os.path.exists(misreg_path):
            raise FileNotFoundError(
                f"Fichier de misregistration introuvable : {misreg_path}"
            )

        param = np.load(misreg_path, allow_pickle=True).item()
        m = MisRegistration(param)

        self.dm1 = DeformableMirror(
            telescope=self.tel1,
            nSubap=10,
            mechCoupling=0.35,
            print_dm_properties=False,
            pitch=0.11,
            misReg=m,
            sign=-1e-5,
        )

        self.dm2 = DeformableMirror(
            telescope=self.tel2,
            nSubap=10,
            mechCoupling=0.35,
            print_dm_properties=False,
            pitch=0.11,
            misReg=m,
            sign=-1e-5,
        )

        if_path = os.path.join(HERE, "IF_vZWFS.npy")
        if not os.path.exists(if_path):
            raise FileNotFoundError(
                f"Fichier d'influence functions introuvable : {if_path}"
            )

        IF = np.load(if_path)
        if IF[0,...].shape != self.tel1.pupil.shape:
            IF = self._rescale_matrix(IF, self.tel1.pupil.shape[0], self.tel1.pupil.shape[1])
        self.IF = IF.reshape(97, -1).T.astype(np.float32) 
        if self.M2C.shape[0] != self.IF.shape[1]:
            raise ValueError(
                "Shape incohérente entre M2C et IF. "
                f"M2C.shape={self.M2C.shape}, IF.shape={self.IF.shape}. "
                "On attend généralement M2C.shape[0] == 97."
            )

        self.M2phase = self.IF @ self.M2C
        self.modes_std = self.M2phase.std(axis=0)
        self.IF_std = self.IF.std(axis=0)

        amplitude_mean = np.ptp(self.IF, axis=0)

        # self.dm1.modes *= amplitude_mean / np.ptp(self.dm1.modes, axis=0)
        # self.dm2.modes *= amplitude_mean / np.ptp(self.dm2.modes, axis=0)
        self.dm1.modes = self.IF.copy()
        self.dm2.modes = self.IF.copy()
    def _build_vzwfs_class(self):
        """
        Construit l'objet double ZWFS.

        Même convention que OZITele :
        - diamètre 2 si onsky,
        - diamètre 2.14 sinon.
        """
        (
            Source,
            Telescope,
            ZWFS,
            ZWFS2,
            DeformableMirror,
            MisRegistration,
            Detector,
        ) = _import_oopao_symbols()

        diam = 2.0 if self.is_onsky else 2.14

        zwfs1 = ZWFS(
            self.tel1,
            diameter=diam,
            phase_shift=0.33,
            zpf=30,
            phase_shift_unit="pi",
        )

        zwfs2 = ZWFS(
            self.tel2,
            diameter=diam,
            phase_shift=-0.74,
            zpf=30,
            phase_shift_unit="pi",
        )

        return ZWFS2(ZWFS1=zwfs1, ZWFS2=zwfs2)

    def _export_reconstruction_setup(self):
        """
        Données minimales nécessaires à la reconstruction parallèle.
        """
        return {
            "is_onsky": self.is_onsky,
            "is_nb": self.is_nb,
            "submask0": self.submasks[0],
            "submask1": self.submasks[1],
            "pupil0": self.pupils[0],
            "pupil1": self.pupils[1],
        }

    # ------------------------------------------------------------------
    # Projecteurs
    # ------------------------------------------------------------------

    def compute_projectors(self):
        """
        Calcule les projecteurs :
        - self.proj_M2C : projection sur les modes définis par M2C,
        - self.proj_IF : projection sur les fonctions d'influence.
        """
        self.proj_M2C = self._compute_proj_dm(self.M2C, self.tel1, self.dm1)

        n_if = self.dm1.modes.shape[-1]
        self.proj_IF = self._compute_proj_dm(
            np.identity(n_if, dtype=np.float32),
            self.tel1,
            self.dm1,
        )

    def _compute_proj_dm(self, modal_basis, tel, dm):
        """
        Calcule un projecteur à partir des modes DM générés dans OOPAO.
        """
        modal_basis = np.asarray(modal_basis, dtype=np.float32)

        if modal_basis.ndim != 2:
            raise ValueError(
                f"modal_basis doit être 2D. Shape reçue : {modal_basis.shape}"
            )

        dm.coefs = modal_basis
        tel * dm

        modes = tel.OPD.copy()
        modes = modes.reshape((tel.resolution**2, modes.shape[-1]))

        std = tel.OPD[tel.pupil, :].std(axis=0)
        std = np.where(np.abs(std) < 1e-30, 1.0, std)

        modes = modes / std[None, :]

        cov_modes = modes.T @ modes
        diag = np.diag(cov_modes)
        diag = np.where(np.abs(diag) < 1e-30, 1.0, diag)

        return (np.diag(1.0 / diag) @ modes.T).astype(np.float32)

    def _compute_proj_OPDs(self, modes, tel):
        """
        Calcule un projecteur à partir de modes OPD déjà définis.
        """
        modes = np.asarray(modes, dtype=np.float32)

        if modes.ndim != 3:
            raise ValueError(
                "modes doit avoir la shape (resolution, resolution, n_modes). "
                f"Shape reçue : {modes.shape}"
            )

        if modes.shape[:2] != tel.pupil.shape:
            raise ValueError(
                "Shape spatiale des modes incompatible avec la pupille. "
                f"modes.shape[:2]={modes.shape[:2]}, pupil.shape={tel.pupil.shape}"
            )

        std = modes[tel.pupil, :].std(axis=0)
        std = np.where(np.abs(std) < 1e-30, 1.0, std)

        modes_flat = modes.reshape((tel.resolution**2, modes.shape[-1]))
        modes_flat = modes_flat / std[None, :]

        cov_modes = modes_flat.T @ modes_flat
        diag = np.diag(cov_modes)
        diag = np.where(np.abs(diag) < 1e-30, 1.0, diag)

        return (np.diag(1.0 / diag) @ modes_flat.T).astype(np.float32)

    def compute_Zernike_basis(self, nmodes: int = 30):
        """
        Calcule une base de Zernike et son projecteur.
        """
        Zer_basis = Zernike(self.tel1, J=nmodes)
        Zer_basis.computeZernike(self.tel1)

        self.Zer_modes = Zer_basis.modesFullRes.copy().astype(np.float32)
        self.proj_Zer = self._compute_proj_OPDs(self.Zer_modes, self.tel1)

    # ------------------------------------------------------------------
    # Reconstruction phase / OPD
    # ------------------------------------------------------------------
    def compute_reconstructor(self):
        self.IM1 = self.intmat[self.global_masks[0],:]
        self.IM2 = self.intmat[self.global_masks[1],:]
        self.rec1 = np.linalg.pinv(self.IM1)
        self.rec2 = np.linalg.pinv(self.IM2)
        
    def _reconstruct_cmd(self, reconstructor,signal):
        reconstructed = np.zeros((self.img.shape[0],self.img.shape[1],self.M2C.shape[-1])).astype(np.float32)
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                reconstructed[i,j]= reconstructor@signal[i,j]
        return reconstructed
    def reconstructed_cmd(self):
        
        self.rec_cmd1 = self._reconstruct_cmd(self.rec1, self.img[:,:,self.global_masks[0]])
        self.rec_cmd1 -=self.rec_cmd1[:,self.stroke==0,:]
        
        self.rec_cmd2 = self._reconstruct_cmd(self.rec2,self.img[:,:,self.global_masks[1]])
        self.rec_cmd2 -=self.rec_cmd2[:,self.stroke==0,:]        
        
        self.synth_rec_cmd1 = self._reconstruct_cmd(self.synth_rec1,self.img_ZWFS1[:,:,self.submasks[0]])
        self.synth_rec_cmd1 -=self.synth_rec_cmd1[:,self.stroke==0,:]        
        
        self.synth_rec_cmd2 = self._reconstruct_cmd(self.synth_rec2,self.img_ZWFS2[:,:,self.submasks[1]])
        self.synth_rec_cmd2 -=self.synth_rec_cmd2[:,self.stroke==0,:]  
        
    def _compute_synth_IM(self,stroke_nm = 12):
        M2Phase = self.IF*1e9 @ self.M2C
        
        # Std across actuators? You used axis=0. Keep as-is, but at least be consistent.
        std_phase = np.std(M2Phase, axis=0)

        # Use stroke_nm instead of hardcoded 12
        # Guard against division by zero
        
        M2C = self.M2C.copy()
        stroke = stroke_nm/std_phase#np.ones(M2C.shape[1])*0.0001
        IM1 = []
        IM2 = []
        
        for i in range(M2C.shape[-1]):
            self.dm1.coefs = stroke[i]*M2C[:,i]
            self.dm2.coefs = stroke[i]*M2C[:,i]
            self.tel1*self.dm1
            self.tel1*self.zwfs1
            img1_pos = self.zwfs1.signal
            self.tel2*self.dm2
            self.tel2*self.zwfs2
            img2_pos = self.zwfs2.signal
            self.dm1.coefs = -stroke[i]*M2C[:,i]
            self.dm2.coefs = -stroke[i]*M2C[:,i]
            self.tel1*self.dm1
            self.tel1*self.zwfs1
            IM1.append((img1_pos - self.zwfs1.signal)/(2*stroke[i]))
            self.tel2*self.dm2
            self.tel2*self.zwfs2
            IM2.append((img2_pos - self.zwfs2.signal)/(2*stroke[i]))
        self.synth_IM1 = np.array(IM1).T
        self.synth_IM2 = np.array(IM2).T
        self.synth_rec1 = np.linalg.pinv(self.synth_IM1)
        self.synth_rec2 = np.linalg.pinv(self.synth_IM2)
    
    def reconstruct_phase(
        self,
        im1,
        im2,
        method: str = "atan",
        damping: float = 0.5,
        iteration: int = 10,
    ):
        """
        Reconstruit une phase à partir d'une paire d'images ZWFS.
        """
        self.vzwfs.zwfs1.img_ZWFS = im1
        self.vzwfs.zwfs2.img_ZWFS = im2

        return self.vzwfs.reconstructor(
            iteration=iteration,
            damping_iteration=damping,
            reconstructor=method,
        )

    def reconstruct_all_phase(
        self,
        method: str = "atan",
        iteration: int = 10,
        damping: float = 0.5,
        parallel: bool = True,
        parall_njob: int = 4,
    ):
        """
        Reconstruit la phase pour toutes les images de linéarité.

        Important :
            L'ordre reconstruit est :
                image_index = command_index * n_strokes + stroke_index

            Puis le résultat est remis sous forme :
                self.phase.shape = (n_commands, n_strokes, npix, npix)

        Il n'y a toujours aucune notion de temps ici.
        """
        if not hasattr(self, "img_ZWFS1_flat") or not hasattr(self, "img_ZWFS2_flat"):
            raise RuntimeError(
                "Les images ZWFS ne sont pas extraites. "
                "Appelle extract_Zimages() avant reconstruct_all_phase()."
            )

        n_images = self.img_ZWFS1_flat.shape[0]

        if parallel:
            setup = self._export_reconstruction_setup()

            gen = Parallel(
                n_jobs=parall_njob,
                prefer="processes",
                return_as="generator",
            )(
                delayed(_reconstruct_phase_worker)(
                    self.img_ZWFS1_flat[i],
                    self.img_ZWFS2_flat[i],
                    setup,
                    method,
                    damping,
                    iteration,
                )
                for i in range(n_images)
            )

            phase_flat = np.asarray(
                list(
                    tqdm.tqdm(
                        gen,
                        total=n_images,
                        desc=f"Phase reconstruction ({method})",
                    )
                ),
                dtype=np.float32,
            )

        else:
            phase_flat = np.zeros(
                (
                    n_images,
                    self.tel1.pupil.shape[0],
                    self.tel1.pupil.shape[1],
                ),
                dtype=np.float32,
            )

            for i in tqdm.tqdm(
                range(n_images),
                desc=f"Phase reconstruction ({method})",
            ):
                phase_flat[i] = self.reconstruct_phase(
                    self.img_ZWFS1_flat[i],
                    self.img_ZWFS2_flat[i],
                    method=method,
                    damping=damping,
                    iteration=iteration,
                ).astype(np.float32)

        self.phase_flat = phase_flat

        self.phase = phase_flat.reshape(
            self.n_commands,
            self.n_strokes,
            phase_flat.shape[-2],
            phase_flat.shape[-1],
        )
        self.phase -=self.phase[:,self.stroke==0,:,:]
        self._phase2OPD()
        self.has_reconstructed_phase = True

    def _phase2OPD(self, phase=None):
        """
        Convertit une phase en OPD.

        phase en radians :
            OPD = phase / (2*pi) * wavelength

        Si phase=None, utilise self.phase.
        """
        if phase is None:
            if not hasattr(self, "phase"):
                raise RuntimeError("Aucune phase disponible.")
            phase = self.phase

        self.OPDs = (phase / (2.0 * np.pi) * self.src1.wavelength).astype(np.float32)
        self.OPDs_flat = self.OPDs.reshape(
            self.n_commands * self.n_strokes,
            self.OPDs.shape[-2],
            self.OPDs.shape[-1],
        )

    # ------------------------------------------------------------------
    # Projections
    # ------------------------------------------------------------------

    def project_OPDs(self):
        """
        Projette les OPD reconstruites sur :
        - les fonctions d'influence,
        - les modes M2C.

        Sorties :
            self.OPDs_on_IFs.shape = (n_commands, n_strokes, n_IF)
            self.OPDs_on_modes.shape = (n_commands, n_strokes, n_modes)
        """
        if not self.has_reconstructed_phase:
            raise RuntimeError("Il faut reconstruire la phase avant de projeter les OPD.")

        opd_if_flat = self._project_maps(self.proj_IF, self.OPDs_flat)
        opd_modes_flat = self._project_maps(self.proj_M2C, self.OPDs_flat)

        self.OPDs_on_IFs = opd_if_flat.reshape(
            self.n_commands,
            self.n_strokes,
            opd_if_flat.shape[-1],
        )

        self.OPDs_on_modes = opd_modes_flat.reshape(
            self.n_commands,
            self.n_strokes,
            opd_modes_flat.shape[-1],
        )

        self.has_projected_phase = True

    def project_OPDs_on_projector(self, projector, name: str = None):
        """
        Projette les OPD sur un projecteur externe.

        Parameters
        ----------
        projector : ndarray
            Shape attendue : (n_coefficients, n_pixels)

        name : str, optional
            Si fourni, stocke le résultat dans self.<name>.

        Returns
        -------
        coeffs : ndarray
            Shape : (n_commands, n_strokes, n_coefficients)
        """
        if not self.has_reconstructed_phase:
            raise RuntimeError("Il faut reconstruire la phase avant projection.")

        coeffs_flat = self._project_maps(projector, self.OPDs_flat)

        coeffs = coeffs_flat.reshape(
            self.n_commands,
            self.n_strokes,
            coeffs_flat.shape[-1],
        )

        if name is not None:
            setattr(self, name, coeffs)

        return coeffs

    def _project_maps(self, projector, maps):
        """
        Applique un projecteur à une pile de cartes 2D.

        Parameters
        ----------
        projector : ndarray
            Shape : (n_coefficients, n_pixels)

        maps : ndarray
            Shape : (n_maps, ny, nx)

        Returns
        -------
        projected : ndarray
            Shape : (n_maps, n_coefficients)
        """
        projector = np.asarray(projector, dtype=np.float32)
        maps = np.asarray(maps, dtype=np.float32)

        if projector.ndim != 2:
            raise ValueError(
                f"projector doit être 2D. Shape reçue : {projector.shape}"
            )

        if maps.ndim != 3:
            raise ValueError(
                f"maps doit être 3D : (n_maps, ny, nx). Shape reçue : {maps.shape}"
            )

        n_pixels = maps.shape[-2] * maps.shape[-1]

        if projector.shape[1] != n_pixels:
            raise ValueError(
                "Shape incompatible entre projector et maps. "
                f"projector.shape={projector.shape}, "
                f"maps spatial pixels={n_pixels}."
            )

        projected = np.zeros(
            (maps.shape[0], projector.shape[0]),
            dtype=np.float32,
        )

        for i in tqdm.tqdm(range(maps.shape[0]), desc="Projection OPD"):
            projected[i] = projector @ maps[i].ravel()

        return projected

    # ------------------------------------------------------------------
    # Outils numériques
    # ------------------------------------------------------------------

    def _rescale_matrix(self, A, j, k, anti_aliasing=True):
        """
        Redimensionne un tableau 2D, 3D ou 4D vers une shape spatiale cible.

        Les deux dernières dimensions sont considérées comme spatiales.
        """
        A = np.asarray(A)

        if A.ndim == 2:
            return resize(A, (j, k), order=5, anti_aliasing=anti_aliasing)

        if A.ndim == 3:
            return resize(
                A,
                (A.shape[0], j, k),
                order=5,
                anti_aliasing=anti_aliasing,
            )

        if A.ndim == 4:
            return resize(
                A,
                (A.shape[0], A.shape[1], j, k),
                order=5,
                anti_aliasing=anti_aliasing,
            )

        raise ValueError(
            "_rescale_matrix accepte seulement des tableaux 2D, 3D ou 4D. "
            f"Shape reçue : {A.shape}"
        )

    def _pad_to_square(self, arr: np.ndarray):
        """
        Pad un tableau pour obtenir des dimensions spatiales carrées.

        Accepte :
            2D : (ny, nx)
            3D : (n, ny, nx)
            4D : (n1, n2, ny, nx)
        """
        arr = np.asarray(arr)

        if arr.ndim < 2:
            raise ValueError(
                f"_pad_to_square attend au moins 2 dimensions. Shape reçue : {arr.shape}"
            )

        M, N = arr.shape[-2:]
        size = max(M, N)

        pad_top = (size - M) // 2
        pad_bottom = size - M - pad_top
        pad_left = (size - N) // 2
        pad_right = size - N - pad_left

        pad_width = [(0, 0)] * arr.ndim
        pad_width[-2] = (pad_top, pad_bottom)
        pad_width[-1] = (pad_left, pad_right)

        if np.issubdtype(arr.dtype, np.bool_):
            const_value = False
        elif np.issubdtype(arr.dtype, np.integer):
            const_value = 0
        elif np.issubdtype(arr.dtype, np.floating):
            const_value = 0.0
        else:
            raise TypeError(f"Type non supporté pour padding : {arr.dtype}")

        padded = np.pad(
            arr,
            pad_width=pad_width,
            mode="constant",
            constant_values=const_value,
        )

        padded_cr = [-pad_bottom, -pad_left, pad_top, pad_right]

        return padded, padded_cr

    def _choose_file(self):
        """
        Ouvre une fenêtre de sélection de fichier .npz.
        """
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename(
            title="Select OZIRIIS linearity .npz file",
            filetypes=[
                ("NPZ files", "*.npz"),
                ("All files", "*.*"),
            ],
        )

        root.destroy()
        return file_path

    def _delete_img(self):
        """
        Supprime les gros cubes images de la mémoire.

        À utiliser seulement après reconstruction/projection si tu veux
        libérer de la RAM.
        """
        for attr in [
            "img_raw",
            "img",
            "img_flat",
            "img_ZWFS1",
            "img_ZWFS2",
            "img_ZWFS1_flat",
            "img_ZWFS2_flat",
        ]:
            if hasattr(self, attr):
                delattr(self, attr)
    def _parse_command_indices(self, command_indices, n_commands_total):
        """
        Convertit command_indices en indices numpy valides.
    
        Parameters
        ----------
        command_indices : None, int, list[int], tuple[int], ndarray, slice
            Indices des modes/actionneurs à analyser.
    
            - None : garde tous les modes/actionneurs
            - int : garde un seul mode/actionneur
            - list/tuple/ndarray : garde plusieurs modes/actionneurs
            - slice : garde une tranche, par exemple slice(0, 10)
    
        n_commands_total : int
            Nombre total de modes/actionneurs dans data["linearity_images"].
    
        Returns
        -------
        indices : ndarray
            Indices entiers sélectionnés.
        """
        if command_indices is None:
            indices = np.arange(n_commands_total)
    
        elif isinstance(command_indices, slice):
            indices = np.arange(n_commands_total)[command_indices]
    
        elif isinstance(command_indices, (int, np.integer)):
            indices = np.array([int(command_indices)])
    
        else:
            indices = np.asarray(command_indices, dtype=int).ravel()
    
        if indices.size == 0:
            raise ValueError(
                "command_indices ne sélectionne aucun mode/actionneur."
            )
    
        bad = (indices < 0) | (indices >= n_commands_total)
        if np.any(bad):
            raise IndexError(
                "Certains command_indices sont hors limites. "
                f"indices invalides : {indices[bad].tolist()}. "
                f"Le fichier contient {n_commands_total} modes/actionneurs, "
                f"indices valides : 0 à {n_commands_total - 1}."
            )
    
        return indices
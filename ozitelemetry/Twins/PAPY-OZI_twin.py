# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 10:40:42 2023

Accurate version of PAPYRUS AO System used for reproducing the real system in details.
- 17/06/2025: Update after change of the WFS camera.
- 23/06/2025: Update to prepare the integration with DAO

@author: cheritie - astriffl
"""
import time
import matplotlib.pyplot as plt
import numpy as np
from Twins.OZIRIIS import OZIRIIS
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap
from Twins.Papyrus import Papyrus

from Twins.parallel_utils_twin import _reconstruct_phase_worker, _import_oopao_symbols, _simulate_psf_chunk_worker
from utils import compute_etf
import tqdm
def project_opd_between_pupils(
    OZItwin,
    OPDs,
    input_pupil=None,
    output_pupil=None,
    output_shape=None,
    dtype=np.float32,
):
    """
    Crop une série d'OPD sur une pupille d'entrée, la rescale sur la pupille de sortie,
    puis la réinjecte dans une matrice complète définie par output_shape.

    Parameters
    ----------
    OZItwin : object
        Objet contenant les méthodes _crop_opd, _crop_pupil et _rescale_matrix.

    OPDs : np.ndarray
        OPD 2D ou 3D à projeter.
        Shape attendue :
            - (n_frames, nx, ny)
            - ou (nx, ny)

    input_pupil : np.ndarray, optional
        Pupille utilisée pour cropper OPDs.
        Si None, utilise le comportement par défaut de OZItwin._crop_opd.

    output_pupil : np.ndarray, optional
        Pupille dans laquelle réinjecter l'OPD après rescale.
        Si None, utilise OZItwin.tel.pupil.

    output_shape : tuple, optional
        Shape finale de sortie.
        Si None, utilise output_pupil.shape.

    dtype : type
        Type de sortie.

    Returns
    -------
    OPDs_projected : np.ndarray
        OPD projetée dans la pupille de sortie.
    """

    # Sécurité basique, parce que sinon tu vas encore te battre avec des shapes.
    OPDs = np.asarray(OPDs)

    if output_pupil is None:
        output_pupil = OZItwin.tel.pupil

    output_pupil = output_pupil.astype(bool)

    if output_shape is None:
        output_shape = output_pupil.shape

    # Crop de l'OPD source
    if input_pupil is None:
        _, opd_cropped = OZItwin._crop_opd(OPDs)
    else:
        _, opd_cropped = OZItwin._crop_opd(OPDs, input_pupil)

    # Crop de la pupille cible
    pupil_cropped, _ = OZItwin._crop_pupil(output_pupil)

    # Rescale vers la taille de la pupille cible croppée
    opd_rescaled = OZItwin._rescale_matrix(
        opd_cropped,
        pupil_cropped.shape[0],
        pupil_cropped.shape[1]
    )

    # Réinjection dans la matrice complète
    if OPDs.ndim == 3:
        n_frames = OPDs.shape[0]
        OPDs_projected = np.zeros(
            (n_frames, output_shape[0], output_shape[1]),
            dtype=dtype
        )

        OPDs_projected[:, output_pupil] = opd_rescaled[:, pupil_cropped.astype(bool)]

    elif OPDs.ndim == 2:
        OPDs_projected = np.zeros(output_shape, dtype=dtype)
        OPDs_projected[output_pupil] = opd_rescaled[pupil_cropped.astype(bool)]

    else:
        raise ValueError(f"OPDs doit être 2D ou 3D, pas ndim={OPDs.ndim}")

    return OPDs_projected

#%% Compute the OOPAO Objects

Papytwin = Papyrus()
# telescope object
tel     = Papytwin.tel
# source object
ngs     = Papytwin.ngs
# deformable mirror object
dm      = Papytwin.dm
# Pyramid WFS object
wfs     = Papytwin.wfs
# atmosphere object
atm     = Papytwin.atm
# slow Tip/Tilt object
slow_tt = Papytwin.slow_tt
# parameter file
param   = Papytwin.param

Papytwin.set_pupil(calibration=False,sky_offset=[2,2])

# set back to calibration pupil
Papytwin.set_pupil(calibration=True)
ngs*tel*wfs

#%% PAPYRUS input data from the bench
from pymatreader import read_mat
adr ='C:/Users/mmotte/oopao/OOPAO/tutorials/PAPYRUS/'
M2C = np.load('M2C_1rst.npy')

valid_pixel = np.load('validpix_papyrus.npy')

im = np.load('IM_bench_papyrus.npy')

# index of the KL modes included in the int-mat
ind = [1, 5, 10, 20, 30, 50, 80, 100, 150]

int_mat_extract= im[:,ind]

valid_pixel, int_mat_binned = Papytwin.bin_bench_data(valid_pixel = valid_pixel, full_int_mat = im, ratio = param['ratio'])

#%% PAPYRUS/PAPYTWIN Pyramid Pupils Comparison 

var_im = np.var(im,axis=1).reshape(240,240)
var_im/=var_im.max()
var_im = var_im>0.005


# in case there is a mis-match set the key-word "correct" to True
correct = True
Papytwin.check_pwfs_pupils(valid_pixel_map = var_im, correct=correct,n_it=6)

#%% PAPYRUS/PAPYTWIN DM/WFS Mis-registration calibration

# Slow if index_modes is long
index_modes = np.arange(10,150,50)
calibrate_mis_registration = True
if calibrate_mis_registration:
    Papytwin.calibrate_mis_registration(M2C = M2C,
                           input_im = int_mat_binned,
                           index_modes = index_modes)

#%% PAPYRUS/PAPYTWIN Interaction Matrix Comparison 
from plot_functions import plot_psd_aa, plot_sr_aa, plot_psf_aa, plot_frame_count_aa, plot_phase_map_aa, plot_etf_fit_aa,plot_cumulative_psd_aa, plot_phase_comparison_aa, plot_n_psd_aa, plot_curves_aa

M2C_CL      = M2C
wfs.modulation = 5
stroke = 0.0001
calib = InteractionMatrix(  ngs            = ngs,\
                            atm            = atm,\
                            tel            = tel,\
                            dm             = dm,\
                            wfs            = wfs,\
                            M2C            = M2C_CL,\
                            stroke         = stroke,\
                            phaseOffset    = 0,\
                            nMeasurements  = 1,\
                            noise          = 'off',
                            print_time=False,
                            display=True)
    
#%%
difference = np.linalg.pinv(im)@calib.D
difference = np.linalg.pinv(calib.D)@im
diag_obs = np.diag(difference)
modes = np.arange(M2C.shape[-1])
fig, ax = plot_curves_aa(
    x_list=[modes],
    y_list=[diag_obs],
    labels=[
        r"$\mathrm{diag}\!\left(IM_\mathrm{syn,obs}^{+} IM_\mathrm{bench}\right)$"
    ],
    xlabel="Mode index",
    ylabel=r"Diagonal of $IM_\mathrm{synth}^{+} IM$",
    scale="linear",
    figure_width="one_column",
    show_legend=True,
    legend_loc="best",
    curve_styles={
        r"$\mathrm{diag}\!\left(IM_\mathrm{syn,obs}^{+} IM_\mathrm{bench}\right)$": {
            "color": "black",
            "ls": "-",
            "lw": 1.6,
            "marker": "o",
            "ms": 2.8,
            "mew": 0.0,
            "alpha": 1.0,
        },
    },
    save=False,
    savepath="IM_synth_diagonal_comparison.pdf",
    saveformat="all",
)
fig, ax = plot_curves_aa(
    x_list=[modes],
    y_list=[diag_obs],
    labels=[
        r"$\mathrm{diag}\!\left(IM_\mathrm{syn,obs}^{+} IM_\mathrm{bench}\right)$"
    ],
    xlabel="Mode index",
    ylabel=r"Diagonal of $IM_\mathrm{synth}^{+} IM$",
    scale="linear",
    figure_width="one_column",
    show_legend=True,
    legend_loc="best",
    curve_styles={
        r"$\mathrm{diag}\!\left(IM_\mathrm{syn,obs}^{+} IM_\mathrm{bench}\right)$": {
            "color": "black",
            "ls": "-",
            "lw": 1.6,
            "marker": "o",
            "ms": 2.8,
            "mew": 0.0,
            "alpha": 1.0,
        },
    },
    save=False,
    savepath="IM_synth_diagonal_comparison.pdf",
    saveformat="all",
)
#%%
#%% PAPYTWIN Full Interaction Matrix Computation
Papytwin.set_pupil(calibration = True)
wfs.modulation = 5

stroke = 0.0001
calib = InteractionMatrix(  ngs            = ngs,
                            atm            = atm,
                            tel            = tel,
                            dm             = dm,
                            wfs            = wfs,
                            M2C            = M2C,
                            stroke         = stroke,
                            phaseOffset    = 0,\
                            nMeasurements  = 1,\
                            noise          = 'off',
                            print_time     = False,
                            display        = True)
    

#%%  -----------------------     Close loop  ----------------------------------
tel.resetOPD()

end_mode    = 195 
M2C_CL = M2C[:,:end_mode]
# These are the calibration data used to close the loop
# use of experimental calibration
# if full int-mat is available only
calib_CL    = CalibrationVault(im[:,:end_mode])

    
    
#%%
Papytwin.set_pupil(calibration = False, sky_offset=[2,2]) 
    
# =============================================================================
# Second Stage
# =============================================================================
nmodes = 35
OZItwin = OZIRIIS()

IM_z1, _ = OZItwin.compute_synth_IM()
calib_CL_2nd    = CalibrationVault(IM_z1[:,:nmodes]).M



#%% Second stage atmosphere
OZItwin.atm.initializeAtmosphere(OZItwin.tel, compute_covariance=True)

OZItwin.atm*OZItwin.tel
nLoop = 1000
atm_OPDs_2nd = np.zeros((nLoop, OZItwin.atm.OPD.shape[0],OZItwin.atm.OPD.shape[1]))
for i in tqdm.tqdm(range(nLoop)):
    OZItwin.atm.generateNewPhaseScreen(seed = np.random.randint(0,100000))
    OZItwin.src*OZItwin.atm
    OZItwin.atm.update()
    atm_OPDs_2nd[i] = OZItwin.atm.OPD.copy()
#%%
atm_OPDs_1rst = OZItwin._rescale_matrix(atm_OPDs_2nd, tel.pupil.shape[0],tel.pupil.shape[0] )
#%% first stage atm

atm_OPDs_1rst = project_opd_between_pupils(
    OZItwin,
    OPDs=atm_OPDs_2nd,
    input_pupil=None,
    output_pupil=tel.pupil,
    output_shape=tel.pupil.shape
)
#%%
from OOPAO.GainSensingCamera import GainSensingCamera
pupil = tel.pupil.ravel()==1

tel.resetOPD()
dm.coefs = 0
dm.coefs = M2C.copy()
tel * dm

modes = tel.OPD.copy().reshape(tel.resolution**2, M2C.shape[-1])

# Important : on restreint vraiment à la pupille
modes_pupil = modes.copy()

# Retrait piston spatial des modes, optionnel mais recommandé
modes_pupil -= modes_pupil.mean(axis=0, keepdims=True)

# Normalisation sur la pupille
modes_pupil /= modes_pupil[pupil, :].std(axis=0, keepdims=True)

# Gram complet, pas seulement la diagonale
cov_modes = modes_pupil.T @ modes_pupil
diag = np.diag(cov_modes)
diag = np.where(np.abs(diag) < 1e-30, 1.0, diag)

proj_1rst =(np.diag(1.0 / diag) @ modes_pupil.T).astype(np.float32) # (np.diag(1.0 / diag) @ modes_pupil.T).astype(np.float32) #@ modes_pupil.T).astype(np.float32)
#%%
tel.resetOPD()
dm.coefs = 0
ngs*tel*dm*wfs

gsc = GainSensingCamera(wfs.mask, modes_pupil.reshape(80,80,-1))
nit = 10
og_ol = np.zeros((nit, modes.shape[-1]))
wfs.focal_plane_camera.resolution = wfs.nRes

tel * wfs
wfs * wfs.focal_plane_camera
wfs.focal_plane_camera * gsc
atm.initializeAtmosphere(tel)
for i in tqdm.tqdm(range(nit)):

    atm.update( atm_OPDs_1rst[i])
    ngs*atm*tel*wfs
    wfs * wfs.focal_plane_camera
    wfs.focal_plane_camera * gsc
    og_ol[i] = gsc.og.copy()

print(og_ol.mean())
#%%

atm.initializeAtmosphere(tel)

from OOPAO.Source import Source

#create scientific source
src =   Source('IR1310', 0)
src*tel

# initialize Telescope DM commands
tel.resetOPD()
dm.coefs=0
ngs*tel*dm*wfs
wfs*wfs.focal_plane_camera

# initialize DM commands
atm*ngs*tel
atm*src*tel

ratio_samp = 1/OZItwin.tel.samplingTime/(1/tel.samplingTime)

# allocate memory to save data
SR_NGS                      = np.zeros(nLoop)
SR_SRC                      = np.zeros(nLoop)
total                       = np.zeros(nLoop)
residual_SRC                = np.zeros(nLoop)
residual_NGS                = np.zeros(nLoop)
dm_commands = np.zeros((nLoop, dm.nValidAct))

# loop parameters
gainCL                  = 0
leak = 0.995
wfs.cam.photonNoise     = False
display                 = False
frame_delay             =0 
wfsSignal               = np.zeros((int(nLoop/ratio_samp)+frame_delay,wfs.nSignal))

reconstructor = M2C_CL@calib.M
reconstructed_cmd = np.zeros((int(nLoop/ratio_samp),241))
opds = np.zeros((nLoop,tel.pupil.shape[0],tel.pupil.shape[1]))
opds_res = np.zeros((nLoop,tel.pupil.shape[0],tel.pupil.shape[1]))
k = 0

for i in range(nLoop):

    
    
        
            # update phase screens => overwrite tel.OPD and consequently tel.src.phase  
    atm.update(atm_OPDs_1rst[i]*tel.pupil)
        # save phase variance
    total[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    # propagate light from the NGS through the atmosphere, telescope, DM to the WFS and NGS camera with the CL commands applied
    
    atm*ngs*tel*dm*wfs
    opds[i]= dm.OPD.copy()
    opds_res[i]= tel.OPD.copy()
    wfs*wfs.focal_plane_camera
    # save residuals corresponding to the NGS
    residual_NGS[i] = np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    if i%ratio_samp==0:
        
        wfsSignal[k+frame_delay]=wfs.signal.copy()
        reconstructed_cmd[k] = np.matmul(reconstructor,wfsSignal[k])
        # apply the commands on the DM
        dm.coefs=dm.coefs-gainCL*reconstructed_cmd[k]
        k+=1
    
        
        # propagate light from the SRC through the atmosphere, telescope, DM to the Instrument camera
    atm*src*tel*dm
    
    dm_commands[i,:] = dm.coefs.copy()
    # save residuals corresponding to the NGS
    residual_SRC[i] = np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    OPD_SRC         = tel.mean_removed_OPD.copy()
    
        # store the slopes after computing the commands => 2 frames delay
       
        
    print(
    f'\rLoop {i+1}/{nLoop} '
    f'NGS: {residual_NGS[i]:.3f} '
    f'-- SRC: {residual_SRC[i]:.3f}',
    end='',
    flush=True
    )

      

#%%m
residuals_opds_1rst = project_opd_between_pupils(
    OZItwin,
    opds_res,
    input_pupil=tel.pupil,
    output_pupil=OZItwin.tel.pupil
)
#%%
# allocate memory to save data
SR_NGS_2nd                      = np.zeros(nLoop)
SR_SRC_2nd                        = np.zeros(nLoop)
total_2nd                         = np.zeros(nLoop)
residual_SRC_2nd                  = np.zeros(nLoop)
residual_NGS_2nd                  = np.zeros(nLoop)
dm_commands_2nd   = np.zeros((nLoop, OZItwin.dm.nValidAct))
wfsSignal_2nd                 = np.arange(0,OZItwin.vzwfs.zwfs1.nSignal)*0

# loop parameters
gainCL_2nd                    = 0
leak_2nd = 0.98
frame_delay_2nd               = 2
reconstructor_2nd   =OZItwin.M2C[:,:nmodes]@calib_CL_2nd
reconstructed_cmd_2nd = np.zeros((nLoop,97))

OZItwin.dm.coefs = 0
OZItwin.atm.initializeAtmosphere(OZItwin.tel)
opds_2nd = np.zeros_like(residuals_opds_1rst)
for i in range(nLoop):
    
    a=time.time()
    
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase  
    OZItwin.atm.update(residuals_opds_1rst[i])
    # save phase variance
    total_2nd[i]=np.std(OZItwin.tel.OPD[np.where(OZItwin.tel.pupil>0)])*1e9
    # propagate light from the NGS through the atmosphere, telescope, DM to the WFS and NGS camera with the CL commands applied

    OZItwin.atm*OZItwin.src*OZItwin.tel*OZItwin.dm
    OZItwin.tel*OZItwin.vzwfs

    # save residuals corresponding to the NGS
    residual_NGS_2nd[i] = np.std(OZItwin.tel.OPD[np.where(OZItwin.tel.pupil>0)])*1e9
    OPD_NGS         = OZItwin.tel.mean_removed_OPD.copy()
    opds_2nd[i]= OZItwin.tel.OPD.copy()
    
    
    # propagate light from the SRC through the atmosphere, telescope, DM to the Instrument camera
    OZItwin.atm*OZItwin.src*OZItwin.tel*OZItwin.dm
    dm_commands_2nd[i,:] = OZItwin.dm.coefs.copy()
    # save residuals corresponding to the NGS
    residual_SRC_2nd[i] = np.std(OZItwin.tel.OPD[np.where(OZItwin.tel.pupil>0)])*1e9
    OPD_SRC_2nd         = OZItwin.tel.mean_removed_OPD.copy()
    if frame_delay_2nd ==1:        
        wfsSignal_2nd=OZItwin.vzwfs.zwfs1.signal.copy()
    reconstructed_cmd_2nd[i]=np.matmul(reconstructor_2nd,wfsSignal_2nd)
    # apply the commands on the DM
    OZItwin.dm.coefs=leak_2nd*OZItwin.dm.coefs-gainCL_2nd*reconstructed_cmd_2nd[i]
    
    # store the slopes after computing the commands => 2 frames delay
    if frame_delay_2nd ==2:        
        wfsSignal_2nd=OZItwin.vzwfs.zwfs1.signal.copy()
    
    
    print(
    f'\rLoop {i+1}/{nLoop} '
    f'NGS: {residual_NGS_2nd[i]:.3f} '
    f'-- SRC: {residual_SRC_2nd[i]:.3f}',
    end='',
    flush=True
    )
#%%


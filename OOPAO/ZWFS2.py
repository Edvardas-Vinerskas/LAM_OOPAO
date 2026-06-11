# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:42:34 2024

@author: mmotte
"""

import inspect
import multiprocessing
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sp
import warnings
from OOPAO.Detector import Detector


import numpy as np
import matplotlib.pyplot as plt
from OOPAO.ZWFS import ZWFS
class ZWFS2:
    def __init__(self, tel = None, diameter:float = None, phase_shift = [-np.pi/2,np.pi/2], flux_ratio:float = 1/2, transmittance:complex = 1, zpf = 4, phase_shift_unit = 'radian', propagation_method = 'MFT', ZWFS1 = None, ZWFS2=None):

        
        if tel is not None:
            self.telescope = tel
        if tel is None and (ZWFS1 is None or ZWFS2 is None):
            raise ValueError('If at least one ZWFS is not entered manually, a telescope object needs to be input in order to create the ZWFS class')
        if isinstance(phase_shift, float):
            phase_shift = [phase_shift-np.pi,phase_shift]
        self.phase_shift = np.array(phase_shift)
        self.phase_shift_unit = phase_shift_unit
        if ZWFS1 is None:
            
            self.zwfs1 = ZWFS( tel=self.telescope, diameter=diameter, phase_shift = phase_shift[0], transmittance=flux_ratio*transmittance, zpf = zpf, phase_shift_unit=phase_shift_unit, propagation_method=propagation_method)
        else:
            self.zwfs1 = ZWFS1
        if ZWFS2 is None:
            self.zwfs2 = ZWFS( tel=self.telescope, diameter=diameter, phase_shift = phase_shift[1], transmittance=(1-flux_ratio)*transmittance, zpf = zpf, phase_shift_unit=phase_shift_unit, propagation_method=propagation_method)
        else:
            self.zwfs2 = ZWFS2
        
        
        # self.zwfs1.beta_ref = self.zwfs1.beta
        # self.zwfs1.Ib_ref = self.zwfs1.Ib
        # self.zwfs2.beta_ref = self.zwfs2.beta
        # self.zwfs2.Ib_ref = self.zwfs2.Ib
        self.zpf = zpf
        self.wfs_measure()
        self._ref_signal = np.append(self.zwfs1.ref_signal, self.zwfs2.ref_signal)
        self._signal = np.append(self.zwfs1.signal, self.zwfs2.signal)
        self.pupil_mask = np.concatenate((self.zwfs1.pupil_mask ,self.zwfs2.pupil_mask), axis = 1)
        self._img_ref = np.concatenate((self.zwfs1.img_ref ,self.zwfs2.img_ref), axis = 1)
        self._img_ZWFS = np.concatenate((self.zwfs1.img_ZWFS,self.zwfs2.img_ZWFS), axis = 1)
        self.tag = 'ZWFS'
    def wfs_measure(self, phase_in = None, reconstructor = None, reconstructor_iteration = 1, known_EM = False):
        if reconstructor_iteration<1:
            warnings.warn('Number of iteration ofr the reconstructor should be superior or equal to 1, Taking 1')
            reconstructor_iteration = 1
        # self.b = self.telescope.src.phas
        self.zwfs1.wfs_measure(phase_in, known_EM=known_EM)
        self.zwfs2.wfs_measure(phase_in, known_EM=known_EM)
        if reconstructor is not None:
            self.retrieved_phase = self.reconstructor(reconstructor, iteration= reconstructor_iteration)
        # self.img_raw = np.concatenate((self.zwfs1.img_raw,self.zwfs2.img_raw), axis = 1)#(-self.zwfs1.img_raw+self.zwfs2.img_raw)/(2*self.telescope.pupil**2+1-(self.zwfs1.img_raw+self.zwfs2.img_raw))
        # # self.img_raw[self.telescope.pupil==0]=0
        # self.signal = self.img_raw.reshape(self.img_raw.size)
        self.nSignal = self.signal.size
       
    def reinitialise_class(self):
        self.zwfs1.reset_Ib_beta()
        self.zwfs2.reset_Ib_beta()
        self.wfs_measure(phase_in=np.zeros(self.telescope.pupil.shape))
        
    def _filter_modes(self, projector, phase):
        projected_phase=projector@phase.ravel()
        return projected_phase
    
    def _compute_proj(self, modal_basis):
        pupil = self.zwfs1.telescope.pupil == 1
        modal_basis = modal_basis.reshape((-1, modal_basis.shape[-1]))/modal_basis[pupil, :].std(axis=0)  # Flatten for projection
        cov_modes = modal_basis.T @ modal_basis  # Compute mode covariance
        return np.diag(1 / np.diag(cov_modes)) @ modal_basis.T  # Pseudo-inverse projection matrix
    def reconstructor(self, reconstructor = 'atan', iteration = 1, damping_iteration = 1, filter_modes = False, modal_basis = None, print_error = False):
        clip = False
        self.zwfs1.reset_Ib_beta()
        self.zwfs2.reset_Ib_beta()
        self.zwfs1.sin_phase = self.zwfs1.phase_sin(clipping=clip)
        self.zwfs2.sin_phase = self.zwfs2.phase_sin(clipping=clip)
        phase1 = np.zeros(np.append(iteration+1,self.zwfs1.telescope.pupil.shape))
        phase2 = np.zeros(np.append(iteration+1,self.zwfs1.telescope.pupil.shape))
        tol = 1e-5
        if filter_modes:
            if modal_basis is None:
                raise AttributeError('Modal basis must not be empty when filter modes is None')
            if modal_basis[...,0].shape !=self.zwfs1.telescope.pupil.shape:
                raise ValueError('Axes (0,1) of modal basis should be the shape and axis 3 should be the modes to project the phase on')
            self.modal_basis = modal_basis/modal_basis[self.zwfs1.telescope.pupil==1, :].std(axis = 0)
            projector = self._compute_proj(self.modal_basis)
        convergence = []
        for i in range(iteration):
            self.sin_phase = np.concatenate((self.zwfs1.sin_phase,self.zwfs2.sin_phase), axis = 0)
            M11 = -np.sin(self.zwfs1.beta+self.zwfs1.phase_shift/2)
            M11[np.isnan(M11)]=0
            M12 =  np.cos(self.zwfs1.beta+self.zwfs1.phase_shift/2)
            M12[np.isnan(M12)]=0
            M21 = -np.sin(self.zwfs2.beta+self.zwfs2.phase_shift/2)
            M21[np.isnan(M21)]=0
            M22 = np.cos(self.zwfs2.beta+self.zwfs2.phase_shift/2)
            M22[np.isnan(M22)]=0
            self.det_M = M11*M22-M12*M21
            eps = 1e-5
            bad = np.abs(self.det_M) < eps
            
            if (bad.any() & (self.det_M[(self.zwfs2.telescope.pupil==1)&(self.zwfs1.telescope.pupil==1)]==0)).sum()>0:
                warnings.warn('Some pixels has a null determinant')
            det = self.det_M.copy()
            

            self.cos_phi = (M22*self.zwfs1.sin_phase-self.zwfs2.sin_phase*M12)/det
            self.sin_phi = (-M21*self.zwfs1.sin_phase+self.zwfs2.sin_phase*M11)/det
            if reconstructor == 'linear':
                retrieved_phase = self.sin_phi/self.cos_phi
                
            if reconstructor == 'atan':
                retrieved_phase = np.arctan2(self.sin_phi,self.cos_phi)
                # retrieved_phase[(self.zwfs2.telescope.pupil==1)&(self.zwfs1.telescope.pupil==1)]-=retrieved_phase[(self.zwfs2.telescope.pupil==1)&(self.zwfs1.telescope.pupil==1)].mean()
                # retrieved_phase[(self.zwfs2.telescope.pupil==0)|(self.zwfs1.telescope.pupil==0)]=0
                if filter_modes:
                    projected_phase = self._filter_modes(projector, retrieved_phase)
                    retrieved_phase = (self.modal_basis*projected_phase[None,None, :]).sum(axis=-1)
            if iteration>1:
          
                # retrieved_phase[np.abs(retrieved_phase)>2*np.pi] = 0
                
                phase1[i+1, self.zwfs1.telescope.pupil==1] =(1-damping_iteration)*phase1[i,self.zwfs1.telescope.pupil==1] +damping_iteration*retrieved_phase[self.zwfs1.pupil_mask]
                # phase1[np.where(~np.isfinite(phase1))]=0
                phase1[i+1, self.zwfs1.telescope.pupil==1]-=phase1[i+1, self.zwfs1.telescope.pupil==1].mean()
                phase1[i+1, self.zwfs1.telescope.pupil==0] = 0
                phase2[i+1, self.zwfs2.telescope.pupil==1] = (1-damping_iteration)*phase2[i,self.zwfs2.telescope.pupil==1] +damping_iteration*retrieved_phase[self.zwfs2.pupil_mask]
                phase2[i+1, self.zwfs2.telescope.pupil==1]-=phase2[i+1, self.zwfs2.telescope.pupil==1].mean()
                phase2[i+1, self.zwfs2.telescope.pupil==0] = 0
                err1 = np.linalg.norm(phase1[i+1, self.zwfs1.telescope.pupil==1] - phase1[i,self.zwfs1.telescope.pupil==1]) / np.sqrt(np.sum(self.zwfs1.telescope.pupil==1))
                err2 = np.linalg.norm(phase2[i+1, self.zwfs2.telescope.pupil==1] - phase2[i, self.zwfs2.telescope.pupil==1]) / np.sqrt(np.sum(self.zwfs2.telescope.pupil==1))
    
                err = max(err1, err2)
                convergence.append(err)
                if print_error:
                    if err < tol:
                        print(
                            f"\rConvergence atteinte à l'itération {i}, erreur = {err:.3e}".ljust(100),
                            file=sys.stderr,
                            flush=True,
                        )
                        break
                    else:
                        print(
                            f"\rErreur interne : {err:.3e}".ljust(100),
                            end="",
                            file=sys.stderr,
                            flush=True,
                        )
                                # phase2[np.where(~np.isfinite(phase2))]=0
                Psi_b,self.zwfs1.Ib = self.zwfs1.propagation(mask = self.zwfs1.amplitude_mask, phase=phase1[i+1], computing_Ib=True)
                self.zwfs1.beta = np.angle(Psi_b)
                self.zwfs1.sin_phase = self.zwfs1.phase_sin(clipping=clip)
                Psi_b,self.zwfs2.Ib = self.zwfs2.propagation(mask = self.zwfs2.amplitude_mask, phase=phase2[i+1], computing_Ib=True)
                self.zwfs2.beta = np.angle(Psi_b)
                self.zwfs2.sin_phase = self.zwfs2.phase_sin(clipping=clip)
        retrieved_phase[(self.zwfs2.telescope.pupil==1)&(self.zwfs1.telescope.pupil==1)]-=retrieved_phase[(self.zwfs2.telescope.pupil==1)&(self.zwfs1.telescope.pupil==1)].mean()
        retrieved_phase[(self.zwfs2.telescope.pupil==0)|(self.zwfs1.telescope.pupil==0)]=0
        return retrieved_phase
    def _unconcatenate(self,vzimg):
        z1 = vzimg[...,:,:vzimg.shape[-1]//2]
        z2 = vzimg[...,:,vzimg.shape[-1]//2:]
        return z1, z2
    @property
    def img_ZWFS(self):
        self._img_ZWFS = np.concatenate((self.zwfs1.img_ZWFS,self.zwfs2.img_ZWFS), axis = 1)
        return self._img_ZWFS
    @img_ZWFS.setter
    def img_ZWFS(self,val):
        self.zwfs1.img_ZWFS, self.zwfs2.img_ZWFS = self._unconcatenate(val)
        self._img_ZWFS = val
    @property
    def ref_signal(self):
        return np.append(self.zwfs1.ref_signal, self.zwfs2.ref_signal)
    @ref_signal.setter
    def ref_signal(self,val):
        self._img_ref = np.zeros(self.pupil_mask.shape)
        self._img_ref[self.pupil_mask] = val
        self.zwfs1.img_ref, self.zwfs2.img_ref = self._unconcatenate(self._img_ref)
    @property
    def signal(self):
        self._signal = np.append(self.zwfs1.signal, self.zwfs2.signal)
        return self._signal
    # @signal.setter
    # def signal(self,val):
    #     self._signal = val
    @property
    def img_ref(self):
        self._img_ref = np.concatenate((self.zwfs1.img_ref ,self.zwfs2.img_ref), axis = 1)
        return self._img_ref
    @img_ref.setter
    def img_ref(self, val):
        img1, img2 = self._unconcatenate(val)
        self.zwfs1.img_ref = img1   
        self.zwfs2.img_ref = img2
        self._img_ref = val
        self._ref_signal = np.append(self.zwfs1.ref_signal, self.zwfs2.ref_signal)
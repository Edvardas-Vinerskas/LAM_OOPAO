# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:33:27 2024

@author: mmotte
"""

import warnings

import numpy as np

from numpy.fft import fft2,ifft2,fftshift
from .Detector import Detector

import logging  # For logging messages


logging.basicConfig(level=logging.INFO)  # Set up logging
logger = logging.getLogger(__name__)  # Create logger

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

        
#%%

class ZWFS:
    def __init__(
        self,
        tel,
        diameter: float = 1.06,
        phase_shift: float = np.pi/2,
        transmittance: float = 1.0,
        zpf: int = 4,
        phase_shift_unit: str = "radian",
        propagation_method: str = "MFT",
        ):
        """ ZWFS
        The ZWFS consist in a 2D mask at the focal plane of the telescope to perform the Fourier Filtering of the EM-Field. 
        By default the ZWFS detector is considered to be noise-free (for calibration purposes). For now, the detector parametres has not yet been implemented. 
        
        Parameters
        ----------
        tel: TYPE
            The telescope object to which the ZWFS is associated. This object carries the phase, flux and pupil information.
        diameter: float 
            The diameter of the ZWFS mask in lambda/D: default value 1.06
        phase_shift: float
            The phase shift induce by the ZWFS in radian, lambda or pi depending on chosen phase shift unit: default value pi/2
        transmittance: float
            The portion of flux going into the mask 
        zpf: int 
            In FFT propagation method, it is the Zero padding factor. In MFT propagation method it correspond to the diameter of the mask in fourier plane (ie the resolution of the fourier transform inside the mask): default value 4.
        phase_shift_unit: string
            If one want to express the depth in lambda, pi, or radian. default : radian
        propagation_method: string
            Select the fourier transform propagation method: default MFT, possibility with FFT but MFT recommanded. 
            
        ************************** PROPAGATING THE LIGHT TO THE ZWFS OBJECT **************************
        The light can be propagated from a telescope object tel through the ZWFS object wfs using the * operator:
        _ tel*wfs
        This operation will trigger:
            _ propagation of the tel.src light through the PWFS detector (phase and flux)
            _ computation of the ZWFS signals

        If the tel.src object is an asterism of sources with the same wavelength, each source is propagated to the ZWFS.
        The resulting intensities are summed incoherently before being integrated by the ZWFS camera.


        """
        if propagation_method != 'FFT' and propagation_method != 'MFT':
            warnings.warn('Unknown propagation method, using FFT')
            self.propagation_method = 'FFT'
        else:
            self.propagation_method = propagation_method
        
        self.telescope = tel
        self.tag = 'ZWFS' 
        # if nSubap is None:
        #     self.nSubap = self.telescope.resolution
        # else:
        #     self.nSubap = nSubap
        self.pupil_mask = (self.telescope.pupil != 0)
        self.nSignal = np.sum(self.pupil_mask).astype(np.int64)#self.telescope.pupil.size
        if propagation_method == 'FFT':
            if zpf <=2:
                warnings.warn('Zero-padding factor should be superior to 2 for an FFT propagation method')
            else:
                self.zpf = zpf
                self.resolution = np.int64(self.zpf*self.telescope.resolution)
        elif propagation_method == 'MFT':
            if zpf <=4:
                warnings.warn('Number of pixel inside the mask should be superior to 4 for a MFT propagation method')
            else:
                self.resolution = zpf
                self.zpf = zpf
        
        
        self.transmittance = transmittance
        self.diameter = diameter 
        if diameter <=0:
            warnings.warn(f'{diameter} is not a possible diameter, using d=1.06 instead')
            self.diameter = 1.06
     
                 
        self.radius = self.diameter/2
        if phase_shift_unit == 'lambda':
            self.phase_shift = phase_shift * 2 * np.pi
        elif phase_shift_unit == 'radian':
            self.phase_shift = float(phase_shift)
        elif phase_shift_unit == 'pi':
            self.phase_shift = float(phase_shift) * np.pi
        else:
            raise ValueError("phase_shift_unit must be 'lambda', 'radian', or 'pi'")

        self.epsilon = (1-np.exp(1j*self.phase_shift))
        
        self.pixel_size = self.telescope.D/self.resolution
        self.Z_bool =self.ZWFS_bool()
        self.Zmask = self.ZWFS_mask()
        
        self.amplitude_mask = self.Z_bool*np.sqrt(self.transmittance)
        # _, self.Ib = self.propagation(mask = self.amplitude_mask, phase = 0)
        self.Ip = self.telescope.pupilReflectivity**2*self.transmittance
        self._img_ZWFS = 0
        self._signal = 0
        self.wfs_measure(phase_in=np.zeros(self.telescope.pupil.shape))
        
        self._ref_signal = self.signal
        self._img_ref = self.img_ZWFS
    def ZWFS_bool(self):
        if self.propagation_method == 'MFT':
            Zradius_pixels = self.resolution/2
            center = (self.resolution/2-0.5,self.resolution/2-0.5)
        else:
            Zradius_pixels = self.zpf*self.radius
            center = (self.resolution/2,self.resolution/2)
        
        Y, X = np.ogrid[:self.resolution, :self.resolution]
        Z_bool = (X - center[0])**2 + (Y - center[1])**2 <= Zradius_pixels**2
        return Z_bool
    def ZWFS_mask(self, phase = None):
        if phase is None:
            phase = self.phase_shift
        Zmask =np.zeros((self.resolution,self.resolution), dtype=np.complex128)*np.sqrt(self.transmittance)
        Zmask= (1-(1-np.exp(1j*phase))*self.Z_bool)*np.sqrt(self.transmittance)
        
        return Zmask 
    def propagation(self, mask = None, phase = None, epsilon = None, computing_Ib = False) : 
        if phase is None:
            phase = self.telescope.src.phase
        
        if self.propagation_method == 'MFT':
            # a = time.time()
            Na = self.telescope.resolution
            Nb = self.resolution
            m = self.diameter
            EM = self.telescope.pupilReflectivity*np.exp(1j*phase)
            if epsilon is None: 
                epsilon = self.epsilon
            if computing_Ib:
                _ , EM_det = self._MFT(Na, Nb, m, EM, mask = self.Z_bool)
                EM_det*=np.sqrt(self.transmittance)
            else:
                _ ,EM_det = self._MFT(Na, Nb, m, EM, mask = self.Z_bool)
                EM_det=(EM-(1-np.exp(1j*self.phase_shift))*EM_det)*np.sqrt(self.transmittance)
            # print(np.sum(np.abs(EM)**2))
            EM_det[~self.pupil_mask]=0
            IM_detect = np.abs(EM_det)**2
            # b = time.time()
            # print(b-a)
        else:
            if mask is None:
                mask = self.Zmask
            # a = time.time()
            if self.propagation_method != 'FFT':
                warnings.warn(f'{self.propagation_method} is not a possible propagation method, using FFT instead')
            EM = np.zeros((self.resolution,self.resolution), np.complex128)
            
            EM[self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2,self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2]\
                =self.telescope.pupilReflectivity*np.exp(1j*phase)
            Psi_turb = fftshift(fft2(fftshift(EM))) # 1er prop
            Psi_FP_turb = mask*Psi_turb # PF multiplication par masque
            # print(np.sum(np.abs(Psi_FP_turb)**2))
            EM_det = fftshift(ifft2(fftshift((Psi_FP_turb))))[self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2,self.resolution//2-self.telescope.resolution//2:self.resolution//2+self.telescope.resolution//2] # PP detecteur
            EM_det[~self.pupil_mask]=0
            IM_detect=np.abs(EM_det)**2
            # b = time.time()
            # print(b-a)
        # print(np.sum(IM_detect))
        return EM_det,IM_detect
    def wfs_measure(self, phase_in = None, reconstruction_method = None, known_EM = False, reconstructor_iteration:int = 1, estimated_phase = 0):
        if reconstructor_iteration<1:
            warnings.warn('Number of iteration ofr the reconstructor should be superior or equal to 1, Taking 1')
            reconstructor_iteration = 1
        if phase_in is not None:
            self.telescope.src.phase = phase_in
        self.EM_detect, self.img_ZWFS = self.propagation()
        if known_EM:
            Psi_b,self.Ib = self.propagation(mask = self.amplitude_mask, computing_Ib=True)
            self.beta = np.angle(Psi_b)
            self.Ip =self.telescope.pupilReflectivity**2*self.transmittance #self.propagation(mask = np.sqrt(self.transmittance),phase = 0)
        else:
            Psi_b,self.Ib = self.propagation(mask = self.amplitude_mask, phase = estimated_phase, computing_Ib=True)
            self.beta = np.angle(Psi_b)
        self.sin_phase = self.phase_sin()
        # if np.size(np.where(np.abs(self.sin_phase)==1)[1])>0:
        #     self.sin_phase/=np.nanmax(np.abs(self.sin_phase))
        if reconstruction_method is not None:
            self.retrieved_phase = self.reconstructor(reconstruction_method, iteration= reconstructor_iteration)
    
        # self.signal =  np.reshape(self.img_ZWFS, (self.nSignal,))
    def phase_sin(self, Ib = None, Ip = None, img_ZWFS = None, phase_shift = None, clipping = True):
        if Ib is None:
            Ib = self.Ib
        if Ip is None:
            Ip = self.Ip
        if img_ZWFS is None:
            img_ZWFS = self.img_ZWFS.copy()
        if phase_shift is None:
            phase_shift = self.phase_shift
        # sin_phase = (img_ZWFS-Ip-4*Ib*(np.sin(phase_shift/2))**2)/(4*np.sqrt(Ib*Ip)*np.sin(phase_shift/2))
        eps = 1e-12
        denominator = 4*np.sqrt(np.maximum(Ib, eps)*np.maximum(Ip, eps))*np.sin(phase_shift/2)
        denominator = np.where(np.abs(denominator) < eps, np.sign(denominator)*eps + (denominator==0)*eps, denominator)
        sin_phase = (img_ZWFS - Ip - 4*Ib*(np.sin(phase_shift/2))**2) / denominator
        if clipping:
            sin_phase = np.clip(sin_phase, -1,1)
        sin_phase[~self.pupil_mask] = 0.0
        # sin_phase[self.telescope.pupil==0]=0
        return sin_phase
    def reinitialise_class(self):
        self.reset_Ib_beta()
        self.wfs_measure(phase_in=np.zeros(self.telescope.pupil.shape))
    def reset_Ib_beta(self):
        Psi_b,self.Ib = self.propagation(mask = self.amplitude_mask, phase = 0, computing_Ib=True)
        self.beta = np.angle(Psi_b)
    def reconstructor(self, reconstr = 'linear', iteration = 1, clipping = True):
        self.reset_Ib_beta()

        self.sin_phase = self.phase_sin()
        try:
            
            for i in range(iteration):
                if reconstr == 'linear':
                    phase = 1/np.cos(self.phase_shift/2+self.beta)*(np.sin(self.phase_shift/2+self.beta)+self.sin_phase)
                    phase[self.telescope.pupil==0]=0
                if reconstr == 'asin':
                    phase = self.phase_shift/2+self.beta+np.arcsin(self.sin_phase)
                    phase[self.pupil_mask]-=phase[self.pupil_mask].mean()
                    phase[~self.pupil_mask]=0
                    # phase[self.pupil_mask]-=np.nanmean(phase[self.pupil_mask])
                if iteration>1:
                    phase[np.abs(phase)>2*np.pi] = phase[np.abs(phase)>2*np.pi]/phase[np.abs(phase)>2*np.pi]*np.sign(phase[np.abs(phase)>2*np.pi])
                        # phase[np.where(np.isnan(phase))]= self.phase_shift/2+self.beta[np.where(np.isnan(phase))]+np.arcsin(np.mod(self.sin_phase[np.where(np.isnan(phase))], 1))
                    # phase[~np.isfinite(phase)]=0# np.sqrt(np.pi)*(1/np.cos(self.phase_shift/2+self.beta[np.where(np.isnan(phase))])*(np.sin(self.phase_shift/2+self.beta[np.where(np.isnan(phase))])+self.sin_phase[np.where(np.isnan(phase))])) #np.nanstd(phase[self.telescope.pupil!=0][~np.isnan(phase[self.telescope.pupil!=0])])
                    
                    # phase[phase>clipping] = clipping#np.sign(phase[np.abs(phase)>clipping])*1
                    # if np.sum(np.abs(phase)>clipping)>0:
                    #     phase = np.clip(phase, -clipping, clipping)
                    #     logger.warning('Clipping the phase')
                    Psi_b,self.Ib = self.propagation(mask = self.amplitude_mask, phase=phase, computing_Ib=True)
                    self.beta = np.angle(Psi_b)
                    # self.Ip =self.telescope.pupilReflectivity**2*self.transmittance #self.propagation(mask = np.sqrt(self.transmittance),phase = 0)
                    self.sin_phase = self.phase_sin(clipping = clipping)
            return phase
            
        except Exception as e:
            logger.warning("Reconstruction failed: %s", e, exc_info=True)
            return None
    
    def _MFT(self, Na:int, Nb:int, m, EMF_pupil, mask = 1):
        Nb = np.int64(Nb)
        Na = np.int64(Na)
        x = np.zeros((Na, 1))
        u = np.zeros((Nb, 1))
        x[:,0] = (np.arange(0, Na)-(Na)/2+0.5)*1/Na
        u[:,0] = (np.arange(0, Nb)-(Nb)/2+0.5)*m/Nb
        Ft = (m/(Na*Nb)*np.exp(-2*1j*np.pi*(u@x.T))@EMF_pupil@np.exp(-2*1j*np.pi*(x@u.T)))
        iFt = m/(Na*Nb)*np.exp(2*1j*np.pi*(x@u.T))@(Ft*mask)@np.exp(2*1j*np.pi*(u@x.T))
        return Ft, iFt
    @property
    def img_ZWFS(self):
        return self._img_ZWFS
    @img_ZWFS.setter
    def img_ZWFS(self, val):
        try:
            self._signal = val[self.pupil_mask]-self.ref_signal
        except:
            self._signal = val[self.pupil_mask]
        self._img_ZWFS = val
    @property
    def signal(self):
        return self._signal
    @signal.setter
    def signal(self,val):
        self._img_ZWFS = np.zeros(self._img_ZWFS.shape)
        self._img_ZWFS[self.pupil_mask] = val
        self._signal = val
    @property
    def img_ref(self):
        return self._img_ref
    @img_ref.setter
    def img_ref(self, val):
        self._ref_signal = val[self.pupil_mask]
        self._img_ref = val
    @property
    def ref_signal(self):
        return self._ref_signal
    @ref_signal.setter
    def ref_signal(self,val):
        self._img_ref = np.zeros(self._img_ref.shape)
        self._img_ref[self.pupil_mask] = val
        self._ref_signal = val


    def relay(self, src):
        if src.tag == 'source':
            src_list = [src]
        elif src.tag == 'asterism':
            src_list = src.src
        signal_list = []


        for src in src_list:
            src.optical_path.append([self.tag, self])
            self.src = src
            self.wfs_measure(phase_in=self.src.phase)
            signal_list.append(self.signal)

        self.signal = np.squeeze(np.array(signal_list))

        return





        
    def __mul__(self,obj): 
        if obj.tag=='detector':
            obj._integrated_time+=self.telescope.samplingTime
            obj.integrate(obj._integrated_time)

        else:
            print('Error light propagated to the wrong type of object')
        return -1

#%%

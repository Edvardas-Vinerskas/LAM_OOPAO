import sys, os
sys.path.insert(0, r'c:\Users\evinerskas\PycharmProjects\LAM_OOPAO')
print("CWD:", os.getcwd())

from OOPAO.Telescope import Telescope
from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Source import Source
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.MisRegistration import MisRegistration
import numpy as np
import pylab as plt
import torch



ngs = Source(optBand='H', magnitude=0, display_properties=False)
tel = Telescope(resolution=90, diameter=1.52)
ngs ** tel
atm = Atmosphere(telescope= tel, r0 = 0.1, L0 = 25, windSpeed=[5], fractionalR0=[1], windDirection=[0], altitude=[0], src=ngs)

param = np.load('Examples/dm_second_stage_misreg_dict.npy', allow_pickle=True).item()
print(param)
m = MisRegistration(param)

dm = DeformableMirror(telescope = tel,
                        nSubap=10,
                        mechCoupling=0.35,
                        print_dm_properties=False,
                        pitch=0.11,
                        misReg = m,
                        sign=-1e-5) #influence function peak? (not found in OZIIRIS)




param = np.load('Examples/dm_second_stage_misreg_dict.npy', allow_pickle=True).item()
m = MisRegistration(param)

dm = DeformableMirror(telescope = tel,
                        nSubap=10,
                        mechCoupling=0.35,
                        print_dm_properties=False,
                        pitch=0.11,
                        misReg = m,
                        sign=-1e-5)

M2C = np.load('Examples/M2C_KL.npy')




from Papyrus2ndStage import Papyrus2ndStage #the cnn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cnn = Papyrus2ndStage().to(device = device)

checkpoint_path = 'Examples/OziNewDM2.pth' #cnn weights?
checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
cnn.load_state_dict(checkpoint['PhaseEstimator_state_dict'])


from Frame_Preprocess import Frame_Preprocess

pupil_centers = np.array([[ 81, 128],
                          [175, 128]])

frame_preprocessor = Frame_Preprocess(pupil_centers)


wfs_frames = np.load('Examples/wfs_frames.npy')
reference_intensity = np.load('Examples/reference_intensity.npy')
#TODO records the reference
frame_preprocessor.ProcessReference(reference_intensity)

pupils = frame_preprocessor.ProcessFrame(wfs_frames[0])
cnn_input = torch.from_numpy(pupils).unsqueeze(0).float()
output = cnn(cnn_input).detach().cpu().numpy().squeeze()

dm.coefs = M2C[:, :50] @ output


print(wfs_frames.shape)
plt.figure()
plt.imshow(wfs_frames[0])

plt.figure()
plt.imshow(frame_preprocessor.reference)


plt.figure()
plt.imshow(pupils[1])

plt.show()
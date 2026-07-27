import numpy as np
from ozitelemetry.PSFTelemetry import PSFTele
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from astropy.io import fits


savedir = "D:/bench_sky_04_14/bench_1st200_2nd400_v8_atm_4_3520260415-110810"
filename = "RL v8_2-CL-2026-04-15T11_16_59-Cube"
#v10 for both innit


strehls = []
time_array = np.arange(0, 50, 1) * 1/4

for i in range(50):
    psf_telemetry = PSFTele(
        tele_path=f"{savedir}/{filename}.fits",
        is_onsky= False,
        is_cl= True,
        crop_img=100,
        temporal_crop = [i * 100, (i+1) * 100]
    )

    psf_telemetry.psf_analysis(
        elevation_deg=63,
        polychromatic = False #False for on bench since you are using a monochromatic source
    )
    strehls.append(psf_telemetry.SR)



#PLOTS the strehl timeseries
plt.figure()
plt.plot(time_array, strehls)
plt.title('Strehl timeseries')
plt.ylabel("strehl")
plt.xlabel("seconds")
plt.savefig(f'{savedir}/{filename}.png', dpi=150, bbox_inches='tight')
plt.show()




"""
Define constants for Papyrus
"""

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

TELESCOPE = {'DIAMETER':1.52, # [m]
             'OBSTRUCTION':0.27 # ratio
             }

TELESCOPE = dotdict(TELESCOPE)

#changed by ME to correspond to 2nd dm?
DM = {'D_CALIB':37.5, # [mm]
      'D_SKY':35.5, # [mm]
      'PITCH':3.75, # [mm]
      'NACT':97 # number of actuators
      }

DM = dotdict(DM)


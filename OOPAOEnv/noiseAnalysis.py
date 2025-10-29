# %%

import torch
#torch.set_deterministic(True)
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from PO4AO.util_simple import read_yaml_file


import os
# Parser
from Conf.parser_Configurations import Config, ConfigAction
import argparse
from types import SimpleNamespace

args = SimpleNamespace(**read_yaml_file('../Conf/razor_config_po4ao.yaml'))
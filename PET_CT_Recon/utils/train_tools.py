import torch
import torch.nn as nn
import numpy as np
import importlib
import os
import shutil
import sys


def check_dir_exist(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
    else:
        pass


def load_net_module(net_name):
    """
    Load network module
    :param net_name: the name of network module
    :return: the module object
    """
    lib = importlib.import_module("PET_CT_Recon.network." + net_name)
    return lib




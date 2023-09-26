# create by wangerxiao on 20221013
import os
import codecs
import torch
import importlib

from .transform import VarNetDataTransform
from md.utils.python.file_tools import readlines

def load_config(config_file):
    """
    Load a training config file
    :param config_file: the path of config file with py extension
    :return: a configuration dictionary
    """
    dirname = os.path.dirname(config_file)
    basename = os.path.basename(config_file)
    modulename, _ = os.path.splitext(basename)

    os.sys.path.append(dirname)
    lib = importlib.import_module(modulename)
    del os.sys.path[-1]

    return lib.cfg


def readlines(file):
    """
    read lines by removing '\n' in the end of line
    :param file: a text file
    :return: a list of line strings
    """
    fp = codecs.open(file, 'r', encoding='utf-8')
    linelist = fp.readlines()
    fp.close()
    for i in range(len(linelist)):
        linelist[i] = linelist[i].rstrip('\n')
    return linelist


def read_imlist_file(FullList='/data/temp/MRreconraw/T2FullList.txt'):
    """
    read image list file
    :param: refList: ref t2 list, t1DownList: t1 flair downsampled list, t2DownList: t2 flair downsampled list, t1FullList: t1 flair full list, t2FullList: t2 flair full list
    :return: list of ref, t1 flair downsampled, t2 flair downsampled, t1 flair full sampled, and t2 flair full sampled
    """
    ff = readlines(FullList)

    f_list = []
    for i in range(len(ff)):
        f_list.append(ff[i])

    return f_list


def create_dataloader(dataset, dataset_opt):
    phase = dataset_opt['phase']
    if phase == 'train':
        batch_size = dataset_opt['batch_size']
        shuffle = True
        num_workers = dataset_opt['n_workers']
    else:
        batch_size = 1
        shuffle = False
        num_workers = 1
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode'].upper()
    # if mode == 'LR':
    #     from data.LR_dataset import LRDataset as D
    # elif mode == 'LRHR':
    #     from data.LRHR_dataset import LRHRDataset as D
    if mode == "SENSE":
        from data.mri_data import Acs15Dataset as D
    elif mode == "SHARE":
        from data.mri_data_share import MRIDataset as D
    else:
        raise NotImplementedError("Dataset [%s] is not recognized." % mode)
    # initialize dataset transform
    transform = None
    if dataset_opt['transform'] == 'varnet_trans':
        transform = VarNetDataTransform()
    dataset = D(dataset_opt, transform)
    print('===> [%s] Dataset is created.' % (mode))
    return dataset


def create_dataset_val(dataset_opt):
    mode = dataset_opt['mode'].upper()
    # if mode == 'LR':
    #     from data.LR_dataset import LRDataset as D
    # elif mode == 'LRHR':
    #     from data.LRHR_dataset import LRHRDataset as D
    if mode == "SENSE":
        from data.mri_data import Acs15DatasetVal as D
    else:
        raise NotImplementedError("Dataset [%s] is not recognized." % mode)
    # initialize dataset transform
    transform = None
    if dataset_opt['transform'] == 'varnet_trans':
        transform = VarNetDataTransform()
    dataset = D(dataset_opt, transform)
    print('===> [%s] Dataset is created.' % (mode))
    return dataset


# def create_dataloader_val(dataset, dataset_opt):
#     phase = dataset_opt['phase']
#     if phase == 'train':
#         batch_size = dataset_opt['batch_size']
#         shuffle = True
#         num_workers = dataset_opt['n_workers']
#     else:
#         batch_size = 1
#         shuffle = False
#         num_workers = 1
#     return torch.utils.data.DataLoader(
#         dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
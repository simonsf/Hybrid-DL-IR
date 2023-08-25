import argparse
import glob
import os
import time
import importlib
import csv
from easydict import EasyDict as edict
import numpy as np
import torch
import SimpleITK as sitk

from PET_CT_Recon.utils.train_tools import *
from PET_CT_Recon.impl.pyimpl import ReconPET


def load_model(model_dir):
    """ load python segmentation model from folder
    :param model_dir: model directory
    :return: None
    """
    if not os.path.isdir(model_dir):
        raise ValueError('model dir not found: {}'.format(model_dir))

    param_file = os.path.join(model_dir, 'param_PET.pt')

    if not os.path.isfile(param_file):
        raise ValueError('param file not found: {}'.format(param_file))

    model = edict()

    # load network parameters
    state = torch.load(param_file, map_location=lambda storage, loc: storage)
    net_module = load_net_module("PET_recon_unet")
    net = net_module.ReconPETUNet(5, 1, 48)

    net = net.cuda()
    net.load_state_dict(state['model'])
    net.eval()

    model.net = net
    model.pad = 2

    return model


def set_image_meta_info(image, ref):
    image.SetDirection(ref.GetDirection())
    image.SetOrigin(ref.GetOrigin())
    image.SetSpacing(ref.GetSpacing())

    return image


def inference(input_path, model_folder, output_folder, gpu_id=0):

    assert os.path.isfile(input_path), "input file must exist!"
    assert torch.cuda.is_available(), "CUDA is not available! Please check nvidia driver!"

    torch.cuda.set_device(gpu_id)
    if model_folder != "":
        model_dir = model_folder
    else:
        model_dir = os.path.join(os.path.dirname(__file__), "models")

    model = load_model(model_dir)

    image = sitk.ReadImage(input_path)
    image_array = sitk.GetArrayFromImage(image)

    case_name = os.path.basename(os.path.dirname(input_path))
    try:
        prediction_array = ReconPET(image_array, model)
    except Exception as e:
        print('fails to Recon volume: ', input_path, ', {}'.format(e))

    out_folder = os.path.join(output_folder, case_name)
    check_dir_exist(out_folder)

    prediction_path = os.path.join(out_folder, "prediction.nii.gz")
    prediction = sitk.GetImageFromArray(prediction_array)
    prediction = set_image_meta_info(prediction, image)
    sitk.WriteImage(prediction, prediction_path, useCompression=True)


def do_inference():
    from argparse import RawTextHelpFormatter

    long_description = 'UII Recon Batch validation Engine\n\n'

    parser = argparse.ArgumentParser(description=long_description,
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument('-i', '--input', type=str, default="v1_data/PET/1/low_dose.nii.gz", help="input folder/file for intensity images")
    parser.add_argument('-m', '--model', type=str, default="", help="model root folder")
    parser.add_argument('-o', '--output', type=str, default="v1_data/PET", help="output folder")
    parser.add_argument('-g', '--gpu_id', default='3', help="the gpu id to run model")
    args = parser.parse_args()

    inference(args.input, args.model, args.output, int(args.gpu_id))


if __name__ == '__main__':
    do_inference()

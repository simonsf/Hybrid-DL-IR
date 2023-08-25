
import numpy as np

import torch


def ReconPET(image_array, model):
    net = model.net
    pad = model.pad

    image_array = image_array.copy()
    image_array_copy = image_array.copy()
    array_max, array_min = np.percentile(image_array, [99.5, 0])
    image_array = (image_array - array_min) / (array_max - array_min)
    # image_array = (image_array - image_mean_std[0]) / image_mean_std[1]

    image_array_size = image_array.shape
    padding_array_size = np.array([image_array_size[0] + pad * 2, image_array_size[1], image_array_size[2]],
                                  dtype=np.int32)

    padding_image_array = np.zeros(padding_array_size, dtype=np.float32)

    padding_image_array[pad:image_array_size[0] + pad, :, :] = image_array

    if pad != 0:
        padding_image_array[0:pad, :, :] = image_array[pad:0:-1, :, :]
        padding_image_array[image_array_size[0] + pad:image_array_size[0] + 2 * pad, :, :] = image_array[image_array_size[0] - 2:image_array_size[0] - pad - 2:-1, :, :]

    # eps = 1e-6
    overlapping_array = np.zeros(image_array_size, dtype=np.float32)
    prediction_array = np.zeros(image_array_size, dtype=np.float32)

    step_size = [1, image_array_size[1], image_array_size[2]]
    patch_size = [pad * 2 + 1, image_array_size[1], image_array_size[2]]
    pred_patch_size = [1, image_array_size[1], image_array_size[2]]
    for i in range(0, padding_array_size[0] - patch_size[0] + 1, step_size[0]):
        for j in range(0, padding_array_size[1] - patch_size[1] + 1, step_size[1]):
            for k in range(0, padding_array_size[2] - patch_size[2] + 1, step_size[2]):
                array_patch = padding_image_array[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]]

                if array_patch.ndim == 2:
                    array_patch = np.expand_dims(array_patch, 0)

                input_tensor = torch.from_numpy(array_patch)
                input_tensor = torch.unsqueeze(input_tensor, 0)

                input_tensor = input_tensor.cuda()

                res_tensor = input_tensor[:, pad, ...].clone()
                with torch.no_grad():
                    out_tensor = net(input_tensor, res_tensor)

                out_tensor = out_tensor.cpu()
                out_tensor = torch.squeeze(out_tensor, 0)
                out_array = out_tensor.numpy()
                if len(out_array.shape) == 2:
                    out_array = np.expand_dims(out_array, 0)

                prediction_array[i:i + pred_patch_size[0], j:j + pred_patch_size[1],
                k:k + pred_patch_size[2]] = prediction_array[i:i + pred_patch_size[0], j:j + pred_patch_size[1],
                                            k:k + pred_patch_size[2]] + out_array
                overlapping_array[i:i + pred_patch_size[0], j:j + pred_patch_size[1],
                k:k + pred_patch_size[2]] = overlapping_array[i:i + pred_patch_size[0], j:j + pred_patch_size[1],
                                            k:k + pred_patch_size[2]] + 1

    prediction_array /= overlapping_array

    # prediction_array = prediction_array * target_mean_std[1] + target_mean_std[0]
    prediction_array = prediction_array * (array_max - array_min) + array_min
    prediction_array[np.where(prediction_array <= 0)] = image_array_copy[np.where(prediction_array <= 0)]

    return prediction_array


def ReconCT(image_array, model):
    net = model.net
    pad = model.pad

    image_array = image_array.copy()
    image_array_copy = image_array.copy()
    array_max, array_min = np.percentile(image_array, [99.8, 0.1])
    image_array = (image_array - array_min) / (array_max - array_min)
    # image_array = (image_array - image_mean_std[0]) / image_mean_std[1]

    image_array_size = image_array.shape
    padding_array_size = np.array([image_array_size[0] + pad * 2, image_array_size[1], image_array_size[2]],
                                  dtype=np.int32)

    padding_image_array = np.zeros(padding_array_size, dtype=np.float32)

    padding_image_array[pad:image_array_size[0] + pad, :, :] = image_array

    if pad != 0:
        padding_image_array[0:pad, :, :] = image_array[pad:0:-1, :, :]
        padding_image_array[image_array_size[0] + pad:image_array_size[0] + 2 * pad, :, :] = image_array[image_array_size[0] - 2:image_array_size[0] - pad - 2:-1, :, :]

    # eps = 1e-6
    overlapping_array = np.zeros(image_array_size, dtype=np.float32)
    prediction_array = np.zeros(image_array_size, dtype=np.float32)

    step_size = [1, image_array_size[1], image_array_size[2]]
    patch_size = [pad * 2 + 1, image_array_size[1], image_array_size[2]]
    pred_patch_size = [1, image_array_size[1], image_array_size[2]]
    for i in range(0, padding_array_size[0] - patch_size[0] + 1, step_size[0]):
        for j in range(0, padding_array_size[1] - patch_size[1] + 1, step_size[1]):
            for k in range(0, padding_array_size[2] - patch_size[2] + 1, step_size[2]):
                array_patch = padding_image_array[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]]

                if array_patch.ndim == 2:
                    array_patch = np.expand_dims(array_patch, 0)

                input_tensor = torch.from_numpy(array_patch)
                input_tensor = torch.unsqueeze(input_tensor, 0)

                input_tensor = input_tensor.cuda()

                with torch.no_grad():
                    out_tensor = net(input_tensor)

                out_tensor = out_tensor.cpu()
                out_tensor = torch.squeeze(out_tensor, 0)
                out_array = out_tensor.numpy()
                if len(out_array.shape) == 2:
                    out_array = np.expand_dims(out_array, 0)

                prediction_array[i:i + pred_patch_size[0], j:j + pred_patch_size[1],
                k:k + pred_patch_size[2]] = prediction_array[i:i + pred_patch_size[0], j:j + pred_patch_size[1],
                                            k:k + pred_patch_size[2]] + out_array
                overlapping_array[i:i + pred_patch_size[0], j:j + pred_patch_size[1],
                k:k + pred_patch_size[2]] = overlapping_array[i:i + pred_patch_size[0], j:j + pred_patch_size[1],
                                            k:k + pred_patch_size[2]] + 1

    prediction_array /= overlapping_array

    # prediction_array = prediction_array * target_mean_std[1] + target_mean_std[0]
    prediction_array = prediction_array * (array_max - array_min) + array_min
    prediction_array[np.where(prediction_array <= 0)] = image_array_copy[np.where(prediction_array <= 0)]

    return prediction_array

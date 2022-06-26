import torch
import os
from ..networks import *
import torchvision.transforms as transforms
from Segmentation.Utils import transform_parsing
import numpy as np
from ..image_process import process
import PIL.Image as Image


def __load_model(pretrained_model):
    print("[INFO] Loading segmentation model...")
    model = init_model('resnet101', num_classes=20, pretrained=None)

    state_dict = torch.load(pretrained_model)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    return model


model = __load_model("/content/drive/MyDrive/SCHP/exp_schp_multi_cihp_local.pth")


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def multi_scale_testing(model, batch_input_im, crop_size=[473, 473], flip=True, multi_scales=[1]):
    flipped_idx = (15, 14, 17, 16, 19, 18)
    if len(batch_input_im.shape) > 4:
        batch_input_im = batch_input_im.squeeze()
    if len(batch_input_im.shape) == 3:
        batch_input_im = batch_input_im.unsqueeze(0)

    interp = torch.nn.Upsample(size=crop_size, mode='bilinear', align_corners=True)
    ms_outputs = []
    for s in multi_scales:
        interp_im = torch.nn.Upsample(scale_factor=s, mode='bilinear', align_corners=True)
        scaled_im = interp_im(batch_input_im)
        parsing_output = model(scaled_im)
        parsing_output = parsing_output[0][-1]
        output = parsing_output[0]
        if flip:
            flipped_output = parsing_output[1]
            flipped_output[14:20, :, :] = flipped_output[flipped_idx, :, :]
            output += flipped_output.flip(dims=[-1])
            output *= 0.5
        output = interp(output.unsqueeze(0))
        ms_outputs.append(output[0])
    ms_fused_parsing_output = torch.stack(ms_outputs)
    ms_fused_parsing_output = ms_fused_parsing_output.mean(0)
    ms_fused_parsing_output = ms_fused_parsing_output.permute(1, 2, 0)  # HWC
    parsing = torch.argmax(ms_fused_parsing_output, dim=2)
    parsing = parsing.data.cpu().numpy()
    ms_fused_parsing_output = ms_fused_parsing_output.data.cpu().numpy()
    return parsing, ms_fused_parsing_output


def main(input_image: torch.Tensor, gpu: str = '0'):
    gpus = [int(i) for i in gpu.split(',')]
    assert len(gpus) == 1
    if not gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    input_size = (473, 473)

    IMAGE_MEAN = model.mean
    IMAGE_STD = model.std
    # INPUT_SPACE = model.input_space

    palette = get_palette(20)

    scales = np.zeros((1, 2), dtype=np.float32)
    centers = np.zeros((1, 2), dtype=np.int32)

    with torch.no_grad():
        transform = transforms.Compose([
            transforms.ToTensor(),
            # BGR2RGB_transform(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

        image, meta = process(input_image, transform, input_size)

        if len(image.shape) > 4:
            image = image.squeeze()

        c = meta['center']
        s = meta['scale']
        w = meta['width']
        h = meta['height']

        scales[0, :] = s
        centers[0, :] = c
        parsing, logits = multi_scale_testing(model, image.cuda(), crop_size=input_size, flip=False,
                                              multi_scales=[1])

        parsing_result = transform_parsing(parsing, c, s, w, h, input_size)
        output_im = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
        output_im.putpalette(palette)
        # logits_result = transform_logits(logits, c, s, w, h, input_size)

    return output_im

import os 
import time
import math
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
from torchvision import transforms

from libs.models import *

from libs.utils.metrics import Evaluator
from libs.datasets import TestFullSizeLoader
from mmcv.cnn import get_model_complexity_info

import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

categories = [
    'Impervious surface',
    'Building',
    'Low vegetation',
    'Tree',
    'Car',
    'Clutter/Background'
]

def get_colormap():
    return np.array(
                    [
                        [255, 255, 255],
                        [0,   0,   255],
                        [0,   255, 255],
                        [0,   255, 0  ],
                        [255, 255, 0  ],
                        [255, 0,   0  ],
                        [0,   0,   0  ]
                    ]
                )

def encode_segmap(mask,):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
        (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_colormap()):
        if ii == 6:
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = 255
        else:
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        # label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask

def decode_segmap(label_mask, n_classes = 6):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
        the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
        in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = get_colormap()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes+1):
        if ll != n_classes:
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        else:
            r[label_mask == 255] = label_colours[ll, 0]
            g[label_mask == 255] = label_colours[ll, 1]
            b[label_mask == 255] = label_colours[ll, 2]

    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb

def tta_inference(inp, model, num_classes=6, scales=[1.0], flip=True):
    b, _, h, w = inp.size()
    preds = inp.new().resize_(b, num_classes, h, w).zero_().to(inp.device)
    for scale in scales:
        size = (int(scale*h), int(scale*w))
        resized_img = F.interpolate(inp, size=size, mode='bilinear', align_corners=True,)
        pred = model_inference(model, resized_img.to(inp.device), flip)
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True,)
        preds += pred

    return preds/(len(scales))

def model_inference(model, image, flip=True):
    output = model(image)[0]
    if flip:
        fimg = image.flip(2)
        output += model(fimg)[0].flip(2)
        fimg = image.flip(3)
        output += model(fimg)[0].flip(3)
        return output/3
    

    return output
    
def slide(model, scale_image, num_classes=6, crop_size=512, overlap=1/2, scales=[1.0], flip=True):

    N, C, H_, W_ = scale_image.shape
    print(f"Height: {H_} Width: {W_}")
    
    full_probs = torch.zeros((N, num_classes, H_, W_), device=scale_image.device) #
    count_predictions = torch.zeros((N, num_classes, H_, W_), device=scale_image.device) #

    h_overlap_length = int((1-overlap)*crop_size)
    w_overlap_length = int((1-overlap)*crop_size)

    h = 0
    slide_finish = False
    while not slide_finish:

        if h + crop_size <= H_:
            print(f"h: {h}")
            # set row flag
            slide_row = True
            # initial row start
            w = 0
            while slide_row:
                if w + crop_size <= W_:
                    print(f" h={h} w={w} -> h'={h+crop_size} w'={w+crop_size}")
                    patch_image = scale_image[:, :, h:h+crop_size, w:w+crop_size]
                    #
                    patch_pred_image = tta_inference(patch_image, model, num_classes=num_classes, scales=scales, flip=flip)
                    count_predictions[:,:,h:h+crop_size, w:w+crop_size] += 1
                    full_probs[:,:,h:h+crop_size, w:w+crop_size] += patch_pred_image

                else:
                    print(f" h={h} w={W_-crop_size} -> h'={h+crop_size} w'={W_}")
                    patch_image = scale_image[:, :, h:h+crop_size, W_-crop_size:W_]
                    #
                    patch_pred_image = tta_inference(patch_image, model, num_classes=num_classes, scales=scales, flip=flip)
                    count_predictions[:,:,h:h+crop_size, W_-crop_size:W_] += 1
                    full_probs[:,:,h:h+crop_size, W_-crop_size:W_] += patch_pred_image
                    slide_row = False

                w += w_overlap_length

        else:
            print(f"h: {h}")
            # set last row flag
            slide_last_row = True
            # initial row start
            w = 0
            while slide_last_row:
                if w + crop_size <= W_:
                    print(f"h={H_-crop_size} w={w} -> h'={H_} w'={w+crop_size}")
                    patch_image = scale_image[:,:,H_-crop_size:H_, w:w+crop_size]
                    #
                    patch_pred_image = tta_inference(patch_image, model, num_classes=num_classes, scales=scales, flip=flip)
                    count_predictions[:,:,H_-crop_size:H_, w:w+crop_size] += 1
                    full_probs[:,:,H_-crop_size:H_, w:w+crop_size] += patch_pred_image

                else:
                    print(f"h={H_-crop_size} w={W_-crop_size} -> h'={H_} w'={W_}")
                    patch_image = scale_image[:,:,H_-crop_size:H_, W_-crop_size:W_]
                    #
                    patch_pred_image = tta_inference(patch_image, model, num_classes=num_classes, scales=scales, flip=flip)
                    count_predictions[:,:,H_-crop_size:H_, W_-crop_size:W_] += 1
                    full_probs[:,:,H_-crop_size:H_, W_-crop_size:W_] += patch_pred_image

                    slide_last_row = False
                    slide_finish = True

                w += w_overlap_length

        h += h_overlap_length

    full_probs /= count_predictions

    return full_probs

def main(args):

    # loader
    data_transforms = {
        "potsdam": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4752, 0.3216, 0.3188], [0.2108, 0.1484, 0.1431])
        ]),
        "vaihingen": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4752, 0.3216, 0.3188], [0.2108, 0.1484, 0.1431])
        ]),
    }

    test_set = TestFullSizeLoader.TestFullSize(root=args.data_dir, data_list=args.test_list,\
                transform=data_transforms[args.dataset],)

    test_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    # load model
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    ########################################################################################
    model = eval(args.model)(nclass=args.num_classes, backbone=args.backbone, aux=True)
    ########################################################################################

    flops, params = get_model_complexity_info(
        model, 
        input_shape=(3,512,512), 
        print_per_layer_stat=False, 
        as_strings=True)
    
    print(f"Input shape:{(3,512,512)}\nflops:{flops}\nparams:{params}")

    model.load_state_dict(checkpoint)
    model = model.cuda()

    evaluator = Evaluator(args.num_classes)
    evaluator.reset()

    model.eval()
    with torch.no_grad():
        tqdm_test_loader = tqdm(test_loader, total=len(test_loader))
        for num_iter, (images, gts, name) in enumerate(tqdm_test_loader):
            
            # print(f"num_iter: {num_iter}")

            images = images.cuda()
            gts = gts.cuda()
            assert images.shape[2:] == gts.shape[1:], \
                f"images shape: {images.shape} gts shape: {gts.shape}"

            preds = slide(
                model, 
                images, 
                num_classes=args.num_classes, 
                crop_size=512, 
                overlap=args.overlap, 
                scales=args.scales, 
                flip=args.flip)
            
            preds = preds.data.cpu().numpy()
            gts = gts.cpu().numpy()
            preds = np.argmax(preds, axis=1)

            if args.save_flag:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                pred_images = decode_segmap(preds.squeeze())
                image_save_name = name[0]
                img_save_path = os.path.join(args.save_dir, image_save_name+'.png')
                arrimage = Image.fromarray(pred_images)
                arrimage.save(img_save_path)
            
            evaluator.add_batch(gts, preds)

    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    F1, mF1 = evaluator.Calculate_F1_Score()

    print(f"Test model: {args.model}")
    print(f"Test checkpoint: {args.checkpoint}")
    print(f"Test overlap: {args.overlap}")
    print(f"Test scales: {args.scales}")
    print(f"Test flip: {args.flip}")

    print(f"\n\nEvaluation Metrics:\nAcc: {Acc:.5f} \nAcc_class: {Acc_class:.5f}"
        f"\nmIoU: {mIoU:.5f} \nmF1: {mF1:.5f} \nCategory F1: {dict(zip(categories, F1))}\n")


if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def get_arguments():
        """
        Parse all the arguments
        Returns: args
        A list of parsed arguments.
        """
        parser = argparse.ArgumentParser(description="TGARSLetter Competition")
        parser.add_argument("--batch_size", type=int, default=1,
                            help="Number of images sent to the network in one step.")
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--dataset", type=str, choices=['potsdam', 'vaihingen'],
                            help="choose dataset. ['potsdam', 'vaihingen'].")
        parser.add_argument("--data_dir", type=str, default="./data",
                            help="Path to the directory containing the Cityscapes dataset.")
        parser.add_argument("--save_dir", type=str, default="./data",
                            help="Path to the directory containing the Cityscapes dataset.")
        parser.add_argument("--test_list",  nargs='+', required=True,
                            help="Path to the test set listing the images in the dataset.")
        parser.add_argument("--model", type=str, default='DeepLab_ResNet50', help="model architecture")
        parser.add_argument("--ignore_label", type=int, default=255,
                            help="The index of the label to ignore during the training.")
        parser.add_argument("--num_classes", type=int, default=6,
                            help="Number of classes to predict (including background).")
        parser.add_argument("--backbone", type=str, default='resnet50',
                            help="Number of classes to predict (including background).")
        parser.add_argument("--checkpoint", type=str, default=None,
                            help="Where restore models parameters from.")
        parser.add_argument("--overlap", type=float, default=1/3,
                    help="Where restore models parameters from.")
        parser.add_argument("--scales", nargs='+', type=float,
                    help="Where restore models parameters from.")
        parser.add_argument("--flip", type=str2bool, default=False,
                    help="Where restore models parameters from.")
        parser.add_argument("--save_flag", type=str2bool, default=False,
                    help="Where restore models parameters from.")

        args = parser.parse_args()
        return args

    args = get_arguments()

    start = time.time()
    main(args)
    runtime = time.time() - start
    print(f"Spend Time: {math.floor(runtime//3600):2d}h:"
    f"{math.floor(runtime%3600//60):2d}m:{math.floor(runtime%60):2d}s")
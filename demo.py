import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

#img = np.array(Image.open(imfile)).astype(np.uint8)

def load_image(file_path, image_nb_elements, idx, image_size, image_dims):
    img = np.reshape(np.fromfile(file_path, 'float32', image_nb_elements, '', idx * image_size), image_dims)
    img *= 255
    img = np.transpose(img, (1, 0, 2))
    img = img.astype(np.uint8)
    img_ = img
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    cv2.imwrite("/home/mohamed/Desktop/data/result_raft/1.png", img_)
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imwrite("/home/mohamed/Desktop/data/result_raft/2.png", img_flo[:, :, [2,1,0]]/255.0)
    cv2.imshow('image', img_flo[:, :, ]/255.0)
    cv2.waitKey()


def demo(args):

    dims = [640, 512, 3]
    img_number_elements = dims[0]*dims[1]*dims[2]
    number_frames = 20
    FLOAT_SIZE = 4


    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images_idxs = [i for i in range(number_frames)]
        images = sorted(images)
        for img_idx_1, img_idx_2 in zip(images_idxs[:-1], images_idxs[1:]):
            image1 = load_image(args.path, img_number_elements, img_idx_1, img_number_elements*FLOAT_SIZE, dims)
            image2 = load_image(args.path, img_number_elements, img_idx_2, img_number_elements*FLOAT_SIZE, dims)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)

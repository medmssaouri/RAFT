import sys
import os
current_path = os.path.dirname(os.path.realpath(__file__))
print("directory is ", current_path)
sys.path.append(os.path.join(current_path, 'core'))

import timeit
import argparse
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


DEVICE = 'cuda'
FLOAT_SIZE = 4


def load_image(file_path, image_nb_elements, idx, image_size, image_dims):
    img = np.reshape(np.fromfile(file_path, 'float32', image_nb_elements, '', idx * image_size), image_dims)
    img *= 255
    img = np.transpose(img, (1, 0, 2))
    img = img.astype(np.uint8)
    img_ = img
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    flo_origin = flo

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
    
    return flo_origin

def demo(args):

    dims = [args.width, args.height, 3]
    img_number_elements = dims[0]*dims[1]*dims[2]
    number_frames = 0
    if args.frames==None :
        number_frames = int(os.path.getsize(args.path)/(img_number_elements*4))
    else :
        number_frames= int(args.frames)

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    result=[]
    result = np.array(result)

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images_idxs = [i for i in range(number_frames)]
        images = sorted(images)
        idx = 1
        for img_idx_1, img_idx_2 in zip(images_idxs[:-1], images_idxs[1:]):
            image1 = load_image(args.path, img_number_elements, img_idx_1, img_number_elements*FLOAT_SIZE, dims)
            image2 = load_image(args.path, img_number_elements, img_idx_2, img_number_elements*FLOAT_SIZE, dims)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flo = viz(image1, flow_up)
            flo = np.transpose(flo, (1, 0, 2))
            flo=flo.flatten()
            result = np.append(result, flo)
            
            idx+=1
            torch.cuda.empty_cache()
        result.astype('float32').tofile(args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--output', help="path to the output bin file")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--width', default=640, help='width of images in bin file')
    parser.add_argument('--height', default=512, help='height of images in bin file')
    parser.add_argument('--frames', default=None, help='Number of frames to process from the bin files')

 
    args = parser.parse_args()

    
    start = timeit.default_timer()    
    demo(args)
    stop = timeit.default_timer()
    print('Time: ', stop - start) 

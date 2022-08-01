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
import matplotlib.pyplot as plt
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
    return img_, img[None].to(DEVICE)


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
    opticalflow_error = 1.
    opticalflow_diverge = [-32768, -32768]
    start_frame = 0
    end_frame =0
    number_frames = int(os.path.getsize(args.path)/(img_number_elements*4))

    if args.start_frame==None and args.end_frame==None:
        start_frame = 0
        end_frame = number_frames
    elif args.start_frame==None:
        end_frame = int(args.end_frame)
    elif args.end_frame==None:
        end_frame = int(number_frames)
    else :
        start_frame = int(args.start_frame)
        end_frame = int(args.end_frame)

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    final_result=[]
    flow_result_final=[]
    final_result_forward=[]
    processed_images=[]
    final_result = np.array(final_result)
    flow_result_final= np.array(flow_result_final)
    final_result_forward= np.array(final_result_forward)
    processed_images= np.array(processed_images)


    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images_idxs = [i for i in range(start_frame, end_frame)]
        images = sorted(images)

        for img_idx_1, img_idx_2 in zip(images_idxs[:-1], images_idxs[1:]):

            print("Calculate flow between frame " + str(img_idx_1) + " and "  + str(img_idx_2))
            
            image1_numpy, image1 = load_image(args.path, img_number_elements, img_idx_1, img_number_elements*FLOAT_SIZE, dims)
            _, image2 = load_image(args.path, img_number_elements, img_idx_2, img_number_elements*FLOAT_SIZE, dims)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up_forward = model(image1, image2, iters=20, test_mode=True)
            _, flow_up_backward = model(image2, image1, iters=20, test_mode=True)

            flo_forward = viz(image1, flow_up_forward)
            flo_backward = viz(image2, flow_up_backward)

            #validate forward optical flow using backward optical flow
            result_sub = flo_backward + flo_forward
            result_flow = [np.sqrt(np.dot(result_sub[i], result_sub[i])) for i in np.ndindex(result_sub.shape[:2])]
            result = [flo_forward[i] if result_flow[j] < opticalflow_error else opticalflow_diverge for j, i in enumerate(np.ndindex(result_sub.shape[:2]))]           
            result = np.reshape(result, (args.height, args.width, 2))

            #write bin file of processed images
            if args.processed_images:
                image1_numpy = np.transpose(image1_numpy, (1, 0, 2))/255
                image1_numpy = image1_numpy.flatten()
                processed_images = np.append(processed_images, image1_numpy)

            #result flow heatmap
            if args.flow_error_heatmaps:
                result_flow = np.reshape(result_flow, (args.height, args.width))
                result_flow[result_flow>5] = np.nan
                fig, ax = plt.subplots()
                im = ax.imshow(result_flow, cmap=plt.cm.jet)
                fig.colorbar(im, ax=ax)
                fig.canvas.draw()
                result_flow = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                result_flow = result_flow.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                result_flow = result_flow[:,:,::-1]
                result_flow = np.transpose(result_flow, (1, 0, 2))
                result_flow = result_flow.flatten()
                flow_result_final = np.append(flow_result_final, result_flow)
            
            #forward flow heat maps
            if args.flow_heatmap : 
                flo_forward = [np.sqrt(np.dot(flo_forward[i], flo_forward[i])) for i in np.ndindex(flo_forward.shape[:2])]
                flo_forward = np.reshape(flo_forward, (args.height, args.width))
                fig, ax = plt.subplots()
                im = ax.imshow(flo_forward, cmap=plt.cm.jet)
                fig.colorbar(im, ax=ax)
                fig.canvas.draw()
                result_flow_forward = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                result_flow_forward = result_flow_forward.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                result_flow_forward = result_flow_forward[:,:,::-1]
                flo_forward_heat = np.transpose(result_flow_forward, (1, 0, 2))
                flo_forward_heat = flo_forward_heat.flatten()               
                final_result_forward = np.append(final_result_forward, flo_forward_heat)

            #save optical flow after validation using backward flow
            flo = np.transpose(result, (1, 0, 2))
            flo = flo.flatten()
            final_result = np.append(final_result, flo)

            torch.cuda.empty_cache()

        final_result.astype('float32').tofile(args.output[:-4]+ "_validated_forwardflow"+ ".bin")
        if args.flow_error_heatmaps: flow_result_final.astype('float32').tofile(args.output[:-4]+ "_error_heatmaps"+ ".bin")
        if args.flow_heatmap: final_result_forward.astype('float32').tofile(args.output[:-4]+ "_forward_heatmaps"+ ".bin")
        if args.processed_images: processed_images.astype('float32').tofile(args.output[:-4]+ "_RGB"+ ".bin")

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
    parser.add_argument('--flow_heatmap', default=False, help='output the optical flow heat maps bin file')
    parser.add_argument('--flow_error_heatmaps', default=False, help='output the optical flow heat maps error bin files')
    parser.add_argument('--processed_images', default=False, help='output bin file containing processed images')
    parser.add_argument('--start_frame', default=None, help='Start frame to process from the bin files')
    parser.add_argument('--end_frame', default=None, help='End frame to process from the bin files')


 
    args = parser.parse_args()

    
    start = timeit.default_timer()    
    demo(args)
    stop = timeit.default_timer()
    print('Time: ', stop - start) 

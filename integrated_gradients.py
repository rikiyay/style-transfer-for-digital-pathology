import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
import h5py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='directory path to the original CRC-DX-TEST dataset')
parser.add_argument('--low-data-dir', type=str,
                    help='directory path to the low-frequency version of CRC-DX-TEST dataset')
parser.add_argument('--state-dict-dir', type=str,
                    help='directory path to the state_dicts')
parser.add_argument('--out-dir', type=str,
                    help='path to save the integrated gradient plot')
parser.add_argument('--idx', type=int,
                    help='image index of the CRC-DX-TEST dataset to draw integrated gradient plot')
parser.add_argument('--kfold', type=int,
                    help='cross-validation fold number to draw integrated gradient plot')
parser.add_argument('--radius', type=int, choices=[14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154],
                    help='radius to draw integrated gradient plot')
parser.add_argument('--reference', type=str, choices=['white', 'black', 'blur', 'uniform', 'gaussian'],
                    help='type of the reference for integrated gradient')

def check_path(path):
    if path.endswith("/"):
        return path[:-1]
    else:
        return path

# https://github.com/distillpub/post--attribution-baselines/blob/master/public/data_gen/utils.py
def get_blurred_image(image, sigma=10):
    if len(image.shape) == 4:
        blurred_images = [gaussian_filter(im, (sigma, sigma, 0)) for im in image]
        return np.stack(blurred_images, axis=0)
    elif len(image.shape) == 3:
        return gaussian_filter(image, (sigma, sigma, 0))
    else:
        return gaussian_filter(image, sigma)

# modified from https://github.com/distillpub/post--attribution-baselines/blob/master/public/data_gen/utils.py
def get_uniform_image(image, minval=0, maxval=255):
    return np.random.randint(low=minval, high=maxval, size=image.shape).astype('uint8')

# modified from https://github.com/distillpub/post--attribution-baselines/blob/master/public/data_gen/utils.py
def get_gaussian_image(image, sigma, minval=0, maxval=255):
    gaussian_image = np.random.normal(0, sigma, image.shape) + image
    return np.clip(gaussian_image, a_min=minval, a_max=maxval).astype('uint8')

def attribute_image_features(algorithm, input, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=target_class,
                                              **kwargs,
                                             )
    return tensor_attributions

def get_model(model_path, num_classes):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, num_classes))
    model.cuda()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model

def get_ref(original_image, reference):
    if reference == 'white':
        ref = np.ones_like(original_image).astype('uint8')*255
    elif reference == 'black':
        ref = np.zeros_like(original_image).astype('uint8')*255
    elif reference == 'blur':
        # sigmas_b = np.linspace(2, 20, num=10)
        ref = get_blurred_image(original_image, sigma=10)
    elif reference == 'uniform':
        ref = get_uniform_image(original_image)
    elif reference == 'gaussian':
        # sigmas_g = np.linspace(10, 190, num=10)
        ref = get_gaussian_image(original_image, sigma=100)
    return ref

transform = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize(imgSize),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

if __name__ == "__main__":
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args = parser.parse_args()
    data_dir = check_path(args.data_dir)
    low_data_dir = check_path(args.low_data_dir)
    state_dict_dir = check_path(args.state_dict_dir)
    out_dir = check_path(args.out_dir)

    imgSize=int(224)
    num_classes = 2
    target_class = torch.tensor([1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_list = glob.glob(data_dir+'/MSIMUT/*.png')
    original_image = np.array(Image.open(img_list[args.idx])).astype('uint8')
    ref = get_ref(original_image, args.reference)

    modelnames = ['style transfer (STRAP)', 'stain augmentation (SA)', 'stain normalization (SN)']

    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    _, ax[0] = viz.visualize_image_attr(None, original_image, 
                                method="original_image", title="Original Image", 
                                plt_fig_axis=(fig, ax[0]), use_pyplot=False)
    for i, experiment in enumerate(['style_transfer', 'stain_augmentation', 'stain_normalization']):
        model_path = state_dict_dir+'/'+experiment+'_'+str(args.kfold)+'.pth'
        model = get_model(model_path, num_classes)
        model.eval()

        path2hdf5_msi = low_data_dir + f'/crc-dx-test_msi_low_{args.radius}.hdf5'
        h5_low = h5py.File(path2hdf5_msi)
        low = h5_low['img'][args.idx]
        low = transform(low).unsqueeze(0)
        low.requires_grad = True

        ig = IntegratedGradients(model)
        attr_ig, delta = attribute_image_features(ig, low, 
                                                baselines=transform(ref).unsqueeze(0), 
                                                return_convergence_delta=True)
        attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))

        _, ax[i+1] = viz.visualize_image_attr(attr_ig, original_image, method="alpha_scaling", 
                                    sign="absolute_value",
                                    show_colorbar=False, 
                                    title=f"{modelnames[i]}",
                                    plt_fig_axis=(fig, ax[i+1]),
                                    use_pyplot = False)

    fig.savefig(out_dir+f'/integrated_gradient_kfold{args.kfold}_{args.reference}_radius{args.radius}_idx{args.idx}.pdf', bbox_inches='tight', pad_inches=0.1)


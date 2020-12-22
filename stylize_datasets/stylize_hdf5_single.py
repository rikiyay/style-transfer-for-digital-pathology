# https://github.com/bethgelab/stylize-datasets

import os
import sys
sys.path.append(os.path.dirname(__file__))

import argparse
import random
import numpy as np
from PIL import Image
from pathlib import Path
import tables
import torch
import torch.nn as nn
import torchvision.transforms
from function import adaptive_instance_normalization
import net

def input_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(torchvision.transforms.Resize(size))
    if crop != 0:
        transform_list.append(torchvision.transforms.CenterCrop(crop))
    transform_list.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(transform_list)
    return transform

def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def stylize_hdf5_single(contents, style_dir, out_path, ids, fnames, labels, alpha=1., content_size=1024, style_size=256, save_size=256):

    # collect style files
    style_dir = Path(style_dir)
    style_dir = style_dir.resolve()
    extensions = ['png', 'jpeg', 'jpg']
    styles = []
    for ext in extensions:
        styles += list(style_dir.rglob('*.' + ext))

    assert len(styles) > 0, 'No images with specified extensions found in style directory' + style_dir
    styles = sorted(styles)
    print('Found %d style images in %s' % (len(styles), style_dir))

    decoder = net.decoder
    vgg = net.vgg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load('models/decoder.pth'))
    vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    crop = 0
    content_tf = input_transform(content_size, crop)
    style_tf = input_transform(style_size, 0)

    hdf5_file = tables.open_file(out_path, mode='w')
    data_shape = (0, save_size, save_size, 3)
    img_dtype = tables.UInt8Atom()
    storage = hdf5_file.create_earray(hdf5_file.root, 'img', img_dtype, shape=data_shape)

    # disable decompression bomb errors
    Image.MAX_IMAGE_PIXELS = None
    skipped_imgs = []

    tile_ids = []
    tile_fnames = []
    tile_labels = []
    
    # actual style transfer as in AdaIN
    for idx in range(len(contents)):
        try:
            content_img = Image.fromarray(contents[idx,:,:,:]).convert('RGB')
            for style_path in random.sample(styles, 1):
                style_img = Image.open(style_path).convert('RGB')

                content = content_tf(content_img)
                style = style_tf(style_img)
                style = style.to(device).unsqueeze(0)
                content = content.to(device).unsqueeze(0)
                with torch.no_grad():
                    output = style_transfer(vgg, decoder, content, style, alpha)
                output = output.cpu().squeeze_(0)
                output_img = torchvision.transforms.ToPILImage()(output)
                output_img = output_img.resize((save_size, save_size), Image.LANCZOS)
                output = np.array(output_img)

                storage.append(output[None])

                style_img.close()    
            content_img.close()
            tile_ids.append(ids[idx])
            tile_fnames.append(fnames[idx])
            tile_labels.append(labels[idx])

        except Exception as err:
            print(f'skipped stylization of {fnames[idx]} because of the following error; {err})')
            skipped_imgs.append(fnames[idx])
            continue
    
    if(len(skipped_imgs) > 0):
        with open(os.path.join(os.path.dirname(out_path), 'skipped_imgs.txt'), 'w') as f:
            for item in skipped_imgs:
                f.write("%s\n" % item)

    hdf5_file.create_array(hdf5_file.root, 'ids', tile_ids)
    hdf5_file.create_array(hdf5_file.root, 'fnames', tile_fnames)
    hdf5_file.create_array(hdf5_file.root, 'labels', tile_labels)
    hdf5_file.close()
# https://github.com/bethgelab/stylize-datasets

import argparse
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms
from function import adaptive_instance_normalization
import net

ImageFile.LOAD_TRUNCATED_IMAGES=True

parser = argparse.ArgumentParser()
parser.add_argument('--content-dir', type=str,
                    help='directory path to a batch of content images')
parser.add_argument('--style-dir', type=str,
                    help='directory path to a batch of style images')
parser.add_argument('--output-dir', type=str, default='output',
                    help='directory to save the output image')
parser.add_argument('--num-styles', type=int, default=6, help='number of styles to \
                        create for each image (default: 6)')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='the weight that controls the degree of \
                        stylization, should be between 0 and 1')
parser.add_argument('--extensions', nargs='+', type=str, default=['png', 'jpeg', 'jpg'], help='list of image \
                        extensions to scan style and content directory for (case sensitive), default: png, jpeg, jpg')
parser.add_argument('--content-size', type=int, default=1024,
                    help='new (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style-size', type=int, default=256,
                    help='new (minimum) size for the style image, \
                    keeping the original size if set to 0')

# random.seed(123456)

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

def main():
    args = parser.parse_args()

    # set content and style directories
    content_dir = Path(args.content_dir)
    style_dir = Path(args.style_dir)
    style_dir = style_dir.resolve()
    output_dir = Path(args.output_dir)
    output_dir = output_dir.resolve()
    assert style_dir.is_dir(), 'Style directory not found'

    # collect content files
    extensions = args.extensions
    assert len(extensions) > 0, 'No file extensions specified'
    content_dir = Path(content_dir)
    content_dir = content_dir.resolve()
    assert content_dir.is_dir(), 'Content directory not found'
    dataset = []
    for ext in extensions:
        dataset += list(content_dir.rglob('*.' + ext))

    assert len(dataset) > 0, 'No images with specified extensions found in content directory' + content_dir
    content_paths = sorted(dataset)
    print(f'Found {len(content_paths)} content images in {content_dir}')

    # collect style files
    styles = []
    for ext in extensions:
        styles += list(style_dir.rglob('*.' + ext))

    assert len(styles) > 0, 'No images with specified extensions found in style directory' + style_dir
    styles = sorted(styles)
    print(f'Found {len(styles)} style images in {style_dir}')

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
    content_tf = input_transform(args.content_size, crop)
    style_tf = input_transform(args.style_size, 0)

    # disable decompression bomb errors
    Image.MAX_IMAGE_PIXELS = None
    skipped_imgs = []
    
    # actual style transfer as in AdaIN
    with tqdm(total=len(content_paths)) as pbar:
        for content_path in content_paths:
            try:
                content_img = Image.open(content_path).convert('RGB')
                fig = plt.figure(figsize=(3*(args.num_styles+1)+1, 6), constrained_layout=False)
                widths = [3, 1] + [3]*args.num_styles
                gs1 = fig.add_gridspec(nrows=2, ncols=args.num_styles+2, width_ratios=widths, wspace=0.1, hspace=0.15)
                f_ax1 = fig.add_subplot(gs1[:,0])
                f_ax1.imshow(np.array(content_img))
                f_ax1.set_title('histopathology image\n(content)', fontsize=12)
                f_ax1.set_xticks([])
                f_ax1.set_yticks([])
                f_ax2 = fig.add_subplot(gs1[:,1])
                f_ax2.plot([0,0],[0,1])
                f_ax2.axis('off')
                for x, style_path in enumerate(random.sample(styles, args.num_styles)):
                    style_img = Image.open(style_path).convert('RGB')

                    content = content_tf(content_img)
                    style = style_tf(style_img)
                    style = style.to(device).unsqueeze(0)
                    content = content.to(device).unsqueeze(0)
                    with torch.no_grad():
                        output = style_transfer(vgg, decoder, content, style,
                                                args.alpha)
                    output = output.cpu().squeeze(0)
                    output = np.array(torchvision.transforms.ToPILImage()(output))

                    rel_path = content_path.relative_to(content_dir)
                    out_dir = output_dir.joinpath(rel_path.parent)

                    # create directory structure if it does not exist
                    if not out_dir.is_dir():
                        out_dir.mkdir(parents=True)

                    content_name = content_path.stem
                    style_name = style_path.stem
                    out_filename = content_name + '-stylized-' + style_name + '.pdf' # content_path.suffix
                    output_name = out_dir.joinpath(out_filename)
                    
                    f_ax3 = fig.add_subplot(gs1[0, x+2])
                    f_ax3.imshow(np.array(style_img))
                    f_ax4 = fig.add_subplot(gs1[1, x+2])
                    f_ax4.imshow(output)
                    
                    f_ax3.set_xticks([])
                    f_ax4.set_xticks([])
                    f_ax3.set_yticks([])
                    f_ax4.set_yticks([])

                    if x == 0:
                        f_ax3.set_title('painting image (style)', fontsize=12)
                        f_ax4.set_title('stylized image (output)', fontsize=12)
                    
                    style_img.close()
                
                fig.savefig(out_dir.joinpath(content_name+'-stylized-'+str(np.random.randint(1e4))+'.pdf'))
                plt.show()
                
                content_img.close()
            except OSError as e:
                print(f'Skipping stylization of {content_path} due to an error')
                skipped_imgs.append(content_path)
                continue
            except RuntimeError as e:
                print(f'Skipping stylization of {content_path} due to an error')
                skipped_imgs.append(content_path)
                continue
            finally:
                pbar.update(1)
            
    if(len(skipped_imgs) > 0):
        with open(output_dir.joinpath('skipped_imgs.txt'), 'w') as f:
            for item in skipped_imgs:
                f.write("%s\n" % item)

if __name__ == '__main__':
    main()
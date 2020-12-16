import argparse
import h5py
from PIL import ImageFile
from stylize_datasets.stylize_hdf5_single import stylize_hdf5_single

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument('--content-path', type=str,
                    help='path to the content images in hdf5 format')
parser.add_argument('--style-dir', type=str,
                    help='directory path to a batch of style images')
parser.add_argument('--out-path', type=str,
                    help='path to save the stylized dataset in hdf5 format')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='the weight that controls the degree of \
                          stylization, should be between 0 and 1')
parser.add_argument('--content-size', type=int, default=1024,
                    help='new (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style-size', type=int, default=256,
                    help='new (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--save-size', type=int, default=256,
                    help='output size for the stylized image')
# to generate multiple stylized images for each content image, uncomment below
# parser.add_argument('--num-styles', type=int, default=1, help='number of styles to \
#                         create for each image (default: 1)')

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args = parser.parse_args()
    
    h5 = h5py.File(args.content_path)
    imgs = h5['img']
    ids = [i.decode('UTF-8') for i in h5['ids']]
    fnames = [i.decode('UTF-8') for i in h5['fnames']]
    labels = [i for i in h5['labels']]
    
    stylize_hdf5_single(contents=imgs, style_dir=args.style_dir, out_path=args.out_path, ids=ids, fnames=fnames, labels=labels, alpha=args.alpha, content_size=args.content_size, style_size=args.style_size, save_size=args.save_size)
    # to generate multiple stylized images for each content image, comment out the above line and uncomment below
    # stylize_hdf5_single(contents=imgs, style_dir=args.style_dir, out_path=args.out_path, ids=ids, fnames=fnames, labels=labels, alpha=args.alpha, content_size=args.content_size, style_size=args.style_size, save_size=args.save_size, num_styles=args.num_styles)
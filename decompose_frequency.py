import os
import glob
import argparse
import numpy as np
from PIL import Image, ImageFile
import tables

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='directory path to the CRC-DX-TEST dataset')
parser.add_argument('--save-dir', type=str,
                    help='directory path to save low frequency datasets')

# referred to https://github.com/HaohanWang/HFC/blob/master/utility/frequencyHelper.py
def fft(img):
    return np.fft.fft2(img)

def fftshift(img):
    return np.fft.fftshift(fft(img))

def ifft(img):
    return np.fft.ifft2(img)

def ifftshift(img):
    return ifft(np.fft.ifftshift(img))

def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0
    
def mask_radial(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask

def normalize(arr):
    new_arr = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
    return new_arr

def check_path(path):
    if path.endswith("/"):
        return path[:-1]
    else:
        return path

def generateDataWithDifferentFrequencies_3Channel(data_path, save_path, r, image_size=224):
    data_path = check_path(data_path)
    save_path = check_path(save_path)
    labels = ['MSS', 'MSIMUT']

    for x in labels:
        images = images = glob.glob(f'{data_path}/{x}/*.png')
        mask = mask_radial(np.zeros([image_size, image_size]), r)
        
        img_dtype = tables.UInt8Atom()
        data_shape = (0, image_size, image_size, 3)

        h5_path_low = f'{save_path}/crc-dx-test_{x}_low_{r}.hdf5'
        h5_file_low = tables.open_file(h5_path_low, mode='w')
        storage_low = h5_file_low.create_earray(h5_file_low.root, 'img', img_dtype, shape=data_shape)
        
        # h5_path_high = f'{save_path}/freq/CRC-DX-TEST_{x}_high_{r}.hdf5'
        # h5_file_high = tables.open_file(h5_path_high, mode='w')
        # storage_high = h5_file_high.create_earray(h5_file_high.root, 'img', img_dtype, shape=data_shape)
        
        fnames = []
        
        for i in range(len(images)):
            fname = os.path.basename(images[i])
            fnames.append(fname)
            img = np.array(Image.open(images[i]))
            tmp_low = np.zeros([image_size, image_size, 3])
            # tmp_high = np.zeros([image_size, image_size, 3])

            for j in range(3):
                fd = fftshift(img[:, :, j])

                fd_low = fd * mask
                img_low = ifftshift(fd_low)
                tmp_low[:,:,j] = np.real(img_low)

                # fd_high = fd * (1 - mask)
                # img_high = ifftshift(fd_high)
                # tmp_high[:,:,j] = np.real(img_high)

            storage_low.append(normalize(tmp_low)[None])
            # storage_high.append(normalize(tmp_high)[None])
            
        h5_file_low.create_array(h5_file_low.root, 'fnames', fnames)
        h5_file_low.close()
        # h5_file_high.create_array(h5_file_high.root, 'fnames', fnames)
        # h5_file_high.close()

        print(f'decomposing dataset with low pass filter of radius {r} has been completed')

if __name__ == "__main__":
    args = parser.parse_args()

    data_path = args.data_dir # '/path/to/CRC-DX-TEST/dataset'
    save_path = args.save_dir # '/path/to/save/decomposed/dataset'

    radii = [i*14 for i in range(1, 12)]
    for i in radii:
        generateDataWithDifferentFrequencies_3Channel(data_path, save_path, r=i)    

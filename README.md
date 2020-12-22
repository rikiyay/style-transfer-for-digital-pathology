# Learning domain-agnostic visual representation using unrealistic style transfer augmentation in computational pathology  
  
![](images/sample_style_transfer.png)  

This repository contains the code for learning robust and generalizable visual representation using unrealistic style transfer augmentation in digital pathology. We focus on a particular task of classifying colorectal cancer into distinct genetic subtypes called microsatellite status using H&E-stained FFPE histopathology images.  

## Software Requirements  
This code was developed and tested in the following settings.  
### OS  
- Ubuntu 18.04  
### GPU  
- Nvidia GeForce RTX 2080 Ti  
### Dependencies  
- captum (0.2.0)  
- h5py (2.9.0)  
- histomicstk (1.0.3.dev56)  
- matplotlib (3.1.0)  
- numpy: (1.18.1)  
- pandas (0.25.3)  
- pillow (7.0.0)  
- pytables (3.5.1)  
- python (3.6.10)  
- pytorch (1.4.0)  
- scikit-learn (0.21.3)  
- scipy (1.3.2)  
- seaborn (0.11.0)  
- torchvision (0.5.0)  
- tqdm (4.41.1)  

## Installation  
- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers) on your machine (download the distribution that comes with python3).  
  
- Create a conda environment with environment.yml:
```
conda env create -f environment.yml
```  
- Activate the environment:
```
conda activate strap
```
  
## Demo  
### data collection  
- Prepare your own dataset following [this repository](https://github.com/rikiyay/MSINet).
- Download CRC-DX-TRAIN and CRC-DX-TEST datasets from [here](http://doi.org/10.5281/zenodo.2530835).  
- Download the train.zip file of the Kaggle’s Painter by Numbers dataset from [here](https://www.kaggle.com/c/painter-by-numbers/data).  
- Download the miniImageNet dataset from [here](https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view).  
  
### prepare stylized datasets  
```
python create_stylized_dataset.py --content-path /path/to/content_images.hdf5 \
    --style-dir /path/to/style_images --out-path /path/to/save/stylized_dataset.hdf5 \
    --alpha 1.0 --content-size 1024 --style-size 256 --save-size 256  
```
  
### train models  
```
python train.py --path2hdf5 /path/to/development-dataset.hdf5 \
    --save-dir directory /path/to/save/state-dicts --experiment 'style_transfer'  
```
```
python train.py --path2hdf5 /path/to/development-dataset.hdf5 \
    --save-dir /path/to/save/state-dicts --experiment 'stain_augmentation'  
```
```
python train.py --path2hdf5 /path/to/development-dataset.hdf5 \
    --save-dir /path/to/save/state-dicts --experiment 'stain_normalization'  
```

### evaluate models  
```
python eval.py --data-dir /path/to/CRC-DX-TEST-dataset \
    --state-dict-dir /path/to/state-dicts --experiment 'style_transfer'  
```
```
python eval.py --data-dir /path/to/CRC-DX-TEST-dataset \
    --state-dict-dir /path/to/state-dicts --experiment 'stain_augmentation'  
```
```
python eval.py --data-dir /path/to/CRC-DX-TEST-dataset \
    --state-dict-dir /path/to/state-dicts --experiment 'stain_normalization'  
```
    
### create low-frequency datasets  
```
python decompose_frequency.py --data-dir /path/to/CRC-DX-TEST-dataset \
    --save-dir /path/to/save/low-frequency-datasets  
```

### evaluate models on low-frequency datasets  
```
python eval_on_low_freq.py --data-dir /path/to/low-freq-CRC-DX-TEST-dataset \
    --state-dict-dir /path/to/state-dicts --out-dir /path/to/save/low-frequency-results  
```
```
python plot_low_freq_results.py --csv-path /path/to/low-frequency-results.csv \
    --out-dir /path/to/save/low-frequency-plots  
```
```
python integrated_gradients.py --data-dir /path/to/CRC-DX-TEST-dataset \
    --low-data-dir /path/to/low-freq-CRC-DX-TEST-dataset --state-dict-dir /path/to/state-dicts \
    --out-dir /path/to/save/integrated-gradient-plots \
    --idx 25600 --kfold 1 --radius 70 --reference 'uniform'  
```

Note: please edit paths above.  
  
## Citation  
Learning domain-agnostic visual representation using medically-irrelevant style transfer augmentation in computational pathology.  
  
Rikiya Yamashita, Snikitha Banda, Jeanne Shen, Daniel L Rubin  
import os
import numpy as np
import pandas as pd
import argparse
import h5py
from PIL import Image
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='directory path to the low-frequency version of CRC-DX-TEST dataset')
parser.add_argument('--state-dict-dir', type=str,
                    help='directory path to the state_dicts')
parser.add_argument('--out-dir', type=str,
                    help='path to save the results in a csv format')

def check_path(path):
    if path.endswith("/"):
        return path[:-1]
    else:
        return path

def get_model(model_path, num_classes):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, num_classes))
    model.cuda()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model

class DecompDataset(data.Dataset):
    def __init__(self, path2hdf5_mss, path2hdf5_msi, transform=None):

        h50 = h5py.File(path2hdf5_mss)
        h51 = h5py.File(path2hdf5_msi)

        self.h5_img0 = h50['img']
        self.h5_img1 = h51['img']
        
        self.h5_fname0 = [i.decode('UTF-8') for i in h50['fnames']]
        self.h5_fname1 = [i.decode('UTF-8') for i in h51['fnames']]
        self.fname = self.h5_fname0 + self.h5_fname1
        
        self.label = [0 for _ in range(len(self.h5_fname0))] + [1 for _ in range(len(self.h5_fname1))]
        
        self.transform=transform

    def __getitem__(self, index):

        if index < len(self.h5_fname0):
            img = self.h5_img0[index]
        else:
            img = self.h5_img1[index-len(self.h5_fname0)]
            
        fn = self.fname[index]
        lab = self.label[index]
        
        if self.transform is not None:
            img = self.transform(img)
        return img, lab, fn

    def __len__(self):
        return len(self.fname)

def test(model, dataloader, dataset_size, criterion, device):
    running_corrects = 0
    running_loss=0
    pred = []
    true = []
    pred_wrong = []
    true_wrong = []
    image = []
    paths = []
    prob = []

    for batch_idx, (data, target, path) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        data = data.type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.LongTensor)
        model.eval()
        output = model(data)
        loss = criterion(output, target)
        output = nn.Softmax(dim = 1)(output)
        _, preds = torch.max(output, 1)
        running_corrects += torch.sum(preds == target.data)
        running_loss += loss.item() * data.size(0)
        preds = preds.cpu().numpy()
        target = target.cpu().numpy()
        probs = output.detach().cpu().numpy()[:,1]
        preds = np.reshape(preds,(len(preds),1))
        target = np.reshape(target,(len(preds),1))
        data = data.cpu().numpy()

        for i in range(len(preds)):
            pred.append(preds[i])
            true.append(target[i])
            prob.append(probs[i])
            paths.append(path[i])
            if(preds[i]!=target[i]):
                pred_wrong.append(preds[i])
                true_wrong.append(target[i])
                image.append(data[i])

    epoch_acc = running_corrects.double()/dataset_size
    epoch_loss = running_loss/dataset_size
    print(epoch_acc,epoch_loss)
    return true, pred, prob, paths, image, true_wrong, pred_wrong

def bootstrap_auc(y_true, y_pred, n_bootstraps=2000, rng_seed=42):
    n_bootstraps = n_bootstraps
    rng_seed = rng_seed
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        indices = rng.randint(len(y_pred), size=len(y_pred))
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    bootstrapped_scores = np.array(bootstrapped_scores)

    print("AUROC: {:0.3f}".format(roc_auc_score(y_true, y_pred)))
    print("Confidence interval for the AUROC score: [{:0.3f} - {:0.3}]".format(
        np.percentile(bootstrapped_scores, (2.5, 97.5))[0], np.percentile(bootstrapped_scores, (2.5, 97.5))[1]))
    
    return roc_auc_score(y_true, y_pred), np.percentile(bootstrapped_scores, (2.5, 97.5))

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(imgSize),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args = parser.parse_args()
    data_dir = check_path(args.data_dir)
    state_dict_dir = check_path(args.state_dict_dir)
    out_dir = check_path(args.out_dir)

    batchSize=32
    imgSize=int(224)
    num_classes = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()

    radii = [i*14 for i in range(1, 12)]

    for i in radii:
        print(f'*****radius: {i}*****')
        df_save = pd.DataFrame(columns=['experiment', 'kFold', 'radius', 'auroc', 'ci_low', 'ci_high'])
 
        path2hdf5_mss = data_dir + f'/crc-dx-test_mss_low_{i}.hdf5'
        path2hdf5_msi = data_dir + f'/crc-dx-test_msi_low_{i}.hdf5'
        decomp_dataset = DecompDataset(path2hdf5_mss, path2hdf5_msi, transform = transform)
        decomp_loader = data.DataLoader(decomp_dataset, batch_size=batchSize, shuffle=False)
        decomp_datasize = len(decomp_dataset)
        for m in ['style_transfer', 'stain_augmentation', 'stain_normalization']:
            for k in ['1', '2', '3', '4']:
            
                model_path = state_dict_dir+'/'+m+'_'+str(k)+'.pth'
                model = get_model(model_path, num_classes)
            
                true, pred, prob, paths, image, true_wrong, pred_wrong = test(model, decomp_loader, decomp_datasize, criterion, device)
                slides = ['-'.join(os.path.basename(i).split('-')[2:5]) for i in paths]
                unique = list(set(slides))
                df = pd.DataFrame(columns=['id', 'prob', 'pred', 'label'])
                df.id = slides
                df.prob = prob
                df.pred = [i[0] for i in pred]
                df.label = [i[0] for i in true]
                pt_prob=[]
                pt_pred=[]
                pt_label=[]

                for n in range(len(unique)):
                    ave_prob=np.mean(df[df.id==unique[n]].prob.values)
                    y=df[df.id==unique[n]].label.values.tolist()[0]
                    pt_prob.append(ave_prob)
                    pt_label.append(y)
                auc, low, high = bootstrap_auc(np.array(pt_label), np.array(pt_prob))
                print(f'experiment: {m}, kFold: {k}, radius: {i}, auc: {auc} [95%CI: {low}-{high}]')
                tmp_series = pd.Series([m, k, i, auc, low, high], index=df_save.columns)
                df_save = df_save.append(tmp_series, ignore_index=True)
        print(f'successfully done for radius {i}')

    df_save.to_csv(out_dir + f'/eval_low_freq_results.csv', index=False)
    print(f'successfully saved all results in a csv format in {out_dir}')


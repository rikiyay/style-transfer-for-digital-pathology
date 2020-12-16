import os
import time
import random
import copy
import argparse
import numpy as np
import pandas as pd
import h5py
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--path2hdf5', type=str,
                    help='path to the dataset in hdf5 format')
parser.add_argument('--save-dir', type=str,
                    help='directory path to save state_dicts')
parser.add_argument('--experiment', type=str, choices=['style_transfer', 'stain_augmentation', 'stain_normalization'],
                    help='type of the experiment')

def check_path(path):
    if path.endswith("/"):
        return path[:-1]
    else:
        return path

def get_dfDict(path2hdf5):
    h5 = h5py.File(path2hdf5)
    fnames = [i.decode('UTF-8') for i in h5['fnames']]
    ids = [i.decode('UTF-8') for i in h5['ids']]
    labels = [i for i in h5['labels']]

    df = pd.DataFrame(columns=['fnames', 'ids', 'labels'])
    df.fnames = fnames
    df.ids = ids
    df.labels = labels

    df_dict = {}
    for name, group in df.groupby('ids'):
        df_dict[name] = group
    ids = list(df_dict.keys())

    return df_dict, ids

def get_4fold_ids(ids):
    rand = np.arange(len(ids))
    np.random.shuffle(rand)

    train_ids = {}
    val_ids = {}
    test_ids = {}

    for i in range(4):
        test_ids[i+1] = rand[i*25:i*25+15]
        val_ids[i+1] = rand[i*25+15:i*25+25]
        train_ids[i+1] = list(set(rand.tolist())-set(test_ids[i+1])-set(val_ids[i+1]))

    return train_ids, val_ids, test_ids

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path2hdf5, df_dict, train_ids, val_ids, test_ids, k, dset_type, transform=None):
        h5 = h5py.File(path2hdf5)
        self.dset_type = dset_type
        if self.dset_type == 'train':
            train_df = pd.DataFrame(columns=['fnames', 'ids', 'labels'])
            for i in train_ids[k]:
                train_df = train_df.append(df_dict[ids[i]], 'sort=False')
            self.train_df = train_df.reset_index(drop=True)
        elif self.dset_type == 'val':
            val_df = pd.DataFrame(columns=['fnames', 'ids', 'labels'])
            for i in val_ids[k]:
                val_df = val_df.append(df_dict[ids[i]], 'sort=False')
            self.val_df = val_df.reset_index(drop=True)
        elif self.dset_type == 'test':
            test_df = pd.DataFrame(columns=['fnames', 'ids', 'labels'])
            for i in test_ids[k]:
                test_df = test_df.append(df_dict[ids[i]], 'sort=False')
            self.test_df = test_df.reset_index(drop=True)
        self.h5_imgs = h5['img']
        self.h5_fnames = [i.decode('UTF-8') for i in h5['fnames']]
        self.transform=transform

    def __getitem__(self, index):
        if self.dset_type == 'train':
            fname = self.train_df.fnames[index]
            label = self.train_df.labels[index]
        elif self.dset_type == 'val':
            fname = self.val_df.fnames[index]
            label = self.val_df.labels[index]
        elif self.dset_type == 'test':
            fname = self.test_df.fnames[index]
            label = self.test_df.labels[index]
        h5_idx = self.h5_fnames.index(fname)
        img = self.h5_imgs[h5_idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        if self.dset_type == 'train':
            return len(self.train_df)
        elif self.dset_type == 'val':
            return len(self.val_df)
        elif self.dset_type == 'test':
            return len(self.test_df)

def get_datasets_loaders(path2hdf5, df_dict, train_ids, val_ids, test_ids, k, augment, transform, batchSize):
    datasets = {}
    loaders = {}

    for dset_type in ['train', 'val', 'test']:
        if dset_type == 'train':
            datasets[dset_type] = MyDataset(path2hdf5, df_dict, train_ids, val_ids, test_ids, k=k, dset_type='train', transform = augment)
            loaders[dset_type] = torch.utils.data.DataLoader(datasets[dset_type], batch_size=batchSize, shuffle=True)
        elif dset_type == 'val':
            datasets[dset_type] = MyDataset(path2hdf5, df_dict, train_ids, val_ids, test_ids, k=k, dset_type='val', transform = transform)
            loaders[dset_type] = torch.utils.data.DataLoader(datasets[dset_type], batch_size=batchSize, shuffle=True)
        elif dset_type == 'test':
            datasets[dset_type] = MyDataset(path2hdf5, df_dict, train_ids, val_ids, test_ids, k=k, dset_type='test', transform = transform)
            loaders[dset_type] = torch.utils.data.DataLoader(datasets[dset_type], batch_size=batchSize, shuffle=False)
        print('Finished loading %s dataset: %s samples' % (dset_type, len(datasets[dset_type])))
    
    return datasets, loaders

def get_model(num_classes):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, num_classes))

    return model

def train_model(model, loaders, criterion, optimizer, scheduler, num_epochs=40, estop=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float("inf")
    counter = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                if scheduler is not None:
                    scheduler.step(epoch_loss)
            
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_epoch = epoch
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    counter = 0
                else:
                    counter += 1

        print()

        if counter >= estop:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)

    return model, best_loss, best_acc, best_epoch

def test_model(model, loader, dataset_size, criterion):
    
    print('-' * 10)
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    whole_probs = torch.FloatTensor(dataset_size)
    whole_labels = torch.LongTensor(dataset_size)
    
    with torch.no_grad():

        for i, data in enumerate(loader):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            outputs = F.softmax(outputs, dim=1)
            whole_probs[i*batchSize:i*batchSize+inputs.size(0)]=outputs.detach()[:,1].clone()
            whole_labels[i*batchSize:i*batchSize+inputs.size(0)]=labels.detach().clone()

        total_loss = running_loss / dataset_size
        total_acc = running_corrects.double() / dataset_size

    print('Test Loss: {:.4f} Acc: {:.4f}'.format(total_loss, total_acc))

    return whole_probs.cpu().numpy(), whole_labels.cpu().numpy(), total_loss, total_acc

def bootstrap_auc(y_true, y_pred, n_bootstraps=2000, rng_seed=42):
    n_bootstraps = n_bootstraps
    rng_seed = rng_seed
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        indices = rng.randint(len(y_pred), size=len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for AUROC
            # to be defined: reject the sample
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    bootstrapped_scores = np.array(bootstrapped_scores)

    print("AUROC: {:0.3f}".format(roc_auc_score(y_true, y_pred)))
    print("Confidence interval for the AUROC score: [{:0.3f} - {:0.3}]".format(
        np.percentile(bootstrapped_scores, (2.5, 97.5))[0], np.percentile(bootstrapped_scores, (2.5, 97.5))[1]))
    
    return roc_auc_score(y_true, y_pred), np.percentile(bootstrapped_scores, (2.5, 97.5))

def get_augment(experiment):
    if experiment in ['style_transfer', 'stain_normalization']:
        augment = transforms.Compose([transforms.ToPILImage(),
                                    transforms.RandomResizedCrop(imgSize),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    elif experiment == 'stain_augmentation':
        from histomicstk.preprocessing.augmentation.color_augmentation import rgb_perturb_stain_concentration

        class stain_augment(object):
            def __call__(self, sample):
                sample = np.array(sample)
                rgbaug = rgb_perturb_stain_concentration(sample, sigma1=1., sigma2=1.)
                return rgbaug

        augment = transforms.Compose([stain_augment(),
                                transforms.ToPILImage(),
                                transforms.RandomResizedCrop(imgSize),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    return augment

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(imgSize),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

if __name__ == '__main__':

    args = parser.parse_args()
    save_dir = check_path(args.save_dir)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    df_dict, ids = get_dfDict(args.path2hdf5)
    train_ids, val_ids, test_ids = get_4fold_ids(ids)

    batchSize=32
    imgSize=int(224)
    num_classes = 2

    auroc_4cv = []
    ci_4cv = []

    for i in range(4):
        k = i+1
            
        datasets, loaders = get_datasets_loaders(args.path2hdf5, df_dict, train_ids, val_ids, test_ids, k, get_augment(args.experiment), transform, batchSize)
        dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}

        model = get_model(num_classes)
        model = model.to('cuda')

        # print("Params to learn:")
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print("\t",name)

        optimizer = optim.SGD(params_to_update, lr=0.004, momentum=0.9, nesterov=True)
        criterion = nn.CrossEntropyLoss()

        model, best_loss, best_acc, best_epoch = train_model(model, loaders, criterion, optimizer, scheduler=None, num_epochs=40, estop=5)
        if args.experiment == 'style_transfer':
            torch.save(model.state_dict(), save_dir+'/style_transfer_'+str(k)+'.pth')
        elif args.experiment == 'stain_augmentation':
            torch.save(model.state_dict(), save_dir+'/stain_augmentation_'+str(k)+'.pth')
        elif args.experiment == 'stain_normalization':
            torch.save(model.state_dict(), save_dir+'/stain_normalization_'+str(k)+'.pth')

        prob_test, label_test, loss_test, acc_test = test_model(model, loaders['test'], dataset_sizes['test'], criterion)   

        df_cv = pd.DataFrame(columns=['id', 'prob', 'label'])
        df_cv.id = datasets['test'].test_df.ids.values.tolist()
        df_cv.prob = prob_test
        df_cv.label = label_test
        unique=np.unique(df_cv.id.values).tolist()

        pt_prob=[]
        pt_pred=[]
        pt_label=[]
        for i in range(len(unique)):
            ave_prob=np.mean(df_cv[df_cv.id==unique[i]].prob.values)
            ave_pred=1 if ave_prob>0.50 else 0
            label=df_cv[df_cv.id==unique[i]].label.values.tolist()[0]
            pt_prob.append(ave_prob)
            pt_pred.append(ave_pred)
            pt_label.append(label)

        roc_auc, low, high = bootstrap_auc(np.array(pt_label), np.array(pt_prob))
        auroc_4cv.append(roc_auc)
        ci_4cv.append((low, high))
    
    print(f'mean AUROC for {args.experiment} is {np.mean(np.array(auroc_4cv))}; standard deviation is {np.std(np.array(auroc_4cv))}')
    print()
    print(f'each AUROC for {args.experiment} is as follows: {auroc_4cv}')
    print()
    print(f'each 95% CI for {args.experiment} is as follows: {ci_4cv}')
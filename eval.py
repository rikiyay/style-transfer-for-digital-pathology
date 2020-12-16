import os
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='directory path to the CRC-DX-TEST dataset')
parser.add_argument('--state-dict-dir', type=str,
                    help='directory path to the state_dicts')
parser.add_argument('--experiment', type=str, choices=['style_transfer', 'stain_augmentation', 'stain_normalization'],
                    help='type of the experiment')

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

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def test_data(data_path, transform):
    test_dataset = ImageFolderWithPaths(
        root=data_path,
        transform=transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
    )
    return test_dataset, test_loader

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

transform = transforms.Compose([transforms.Resize(imgSize),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

if __name__ == "__main__":
    args = parser.parse_args()
    state_dict_dir = check_path(args.state_dict_dir)

    batchSize=32
    imgSize=int(224)
    num_classes = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset, test_loader = test_data(args.data_dir, transform)
    test_dataset_size = len(test_dataset)
    criterion = nn.CrossEntropyLoss()

    auroc_4cv = []
    ci_4cv = []

    for i in range(4):
        k = i+1

        model_path = state_dict_dir+'/'+args.experiment+'_'+str(k)+'.pth'
        model = get_model(model_path, num_classes)
        true, pred, prob, paths, image, true_wrong, pred_wrong = test(model, test_loader, test_dataset_size, criterion, device)
        
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
        for i in range(len(unique)):
            ave_prob=np.mean(df[df.id==unique[i]].prob.values)
            s=1 if ave_prob>0.80 else 0
            y=df[df.id==unique[i]].label.values.tolist()[0]
            pt_prob.append(ave_prob)
            pt_pred.append(s)
            pt_label.append(y)
        roc_auc, low, high = bootstrap_auc(np.array(pt_label), np.array(pt_prob))
        auroc_4cv.append(roc_auc)
        ci_4cv.append((low, high))

    print(f'mean AUROC for {args.experiment} is {np.mean(np.array(auroc_4cv))}; standard deviation is {np.std(np.array(auroc_4cv))}')
    print()
    print(f'each AUROC for {args.experiment} is as follows: {auroc_4cv}')
    print()
    print(f'each 95% CI for {args.experiment} is as follows: {ci_4cv}')
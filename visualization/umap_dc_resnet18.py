import numpy as np
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import os
import pdb
import random
import torchvision
import umap.umap_ as umap
import random
import numpy as npz
import time


from glob import glob
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from torch.utils import data
from PIL import Image


IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)



def main(args):
    # # - Making the features and saving it as npz file
    # # - Model
    # model = torchvision.models.resnet18(pretrained=True)
    # model_feature = nn.Sequential(*list(model.children())[:-2])
    # # model.fc = nn.Linear(512, 7)
    # model.cuda()
    # model.eval()

    # # - Dataset (Source)
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=mean, std=std), torchvision.transforms.Resize(64)])
    # for i in args.data_path:
    #     _dataset = torchvision.datasets.ImageFolder(i, transform=transform)
    #     _loader = torch.utils.data.DataLoader(_dataset, batch_size=1, shuffle=True)
    #     class_names = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    #     features = []
    #     class_lbls = []
    #     domain_lbls = []


    #     first = True
    #     for _batch in _loader:
    #         img, label = _batch
    #         img, label = img.cuda(), label.cuda()
    #         output = model_feature(img)
    #         if first:
    #             features = np.array([[output.flatten().cpu().detach().numpy()]])
    #             class_lbls = np.array([[label.cpu().detach().numpy()]])
    #             domain_lbls = np.asarray([[0]])
    #         else:
    #             features = np.concatenate((features, np.array([[output.flatten().cpu().detach().numpy()]])))
    #             class_lbls = np.concatenate((class_lbls, np.array([[label.cpu().detach().numpy()]])))
    #             domain_lbls = np.concatenate((domain_lbls, np.asarray([[0]])))
    #         first = False
    #     os.makedirs(args.save_path, exist_ok=True)
    #     np.savez(f'{args.save_path}/{args.save_name}_features.npz', features=features, class_lbls=class_lbls, domain_lbls=domain_lbls)

    # - Loading the features
    if 'supervised' in args.feature_for_umap:
        npz = np.load(f'{args.save_path}/whole_{"_".join(args.feature_for_umap.split("_")[2:])}_features.npz')
        features = npz['features']
        class_lbls = npz['class_lbls']
        domain_lbls = npz['domain_lbls']
        domain_num = 1
    elif 'dg' in args.feature_for_umap:
        domains = ['art_painting', 'cartoon', 'sketch', 'photo']
        domains.remove("_".join(args.feature_for_umap.split("_")[2:]))
        npz = np.load(f'{args.save_path}/whole_{domains[0]}_features.npz')
        features = npz['features']
        class_lbls = npz['class_lbls']
        domain_lbls = npz['domain_lbls']
        for _idx, domain in enumerate(domains[1:]):
            temp_npz = np.load(f'{args.save_path}/whole_{domain}_features.npz')
            features = np.concatenate((features, temp_npz['features']))
            class_lbls = np.concatenate((class_lbls, temp_npz['class_lbls']))
            domain_lbls = np.concatenate((domain_lbls, temp_npz['domain_lbls'] + _idx + 1))
        domain_num = 3
    else:
        print("Error")
    temp_npz = np.load(f'{args.save_path}/{args.feature_for_umap}_features.npz')
    features = np.concatenate((features, temp_npz['features']))
    class_lbls = np.concatenate((class_lbls, temp_npz['class_lbls']+7))
    domain_lbls = np.concatenate((domain_lbls, temp_npz['domain_lbls']+domain_num))

    features = features.reshape(features.shape[0], features.shape[-1])
    class_lbls = class_lbls.reshape(class_lbls.shape[0], class_lbls.shape[-1])
    domain_lbls = domain_lbls.reshape(domain_lbls.shape[0], domain_lbls.shape[-1])

    # - Drawing UMAP Part
    os.makedirs('./tsne/dcdm/by_class', exist_ok=True)
    for y in [30, 100, 300, 500, 1000]:
        for min_dist in [0.1,  0.3, 0.5, 0.7, 0.9]:
            embedding = umap.UMAP(n_neighbors=y, random_state=33, min_dist=min_dist).fit_transform(features)

            # -- For looop for the number of classes and plot by classes
            for _ in range (7):
                _temp_index_src = np.nonzero((class_lbls % 7 == _) & (class_lbls < 7))[0]
                _temp_index_dc = np.nonzero((class_lbls % 7 == _) & (class_lbls >= 7))[0]
                _temp_everything_else = np.nonzero(class_lbls % 7 != _)[0]
                
                viridis_src = plt.get_cmap('Set1', 1)
                viridis_dc = plt.get_cmap('Set2', 1)
                fig, ax = plt.subplots(1, figsize=(14, 10))
                plt.scatter(*embedding[_temp_everything_else].T, c='white', edgecolors='black', s=10, marker=".", alpha=0.3)
                plt.scatter(*embedding[_temp_index_src].T, c=class_lbls[_temp_index_src], s=20, marker="o", cmap=viridis_src, alpha=0.8)
                plt.scatter(*embedding[_temp_index_dc].T, c=class_lbls[_temp_index_dc], s=200, marker="*", cmap=viridis_dc, alpha=0.8)
                plt.setp(ax, xticks=[], yticks=[])
                cbar = plt.colorbar()
                cbar.set_ticks(ticks=list(range(np.unique(class_lbls).shape[0])))
                # cbar.set_ticklabels(class_lbls)
                plt.title(f'By Each Class! {args.feature_for_umap} (n_neighbors={y} seed=33 min_dist={min_dist})')
                plt.savefig(f'./tsne/dcdm/by_class/class_{_}_{args.feature_for_umap}_{y}_{min_dist}.png', bbox_inches='tight')
                
                viridis_src = plt.get_cmap('Set1', domain_num)
                viridis_dc = plt.get_cmap('Set2', 1)
                fig, ax = plt.subplots(1, figsize=(14, 10))
                plt.scatter(*embedding[_temp_everything_else].T, c='white', edgecolors='black', s=10, marker=".", alpha=0.3)
                plt.scatter(*embedding[_temp_index_src].T, c=domain_lbls[_temp_index_src], s=20, marker="o", cmap=viridis_src, alpha=0.8)
                plt.scatter(*embedding[_temp_index_dc].T, c=domain_lbls[_temp_index_dc], s=200, marker="*", cmap=viridis_dc, alpha=0.8)
                plt.setp(ax, xticks=[], yticks=[])
                cbar = plt.colorbar()
                cbar.set_ticks(ticks=list(range(np.unique(domain_lbls).shape[0])))
                # cbar.set_ticklabels(domain_lbls)
                plt.title(f'By Domain! {args.feature_for_umap} (n_neighbors={y} seed=33 min_dist={min_dist})')
                plt.savefig(f'./tsne/dcdm/by_class/domain_for_class_{_}_{args.feature_for_umap}_{y}_{min_dist}.png', bbox_inches='tight')
           
            _temp_index_src = np.nonzero(class_lbls < 7)[0]
            _temp_index_dc = np.nonzero(class_lbls >= 7)[0] 
            
            viridis_src = plt.get_cmap('Set1', 7)
            viridis_dc = plt.get_cmap('Set2', 7)
            fig, ax = plt.subplots(1, figsize=(14, 10))
            plt.scatter(*embedding[_temp_index_src].T, c=class_lbls[_temp_index_src], s=20, marker="o", cmap=viridis_src, alpha=0.8)
            plt.scatter(*embedding[_temp_index_dc].T, c=class_lbls[_temp_index_dc], s=200, marker="*", cmap=viridis_dc, alpha=0.8)
            plt.setp(ax, xticks=[], yticks=[])
            cbar = plt.colorbar()
            cbar.set_ticks(ticks=list(range(np.unique(class_lbls).shape[0])))
            # cbar.set_ticklabels(class_lbls)
            plt.title(f'By Class! {args.feature_for_umap} (n_neighbors={y} seed=33 min_dist={min_dist})')
            plt.savefig(f'./tsne/dcdm/class_{args.feature_for_umap}_{y}_{min_dist}.png', bbox_inches='tight')

            viridis_src = plt.get_cmap('Set1', domain_num)
            viridis_dc = plt.get_cmap('Set2', 1)
            fig, ax = plt.subplots(1, figsize=(14, 10))
            plt.scatter(*embedding[_temp_index_src].T, c=domain_lbls[_temp_index_src], s=20, marker="o", cmap=viridis_src, alpha=0.8)
            plt.scatter(*embedding[_temp_index_dc].T, c=domain_lbls[_temp_index_dc], s=200, marker="*", cmap=viridis_dc, alpha=0.8)
            plt.setp(ax, xticks=[], yticks=[])
            cbar = plt.colorbar()
            cbar.set_ticks(ticks=list(range(np.unique(domain_lbls).shape[0])))
            # cbar.set_ticklabels(domain_lbls)
            plt.title(f'By Domain! {args.feature_for_umap} (n_neighbors={y} seed=33 min_dist={min_dist})')
            plt.savefig(f'./tsne/dcdm/domain_{args.feature_for_umap}_{y}_{min_dist}.png', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '--dp', nargs='+', type=str)
    parser.add_argument('--save_path', '--sp', type=str, default='saved_npz/dcdm')
    parser.add_argument('--save_name', '--sn', type=str)
    parser.add_argument('--feature_for_umap', '--ffu', type=str)
    args = parser.parse_args()

    main(args)

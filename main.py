import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import pdb
import random
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, wandb_init_project, base_settings


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DM/MTT')
    parser.add_argument('--domain_discriminator', action='store_true')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset/PACS_by_domain/PACS')
    parser.add_argument('--da_source', type=str, default=None)
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--syn_data_path', type=str, default='None')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--trg_domain', type=str, default='None', help='target domain')
    parser.add_argument('--num_eval_sep', type=int, default=0, help='Used when running the evaluationa separately.')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')

    # --- # Only Train / Eval Setting # --- #
    parser.add_argument('--train_only', action='store_true', help='train only')
    parser.add_argument('--eval_only', action='store_true', help='eval only')

    # --- # Coreset Setting # --- #
    parser.add_argument('--coreset', action='store_true', help='turn on when using coreset selection method or training whole dataset')
    parser.add_argument('--coreset_method', type=str, default=None, help='whole/random/k-center/herding')
    
    # --- # Wandb Setting # --- #
    parser.add_argument('--wandb', action='store_true', help='use wandb')
    parser.add_argument('--wandb_project_name', type=str, default='MD3', help='wandb project name')
    parser.add_argument('--wandb_name', type=str, default='None')
    

    args = parser.parse_args()
    if not args.coreset:
        args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    if args.wandb:
        run = wandb_init_project(args)
    _iteration, _eval_it_jump = base_settings(args.method)

    for exp in range(args.num_exp):
        os.makedirs(f'{args.save_path}/{exp}', exist_ok=True)
        _arg_log = open(os.path.join(f'{args.save_path}/{exp}', 'args.txt'), 'w')
        _arg_log.write(str(args.__dict__))
        _arg_log.close()
    
        _log = open(os.path.join(f'{args.save_path}/{exp}', f'log_{exp}.txt'), 'a+')

        eval_it_pool = np.arange(0, _iteration+1, _eval_it_jump).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [_iteration]
        print('eval_it_pool: ', eval_it_pool)
        channel, im_size, num_classes, num_domains, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(0, args.dataset, args.syn_data_path, args.data_path, args.trg_domain, args.da_source, exp)
        model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


        accs_all_exps = dict() # record performances of all experiments
        for key in model_eval_pool:
            accs_all_exps[key] = []

        data_save = []

        it = 0

        # it_eval = args.num_eval_sep
        # _log.write('Evaluation iteration: %d\n'%it_eval)

        print(f'\n================== Exp {exp} Save PTH {args.save_path}_{exp}  ==================\n ')
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n, domain=False, temp_domain_lbl=-1): # get random n images from class c
            if domain:
                assert temp_domain_lbl != -1, "domain_lbl cannot be -1"
                _temp_domain = []
                for class_idx in range(num_classes):
                    _temp_domain.extend(indices_class[class_idx][temp_domain_lbl])
                idx_shuffle=np.random.permutation(_temp_domain[:n])
                return images_all[idx_shuffle], torch.tensor(torch.ones(images_all[idx_shuffle].shape[0]) * temp_domain_lbl, dtype=torch.long, requires_grad=False, device=args.device)
            else:
                try:
                    idx_shuffle = np.random.permutation(np.concatenate(indices_class[c]))[:n]
                except:
                    idx_shuffle = np.random.permutation(indices_class[c])[:n]
                return images_all[idx_shuffle], None

                for ch in range(channel):
                    print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))
        
        for model_eval in model_eval_pool:

            ''' organize the real dataset '''
            train_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_real, shuffle=True, num_workers=20)
            test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.batch_real, shuffle=False, num_workers=20)

            ''' Evaluate synthetic data '''
            if 'DM' not in args.syn_data_path:
                if args.dsa:
                    args.epoch_eval_train = 1000
                    args.dc_aug_param = None
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                if args.dsa or args.dc_aug_param['strategy'] != 'none':
                    args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                else:
                    args.epoch_eval_train = 300

            accs = []
            net_eval = get_network(model_eval, channel, num_classes, im_size, args.domain_discriminator).to(args.device) # get a random model
            for it_eval in range(args.num_eval):
                if args.coreset:
                    image_coreset = torch.zeros(num_classes*args.ipc, channel, im_size[0], im_size[1]).to(args.device)
                    label_coreset = torch.zeros(num_classes*args.ipc, dtype=torch.long, requires_grad=False, device=args.device)
                    for _num_c in range(num_classes):
                        label_coreset[_num_c*args.ipc:(_num_c+1)*args.ipc] = _num_c
                    if args.coreset_method == 'k-center':
                        for c in range(num_classes):
                            print("Generating synthetic data for class {}".format(c))
                            imgs = images_all[indices_class[c]]
                            features = net_eval.embed(imgs)
                            mean = torch.mean(features, dim=0, keepdim=True)
                            dis = torch.norm(features - mean, dim=1)
                            rank = torch.argsort(dis) 
                            idx_centers = rank[:1].tolist() 
                            for i in range(args.ipc-1):
                                feature_centers = features[idx_centers]
                                if feature_centers.shape[0] == features.shape[1]:
                                    feature_centers = feature_centers.unsqueeze(0)
                                dis_center = torch.cdist(features, feature_centers)
                                dis_min, _ = torch.min(dis_center, dim=-1)
                                id_max = torch.argmax(dis_min).item()
                                idx_centers.append(id_max)
                            image_coreset[c*args.ipc:(c+1)*args.ipc] = imgs[idx_centers]
                        print("Synthetic data generated by k-center")
                        _, acc_train, acc_test = evaluate_synset(it, it_eval, net_eval, image_coreset, label_coreset, testloader, args, logs=_log)
                    elif args.coreset_method == 'herding':
                        for c in range(num_classes):
                            print("Generating synthetic data for class {}".format(c))
                            imgs = images_all[indices_class[c]]
                            features = net_eval.embed(imgs)
                            mean = torch.mean(features, dim=0, keepdim=True)
                            idx_selected = []
                            idx_left = np.arange(features.shape[0]).tolist()

                            for i in range(args.ipc):
                                if len(idx_selected) > 0:
                                    det = mean*(i+1) - torch.sum(features[idx_selected], dim=0)
                                else:
                                    det = mean*(i+1)
                                dis = torch.norm(det-features[idx_left], dim=1)
                                idx = torch.argmin(dis).item()
                                idx_selected.append(idx_left[idx])
                                del idx_left[idx]
                            image_coreset[c*args.ipc:(c+1)*args.ipc] = imgs[idx_selected]
                        _, acc_train, acc_test = evaluate_synset(it, it_eval, net_eval, image_coreset, label_coreset, testloader, args, logs=_log)
                    elif args.coreset_method == 'random':
                        for c in range(num_classes):
                            print("Generating synthetic data for class {}".format(c))
                            image_coreset[c*args.ipc:(c+1)*args.ipc] = images_all[random.sample(indices_class[c], args.ipc)]
                        _, acc_train, acc_test = evaluate_synset(it, it_eval, net_eval, image_coreset, label_coreset, testloader, args, logs=_log)
                    else: # ignored when whole dataset is used
                        _, acc_train, acc_test = evaluate_synset(it, it_eval, net_eval, None, None, testloader, args, logs=_log, trainloader=train_loader)
                else:
                    _, acc_train, acc_test = evaluate_synset(it, it_eval, net_eval, None, None, testloader, args, logs=_log, trainloader=train_loader)
                accs.append(acc_test)
                # _log.write(accs +'\n')
        _log.write('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
        _log.close()


if __name__ == '__main__':
    main()
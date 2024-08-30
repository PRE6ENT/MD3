import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import pdb
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--domain_discriminator', action='store_true')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset/PACS_by_domain/PACS')
    parser.add_argument('--da_source', type=str, default=None)
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
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
    parser.add_argument('--num_eval_sep', type=int, help='Used when running the evaluationa separately.')
    parser.add_argument('--wandb', action='store_true')

    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    for exp in range(args.num_exp):
        os.makedirs(f'{args.save_path}_{exp}', exist_ok=True)
        _arg_log = open(os.path.join(f'{args.save_path}_{exp}', 'args.txt'), 'w')
        _arg_log.write(str(args.__dict__))
        _arg_log.close()
    
        _log = open(os.path.join(f'{args.save_path}_{exp}', f'log_{exp}.txt'), 'a+')

        eval_it_pool = np.arange(0, args.Iteration+1, 500).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
        print('eval_it_pool: ', eval_it_pool)
        channel, im_size, num_classes, num_domains, class_names, mean, std, dst_train, dst_test, dst_domain, testloader, domainloader = get_dataset(args.dataset, args.syn_data_path, args.data_path, args.trg_domain, args.da_source, exp)
        model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


        accs_all_exps = dict() # record performances of all experiments
        for key in model_eval_pool:
            accs_all_exps[key] = []

        data_save = []

        it = 0

        it_eval = args.num_eval_sep
        _log.write('Evaluation iteration: %d\n'%it_eval)

        print(f'\n================== Exp {exp} Save PTH {args.save_path}_{exp}  ==================\n ')
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)
        
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

            # accs = []
            # for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size, args.domain_discriminator).to(args.device) # get a random model
            _, acc_train, acc_test = evaluate_synset(it, it_eval, net_eval, None, None, testloader, args, logs=_log, trainloader=train_loader)
        _log.close()
                # accs.append(acc_test)
            # _log.write(accs +'\n')
            # _log.write('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
            # print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))


if __name__ == '__main__':
    main()
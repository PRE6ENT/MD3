''' 
TODO
[Done] Whole Dataset Training + Evaluation
Why was the first network working with [{class_num}, 8192] when the second network was working with [{class_num}, 2048]?
Why was the evaulation not coming out the same all the time? (The two values differ too much)
[Done] 64 x 64 Images work on model with depth 4
Coreset Training + Evaluation
DM Training + Evaluation 
MD3 Training + Evaluation

확인해볼 것
[Done] Whole Dataset의 setting (train iteration 등등) -> 이때까지 비슷하게 나왔으니깐 같은 셋팅이라고 생각됨.
For Coreset (Herding , K-Center) the model trained on the whole dataset is used to extract features.
'''

import argparse
import copy
import os
import pdb
import numpy as np
import time
import torch
import torch.nn as nn

from torchvision.utils import save_image
from utils import *


def main():
    # - Arguments
    parser = argparse.ArgumentParser()
    # - Fixed (usually) settings
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments') #TODO Change it to 5 after finalizing the work
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.') # All the experiments are conducted with noise for MD3!
    parser.add_argument('--freeze_seed', type=bool, default=True, help='freeze random seed for reproducibility') #TODO Disable after finalizing the work

    # - General settings
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--method', type=str, help='GM/DM/MTT/MD3')
    parser.add_argument('--ipc', type=int, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters') # This is used when training the evaluation models with synthetic data
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, help='dataset path')
    parser.add_argument('--save_path', type=str, help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    # - Whole / Coreset settings
    parser.add_argument('--coreset', action='store_true')
    parser.add_argument('--coreset_type', type=str, help='whole/random/k-center/herding')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.outer_loop, args.inner_loop = get_loops(args.ipc) if not args.coreset else (None, None)
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True


    #TODO 현재는 모든 것을 고정하고 실험을 진행할 수 있도록 모든 시드를 고정하는 것을 코딩하기 (이게 혹시 모든 eval model들이 같은 결과를 내도록 할 수도 있을까?)
    # - Freeze random seed
    if args.freeze_seed:
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    # - Check for essential arguments
    assert args.data_path is not None, 'Please specify the path to the dataset'
    assert args.save_path is not None, 'Please specify the path to save the results'

    eval_it_pool = np.arange(0, args.Iteration+1, 2000).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    start_time = time.time()
    channel, im_size, num_classes, class_names, num_domains, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    print('line 81: ', time.time()-start_time)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    accs_all_exps = dict() # record performances of all experiments (dictionary utilized mainly for multiple models)
    for key in model_eval_pool:
        accs_all_exps[key] = []
    data_save = []


    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)


        # - Make directories
        os.makedirs(args.data_path, exist_ok=True) # For the data that gets downloaded during training
        os.makedirs(f'{args.save_path}/{exp}', exist_ok=True)


        # - Get real data (modified to save time)
        start_time = time.time()
        if os.path.exists(f'saved_data/{args.dataset}_images_all.pt') and os.path.exists(f'saved_data/{args.dataset}_labels_all.pt') and os.path.exists(f'saved_data/{args.dataset}_indices_class.pt'):
            images_all = torch.load(f'saved_data/{args.dataset}_images_all.pt')
            labels_all = torch.load(f'saved_data/{args.dataset}_labels_all.pt')
            indices_class = torch.load(f'saved_data/{args.dataset}_indices_class.pt')
        else:
            images_all = []
            labels_all = []
            indices_class = [[] for c in range(num_classes)]
            images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
            labels_all = [dst_train[i][1] for i in range(len(dst_train))]
            for i, lab in enumerate(labels_all):
                indices_class[lab].append(i)
            images_all = torch.cat(images_all, dim=0).to(args.device)
            labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
            os.makedirs('saved_data', exist_ok=True)
            torch.save(images_all, f'saved_data/{args.dataset}_images_all.pt')
            torch.save(labels_all, f'saved_data/{args.dataset}_labels_all.pt')
            torch.save(indices_class, f'saved_data/{args.dataset}_indices_class.pt')
        print('line 97-114: ', time.time()-start_time)

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))
        ### Does not need this code after checking the Mean and STD of the dataset
        # for ch in range(channel):
        #     print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

        def get_images(c, n): 
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]


        # - Train for whole/coreset
        if args.coreset:
            if args.coreset_type == 'random':
                pass

            elif args.coreset_type == 'k-center':
                pass

            elif args.coreset_type == 'herding':
                pass

            else: # whole dataset
                image_for_train = images_all
                label_for_train = labels_all
            
            accs = []
            args.log_file = open(f'{args.save_path}/{exp}/log_{"_".join(list(map(str, time.localtime()[0:-4])))}.txt', 'w+')
            for it_eval in range(args.num_eval):
                net_eval = get_network(model_eval_pool[0], channel, num_classes, im_size).to(args.device) 
                _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_for_train, label_for_train, testloader, args)
                accs.append(acc_test)
            print(f'\n\n# - {args.coreset_type} {args.dataset}\n')
            print(f'evaluate {len(accs)} random {model_eval_pool[0]}, mean = {np.mean(accs)*100:.4f} std = {np.std(accs)*100:.4f}\n\n')
            args.log_file.write(f'\n\n# - {args.coreset_type} {args.dataset}\n')
            args.log_file.write(f'evaluate {len(accs)} random {model_eval_pool[0]}, mean = {np.mean(accs)*100:.4f} std = {np.std(accs)*100:.4f}\n\n')

        # - Train for general dataset condensation
        else:
            # - Initialize synthetic data
            image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
            label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)
            ### Currently I am only testing with args.init == 'noise'
            # if args.init == 'real':
            #     print('initialize synthetic data from random real images')
            #     for c in range(num_classes):
            #         image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
            # else:
            #     print('initialize synthetic data from random noise')


            # - Training
            optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) #TODO Check how the momentum value is selected
            optimizer_img.zero_grad()
            print('%s training begins'%get_time())

            for it in range(args.Iteration+1):
                # - Evaluation on currently made synthetic data
                if it in eval_it_pool:
                    for model_eval in model_eval_pool:
                        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                        ### Not as important to pring these rn
                        # print('DSA augmentation strategy: \n', args.dsa_strategy)
                        # print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                        accs = []
                        for it_eval in range(args.num_eval):
                            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) 
                            image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
                            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                            accs.append(acc_test)
                        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                        if it == args.Iteration:
                            accs_all_exps[model_eval] += accs


                    # - Save synthetic images
                    image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                    for ch in range(channel):
                        image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                    image_syn_vis[image_syn_vis<0] = 0.0
                    image_syn_vis[image_syn_vis>1] = 1.0
                    for sample_idx in range(image_syn_vis.shape[0]):#TODO Check the shape -> For the saving process sample_idx // args.ipc?
                        os.makedirs(f'{args.save_path}/exp{exp}/iter{it}/{class_names[sample_idx // args.ipc]}', exist_ok=True) #TODO Check the path
                        save_image(image_syn_vis[sample_idx], f'{args.save_path}/exp{exp}/iter{it}/{class_names[sample_idx // args.ipc]}/sample{sample_idx}_{sample_idx}.png')


                # - Continue training synthetic data 
                net = get_network(args.model, channel, num_classes, im_size).to(args.device)
                net.train()
                for param in list(net.parameters()):
                    param.requires_grad = False
                embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel
                loss_avg = 0


                # - Update synthetic data
                ### Currenlty only training/testing with ConvNet
                # if 'BN' not in args.model: # for ConvNet
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)

                    loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2) #TODO Check the shape of the output


                ### Currently only training/testing with ConvNet
                # else: # for ConvNetBN
                #     images_real_all = []
                #     images_syn_all = []
                #     loss = torch.tensor(0.0).to(args.device)
                #     for c in range(num_classes):
                #         img_real = get_images(c, args.batch_real)
                #         img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                #         if args.dsa:
                #             seed = int(time.time() * 1000) % 100000
                #             img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                #             img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                #         images_real_all.append(img_real)
                #         images_syn_all.append(img_syn)

                #     images_real_all = torch.cat(images_real_all, dim=0)
                #     images_syn_all = torch.cat(images_syn_all, dim=0)

                #     output_real = embed(images_real_all).detach()
                #     output_syn = embed(images_syn_all)

                #     loss += torch.sum((torch.mean(output_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()
                loss_avg /= (num_classes)
                if it%10 == 0:
                    print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))
                if it == args.Iteration: #TODO Save more often in case the GPU crashes
                    data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                    torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))

            print('\n==================== Final Results ====================\n')
            for key in model_eval_pool:
                accs = accs_all_exps[key]
                print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))


if __name__ == '__main__':
    main()


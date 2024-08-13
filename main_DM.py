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
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--trg_domain', type=str, default='None', help='target domain')
    parser.add_argument('--domain_info', action='store_true', help='use domain information')
    parser.add_argument('--wandb', action='store_true', help='use wandb')
    parser.add_argument('--wandb_name', type=str, default='None')
    parser.add_argument('--weight_class_classifier', type=float, default=1.0, help='weight for class classifier')
    parser.add_argument('--weight_domain_classifier', type=float, default=1.0, help='weight for domain classifier')
    parser.add_argument('--continual', action='store_true', help='continual learning')
    parser.add_argument('--num_workers', type=int, defualt=0)


    args = parser.parse_args()
    args.method = 'DM'
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    if args.wandb:
        import wandb
        assert args.wandb_name != 'None', 'Please specify the wandb_name'
        wandb.login()
        run = wandb.init(
            project="Domain Condensation Domain Test!",
            name=args.wandb_name,
            config={
                "ipc": args.ipc,
                "learning_rate": args.lr_img,
                "weight_class_classifier": args.weight_class_classifier,
                "weight_domain_classifier": args.weight_domain_classifier,
            },
        )

    os.makedirs(args.save_path, exist_ok=True)

    _log = open(os.path.join(args.save_path, 'log.txt'), 'w')

    eval_it_pool = np.arange(0, args.Iteration+1, 2000).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    # eval_it_pool = [args.Iteration] if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    # channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, None, args.data_path, args.trg_domain)
    channel, im_size, num_classes, num_domains, class_names, mean, std, dst_train, dst_test, dst_domain, testloader, domainloader = get_dataset(args.num_workers, args.dataset, None, args.data_path, args.trg_domain)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        if args.domain_info:
            indices_class = [[[] for _ in range(num_domains)] for _ in range(num_classes)]
        else:
            indices_class = [[] for _ in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        if args.domain_info:
            domain_labels_all = [dst_train[i][2] for i in range(len(dst_train))]
            for i, (lab, domain_lab) in enumerate(zip(labels_all, domain_labels_all)):
                indices_class[lab][domain_lab].append(i)
        else:
            for i, lab in enumerate(labels_all):
                indices_class[lab].append(i)

        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        for c in range(num_classes):
            try:
                print('class c = %d: %d real images'%(c, len(np.concatenate(indices_class[c]))))
            except:
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

        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc], temp_domain_label_real = get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')

        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        print('%s training begins'%get_time())

        for it in range(args.Iteration+1):
            
            if args.continual:
                args.weight_class_classifier = min(1.0, 0.5 + (it / args.Iteration))
                args.weight_domain_classifier = max(0.0, 0.5 - (it / args.Iteration))

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                os.makedirs(os.path.join(args.save_path, str(it)), exist_ok=True)
                _log_temp = open(os.path.join(args.save_path, str(it), 'log.txt'), 'a+')
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

                    accs = []
                    for it_eval in range(args.num_eval):
                        _log.write('Evaluation iteration: %d\n'%it_eval)
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        _, acc_train, acc_test = evaluate_synset(it, it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, logs=_log_temp)
                        accs.append(acc_test)
                    if args.wandb:
                        run.log({f"accuracy": np.mean(accs)})
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs
                _log_temp.close()

                ''' visualize and save '''
                # save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                for sample_idx in range(image_syn_vis.shape[0]):
                    os.makedirs(f'{args.save_path}/exp{exp}/iter{it}/{class_names[sample_idx // args.ipc]}', exist_ok=True)
                    save_image(image_syn_vis[sample_idx], f'{args.save_path}/exp{exp}/iter{it}/{class_names[sample_idx // args.ipc]}/sample{sample_idx}_{sample_idx}.png')
                # save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.

            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net = torch.nn.DataParallel(net)
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False
            if args.domain_info:
                domain_net = get_network(args.model, channel, num_classes, im_size, args.domain_info).to(args.device) # get a random model
                domain_net = torch.nn.DataParallel(domain_net)
                domain_net.train()
                for param in list(domain_net.parameters()):
                    param.requires_grad = False

            if args.model == 'ResNet18P':
                assert False, 'Not implemented yet!'
                embed = torch.nn.Sequential(*list(net_eval.children())[:-1])
            else:
                embed = net.module.embed
                if args.domain_info:
                    embed_domain = domain_net.module.embed
                # embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel
                # embd_domain = domain_net.module.embed if torch.cuda.device_count() > 1 else domain_net.embed # for GPU parallel

            loss_avg = 0
            domain_avg = 0

            ''' update synthetic data '''
            if 'BN' not in args.model: # for ConvNet
                # --- update using class information
                loss = torch.tensor(0.0).to(args.device)
                domain_loss = torch.tensor(0.0).to(args.device)

                for c in range(num_classes):
                    img_real, _ = get_images(c, args.batch_real)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)

                    loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
                loss *= args.weight_class_classifier

                # --- update using domain information
                if args.domain_info:
                    cropped_image_syn = [image_syn[:, :, 0:32, 0:32], image_syn[:, :, 0:32, 32:64], image_syn[:, :, 32:64, 0:32], image_syn[:, :, 32:64, 32:64]]
                    for _domain in range(num_domains):
                        img_real, _ = get_images(c, args.batch_real, domain=True, temp_domain_lbl=_domain)
                        domain_img_syn = cropped_image_syn[_domain]

                        if args.dsa:
                            seed = int(time.time() * 1000) % 100000
                            img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                            domain_img_syn = DiffAugment(domain_img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        
                        output_real = embed_domain(img_real).detach()
                        output_syn = embed_domain(domain_img_syn)

                        domain_loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
                    domain_loss *= args.weight_domain_classifier
                if args.wandb:
                    wandb.log({'class_loss': loss.item(), 'domain_loss': domain_loss.item()})
                loss += domain_loss

            else: # for ConvNetBN  --> 현재는 고려하고 있지 않음!
                assert False, 'Not implemented yet!'
                images_real_all = []
                images_syn_all = []
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    images_real_all.append(img_real)
                    images_syn_all.append(img_syn)

                images_real_all = torch.cat(images_real_all, dim=0)
                images_syn_all = torch.cat(images_syn_all, dim=0)

                output_real = embed(images_real_all).detach()
                output_syn = embed(images_syn_all)

                loss += torch.sum((torch.mean(output_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()

            loss_avg /= (num_classes)
            domain_avg += domain_loss.item()
            domain_avg /= 4

            if it%10 == 0:
                print('%s iter = %05d, loss = %.4f domain loss = %.4f' % (get_time(), it, loss_avg, domain_avg))

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))

    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))
        _log.write('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%\n'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))

    _log.close()



if __name__ == '__main__':
    main()



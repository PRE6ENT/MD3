import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import pdb
import gc
from glob import glob
from tqdm import tqdm
# from memory_profiler import profile
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, CustomDatasetFolder

torch.autograd.set_detect_anomaly(True)

# if hits and time for each line "kernprof -l -v main.py"
# if cpu memory consumption for each line "mprof run main.py"

# @profile
def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
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
    parser.add_argument('--lr_domain_net', type=float, default=0.001, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--trg_domain', type=str, default='None', help='target domain')
    parser.add_argument('--domain_info', action='store_true')
    parser.add_argument('--domain_test', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--weight_class_classifier', type=float, default=1.0)
    parser.add_argument('--weight_domain_classifier', type=float, default=1.0)
    parser.add_argument('--wandb_name', type=str, default='None')
    parser.add_argument('--continual', action='store_true')
    parser.add_argument('--profiling', action='store_true')
    parser.add_argument('--only_training', action='store_true')
    parser.add_argument('--only_evaluation', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--syn_data_path', type=str, default='None')

    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    if args.profiling:
        args.continual = True
        args.domain_info = True
        args.num_exp = 1
        args.batch_real = 64
        args.batch_train = 64
        args.dataset = 'PACS_in_dist'
        args.model = 'ConvNet'
        args.ipc = 1
        args.save_path = 'results/test'
        args.data_path = '/local_data/PACS/'
        args.lr_domain_net = 0.1
        args.init = 'noise'
        args.Iteration = 100


    if args.wandb:
        import wandb
        assert args.wandb_name != 'None', 'Please specify the wandb name.'
        args.wandb_name += 'only_train' if args.only_training else 'only_eval' if args.only_evaluation else 'train_eval'
        wandb.login()
        run = wandb.init(
            project="Domain Condensation Domain Test!",
            name=args.wandb_name,
            config=args,
        )

    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    _log = open(os.path.join(args.save_path, 'log.txt'), 'w')

    if args.only_evaluation:
        for exp in range(args.num_exp):
            eval_it_pool = np.arange(0, args.Iteration+1, 500).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
            # eval_it_pool = [args.Iteration] if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
            print('eval_it_pool: ', eval_it_pool)
            channel, im_size, num_classes, num_domains, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(0, args.dataset, args.syn_data_path, args.data_path, args.trg_domain, args.da_source, exp)
            model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

            accs_all_exps = dict() # record performances of all experiments
            for key in model_eval_pool:
                accs_all_exps[key] = []

            data_save = []
            for it in eval_it_pool:
                while not os.path.exists(f'{args.save_path}/exp{exp}/iter{it}'):
                    print(f"# ----- Waiting For the Training to be Done! Waiting For the PTH {args.save_path}/exp{exp}/iter{it}----- #")
                    time.sleep(60)
                while len(glob(f'{args.save_path}/exp{exp}/iter{it}/*/*.png')) != num_classes * args.ipc:
                    print(f"# ----- Waiting For the Full Synthetic Images to be Saved! {len(glob(f'{args.save_path}/exp{exp}/iter{it}/*/*.png'))} {num_classes * args.ipc}----- #")
                    time.sleep(30)
                _syn_dataset = CustomDatasetFolder(root=f'{args.save_path}/exp{exp}/iter{it}/', transform=torchvision.transforms.ToTensor(), dataset=args.dataset)
                _syn_loader = torch.utils.data.DataLoader(_syn_dataset, batch_size=64, shuffle=True, num_workers=args.num_workers)

                for model_eval in model_eval_pool:
                    # os.makedirs(os.path.join(args.save_path, str(it)), exist_ok=True)
                    # _log_temp = open(os.path.join(args.save_path, str(it), 'log.txt'), 'a+')
                    # _log_temp.write('Evaluation iteration: %d\n'%it)
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
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
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        # _, acc_train, acc_test = evaluate_synset(it, it_eval, net_eval, None, None, testloader, args, logs=_log_temp, trainloader=_syn_loader)
                        _, acc_train, acc_test = evaluate_synset(it, it_eval, net_eval, None, None, testloader, args, trainloader=_syn_loader)
                        accs.append(acc_test)
                    if args.wandb:
                        run.log({f"accuracy": np.mean(accs)})
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

                #     if it == args.Iteration: # record the final results
                #         accs_all_exps[model_eval] += accs
                # _log_temp.close()
        

    else:

        for exp in range(args.num_exp):

            eval_it_pool = np.arange(0, args.Iteration+1, 500).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
            # eval_it_pool = [args.Iteration] if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
            print('eval_it_pool: ', eval_it_pool)
            channel, im_size, num_classes, num_domains, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(0, args.dataset, args.syn_data_path, args.data_path, args.trg_domain, args.da_source, exp)
            model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

            accs_all_exps = dict() # record performances of all experiments
            for key in model_eval_pool:
                accs_all_exps[key] = []

            data_save = []

            print('\n================== Exp %d ==================\n '%exp)
            print('Hyper-parameters: \n', args.__dict__)
            print('Evaluation model pool: ', model_eval_pool)

            ''' organize the real dataset '''
            images_all_cpu = []
            labels_all_cpu = []
            if args.domain_info:
                indices_class = [[[] for _ in range(num_domains)] for _ in range(num_classes)]
            else:
                indices_class = [[] for _ in range(num_classes)]

            images_all_cpu = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
            labels_all_cpu = [dst_train[i][1] for i in range(len(dst_train))]
            if args.domain_info:
                domain_labels_all = [dst_train[i][2] for i in range(len(dst_train))]
                for i, (lab, domain_lab) in enumerate(zip(labels_all_cpu, domain_labels_all)):
                    indices_class[lab][domain_lab].append(i)
            else:
                for i, lab in enumerate(labels_all):
                    indices_class[lab].append(i)
            
            images_all = torch.cat(images_all_cpu, dim=0).to(args.device)
            labels_all = torch.tensor(labels_all_cpu, dtype=torch.long, device=args.device)

            dst_train.clear_memory()
            del images_all_cpu, labels_all_cpu
            gc.collect()

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
            domain_label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)] * num_classes, dtype=torch.long, requires_grad=False, device=args.device).view(-1) 

            if args.init == 'real':
                print('initialize synthetic data from random real images')
                for c in range(num_classes):
                    image_syn.data[c*args.ipc:(c+1)*args.ipc], temp_domain_label_real = get_images(c, args.ipc).detach().data
            else:
                print('initialize synthetic data from random noise')


            ''' training '''
            optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
            optimizer_img.zero_grad()
            criterion = nn.CrossEntropyLoss().to(args.device)
            print('%s training begins'%get_time())

            for it in range(args.Iteration+1):
                if args.continual:
                    args.weight_class_classifier = min(1.0, 0.5 + (it / args.Iteration))
                    args.weight_domain_classifier = max(0.0, 0.5 - (it / args.Iteration))
                ''' Evaluate synthetic data '''
                if not args.domain_test:
                    if it in eval_it_pool:
                        if not args.only_training:
                            for model_eval in model_eval_pool:
                                os.makedirs(os.path.join(args.save_path, str(it)), exist_ok=True)
                                _log_temp = open(os.path.join(args.save_path, str(it), 'log.txt'), 'a+')
                                _log_temp.write('Evaluation iteration: %d\n'%it)
                                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
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
                                for it_eval in range(args.num_eval):
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
                        for sample_idx in tqdm(range(image_syn_vis.shape[0]), desc='Saving Images'):
                            os.makedirs(f'{args.save_path}/exp{exp}/iter{it}/{class_names[sample_idx // args.ipc]}', exist_ok=True)
                            save_image(image_syn_vis[sample_idx], f'{args.save_path}/exp{exp}/iter{it}/{class_names[sample_idx // args.ipc]}/sample{sample_idx}_{sample_idx}.png')
                        # save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.


                ''' Train synthetic data '''
                net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
                net = torch.nn.DataParallel(net)
                net.train()
                net_parameters = list(net.parameters())
                optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
                optimizer_net.zero_grad()
                if args.domain_info:
                    domain_net = get_network(args.model, channel, num_domains, im_size, args.domain_info).to(args.device) # get a random model
                    domain_net = torch.nn.DataParallel(domain_net)
                    domain_net.train()
                    domain_net_parameters = list(domain_net.parameters())
                    optimizer_domain_net = torch.optim.SGD(domain_net.parameters(), lr=args.lr_domain_net)  # optimizer_img for synthetic data
                    optimizer_domain_net.zero_grad()
                loss_avg = 0
                domain_loss_avg = 0
                args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.


                for ol in range(args.outer_loop):

                    ''' freeze the running mu and sigma for BatchNorm layers '''
                    # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                    # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                    # This would make the training with BatchNorm layers easier.

                    BN_flag = False
                    BNSizePC = 16  # for batch normalization
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name(): #BatchNorm
                            BN_flag = True
                    if BN_flag:
                        img_real, _ = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                        net.train() # for updating the mu, sigma of BatchNorm
                        output_real, _ = net(img_real) # get running mu, sigma
                        for module in net.modules():
                            if 'BatchNorm' in module._get_name():  #BatchNorm
                                module.eval() # fix mu and sigma of every BatchNorm layer
                    # BN_flag = False
                    # if args.domain_info:
                    #     for module in domain_net.modules():
                    #         if 'BatchNorm' in module._get_name(): #BatchNorm
                    #             BN_flag = True
                    #     if BN_flag:
                    #         img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=1)
                    #         domain_net.train() # for updating the mu, sigma of BatchNorm
                    #         output_real, _ = domain_net(img_real) # get running mu, sigma
                    #         for module in domain_net.modules():
                    #             if 'BatchNorm' in module._get_name():  #BatchNorm
                    #                 module.eval() # fix mu and sigma of every BatchNorm layer


                    ''' update synthetic data '''
                    net.to(args.device)
                    if args.domain_info:
                        domain_net.to(args.device)
                    loss = torch.tensor(0.0).to(args.device)
                    domain_loss = torch.tensor(0.0).to(args.device)
                    if not args.domain_test:
                        for c in range(num_classes):
                            img_real, _ = get_images(c, args.batch_real)
                            lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                            img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                            lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c
                            if args.dsa:
                                seed = int(time.time() * 1000) % 100000
                                img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                                img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                            output_real, _ = net(img_real)
                            loss_real = criterion(output_real, lab_real)
                            gw_real = torch.autograd.grad(loss_real, net_parameters, allow_unused=True)
                            gw_real = list((_.detach().clone() for _ in gw_real if _ is not None))

                            output_syn, _ = net(img_syn)
                            if args.wandb:
                                run.log({f"class syn train accuracy {c}": output_syn.argmax(dim=1).eq(lab_syn).sum().item() / lab_syn.shape[0]})
                            loss_syn = criterion(output_syn, lab_syn)
                            gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True, allow_unused=True)

                            loss += match_loss(gw_syn, gw_real, args) * args.weight_class_classifier
                        loss_avg += loss.item()
                    
                    if args.domain_info:
                        cropped_image_syn = [image_syn[:, :, 0:32, 0:32], image_syn[:, :, 0:32, 32:64], image_syn[:, :, 32:64, 0:32], image_syn[:, :, 32:64, 32:64]]
                        for _domain in range(num_domains):
                            img_real, temp_domain_label_real = get_images(c, args.batch_real, domain=True, temp_domain_lbl=_domain)
                            domain_lab_syn = torch.tensor([np.ones(args.ipc)*_domain for i in range(num_classes)], dtype=torch.long, device=args.device).view(-1)
                            domain_img_syn = cropped_image_syn[_domain]

                            output_domain_real, _ = domain_net(img_real)
                            loss_domain_real = criterion(output_domain_real, temp_domain_label_real)
                            gw_domain_real = torch.autograd.grad(loss_domain_real, domain_net_parameters, allow_unused=True)
                            gw_domain_real = list((_.detach().clone() for _ in gw_domain_real if _ is not None))

                            output_domain_syn, _ = domain_net(domain_img_syn)
                            if args.wandb:
                                run.log({f"domain syn train accuracy {_domain}": output_domain_syn.argmax(dim=1).eq(domain_lab_syn).sum().item() / domain_lab_syn.shape[0]})
                            loss_domain_syn = criterion(output_domain_syn, domain_lab_syn)
                            gw_domain_syn = torch.autograd.grad(loss_domain_syn, domain_net_parameters, create_graph=True, allow_unused=True)
                            domain_loss += match_loss(gw_domain_syn, gw_domain_real, args) * args.weight_domain_classifier
                        domain_loss_avg += domain_loss.item()

                    total_loss = loss if not args.domain_info else loss + domain_loss
                    optimizer_img.zero_grad()
                    total_loss.backward()
                    optimizer_img.step()

                    if ol == args.outer_loop - 1:
                        break

                    ''' update network '''
                    image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                    dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                    trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=args.num_workers)
                    if args.domain_info:
                        cropped_image_syn_train = copy.deepcopy(torch.cat(cropped_image_syn).detach()) 
                        domain_label_syn_train = []
                        for i in range(num_domains):
                            domain_label_syn_train.extend([i] * num_classes * args.ipc)
                        domain_label_syn_train = torch.tensor(domain_label_syn_train, dtype=torch.long, device=args.device).detach()
                        dst_syn_train = TensorDataset(cropped_image_syn_train, domain_label_syn_train)
                        domainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=args.num_workers)
                    for il in range(args.inner_loop):
                        if args.domain_info:
                            epoch('train', trainloader, domainloader, net, optimizer_net, criterion, args, aug = True if args.dsa else False, run=run if args.wandb else None, domain_net=domain_net, optimizer_domain_net=optimizer_domain_net)
                        else:
                            epoch('train', trainloader, domainloader, net, optimizer_net, criterion, args, aug = True if args.dsa else False, run=run if args.wandb else None)


                loss_avg /= (num_classes*args.outer_loop)
                if args.wandb:
                    run.log({f"class loss": loss_avg})
                if args.domain_info:
                    domain_loss_avg /= (num_classes*args.outer_loop)
                    if args.wandb:
                        run.log({f"domain loss": domain_loss_avg})

                if it%10 == 0:
                    if args.domain_info:
                        print('%s iter = %04d, loss = %.4f, domain loss = %.4f' % (get_time(), it, loss_avg, domain_loss_avg))
                        _log.write('%s iter = %04d, loss = %.4f, domain loss = %.4f\n' % (get_time(), it, loss_avg, domain_loss_avg))
                    else:
                        print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))
                        _log.write('%s iter = %04d, loss = %.4f\n' % (get_time(), it, loss_avg))

                if it == args.Iteration: # only record the final results
                    data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                    torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))


            print('\n==================== Final Results ====================\n')
            for key in model_eval_pool:
                accs = accs_all_exps[key]
                print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))
                _log.write('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%\n'%(exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))

            _log.close()


if __name__ == '__main__':
    main()
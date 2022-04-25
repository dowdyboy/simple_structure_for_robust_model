"""
Created on Sun Oct 25 2018
@author: Kimin Lee
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import numpy as np
import models
import os
import lib.adversary as adversary
import accelerate
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image

from dowdyboy_lib.log import logging_conf, log
from dowdyboy_lib.rand import set_seed
from dowdyboy_lib.model_util import frozen_module, unfrozen_module


parser = argparse.ArgumentParser(description='PyTorch code to create adv samples')
parser.add_argument('--batch-size', type=int, default=8, metavar='N', help='batch size for data loader')
parser.add_argument('--data-dir', required=True, help='image folder path to dataset')
parser.add_argument('--out-dir', required=True, help='folder to output results')
parser.add_argument('--log-file', required=True, help='give a file path to log')
parser.add_argument('--num-classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net-depth', type=int, default=34, help='resnet depth')
parser.add_argument('--adv-type', type=str, required=True, help='FGSM | BIM | DeepFool | CWL2 | Noise | NST')
parser.add_argument('--adv-noise', type=float, default=0.1, help='adv noise value')
parser.add_argument('--img-size', type=int, default=32, help='input img size')
parser.add_argument('--checkpoint', type=str, required=True, help='model checkpoint path')
parser.add_argument('--seed', type=int, default=0, help='random seed value')
args = parser.parse_args()

accelerator = accelerate.Accelerator(mixed_precision="fp16", )


def build_model():
    if args.net_depth == 34:
        model = models.ResNet34(num_c=args.num_classes, )
    else:
        raise NotImplementedError()
    model.load_state_dict(torch.load(args.checkpoint))
    model = accelerator.prepare_model(model)
    return model, accelerator.device


def build_dataset():
    data_transform = transforms.Compose([
        transforms.CenterCrop(args.img_size, ),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(args.data_dir, transform=data_transform, )
    return dataset


def build_adv_data(bat_x, min_pixel, max_pixel, device, bat_y=None, model=None, loss_func=None, dataset=None):
    if args.adv_type == 'Noise':
        noisy_x = torch.add(bat_x.data, torch.randn(bat_x.size()).to(device), alpha=args.adv_noise)
        noisy_x = torch.clamp(noisy_x, min_pixel, max_pixel)
    else:
        model.zero_grad()
        bat_x_cp = Variable(bat_x.data, requires_grad=True)
        out = model(bat_x_cp)
        loss = loss_func(out, bat_y)
        # loss.backward()
        accelerator.backward(loss)

        if args.adv_type == 'FGSM':
            gradient = torch.ge(bat_x_cp.grad.data, 0)  # 获取图片张量对优的负梯度方向
            gradient = (gradient.float() - 0.5) * 2  # 把0、1变为-1、1
            # 就是对三个通道的值分别除以一个系数
            gradient.index_copy_(1, torch.LongTensor([0]).to(device), \
                                 gradient.index_select(1, torch.LongTensor([0]).to(device)) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).to(device), \
                                 gradient.index_select(1, torch.LongTensor([1]).to(device)) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).to(device), \
                                 gradient.index_select(1, torch.LongTensor([2]).to(device)) / (0.2010))
            noisy_x = torch.add(bat_x_cp.data, gradient, alpha=args.adv_noise)  # 生成对抗图片
            noisy_x = torch.clamp(noisy_x, min_pixel, max_pixel)
        elif args.adv_type == 'BIM':
            gradient = torch.sign(bat_x_cp.grad.data)
            for k in range(5):
                bat_x_cp = torch.add(bat_x_cp.data, gradient, alpha=args.adv_noise)
                bat_x_cp = torch.clamp(bat_x_cp, min_pixel, max_pixel)
                bat_x_cp = Variable(bat_x_cp, requires_grad=True)
                out = model(bat_x_cp)
                loss = loss_func(out, bat_y)
                # loss.backward()
                accelerator.backward(loss)
                gradient = torch.sign(bat_x_cp.grad.data)
                gradient.index_copy_(1, torch.LongTensor([0]).to(device), \
                                     gradient.index_select(1, torch.LongTensor([0]).to(device)) / (0.2023))
                gradient.index_copy_(1, torch.LongTensor([1]).to(device), \
                                     gradient.index_select(1, torch.LongTensor([1]).to(device)) / (0.1994))
                gradient.index_copy_(1, torch.LongTensor([2]).to(device), \
                                     gradient.index_select(1, torch.LongTensor([2]).to(device)) / (0.2010))
            noisy_x = torch.add(bat_x_cp.data, gradient, alpha=args.adv_noise)  # 生成对抗图片
            noisy_x = torch.clamp(noisy_x, min_pixel, max_pixel)
        elif args.adv_type == 'DeepFool':
            _, noisy_x = adversary.deepfool(model, bat_x.data.clone(), bat_y.data.cpu(), args.num_classes, step_size=args.adv_noise, train_mode=False)
            noisy_x = noisy_x.to(device)
            noisy_x = torch.clamp(noisy_x, min_pixel, max_pixel)
        elif args.adv_type == 'CWL2':
            _, noisy_x = adversary.cw(model, bat_x.data.clone(), bat_y.data.cpu(), 1.0, 'l2', crop_frac=1.0)
            noisy_x = noisy_x.to(device)
            noisy_x = torch.clamp(noisy_x, min_pixel, max_pixel)
        elif args.adv_type == 'NST':
            import random
            frozen_module(model)

            trg_bat_x = []
            for i in range(len(bat_x)):
                idx = int(random.random() * len(dataset))
                while dataset[idx][1] == bat_y[i].item():
                    idx = int(random.random() * len(dataset))
                trg_bat_x.append(torch.unsqueeze(dataset[idx][0], dim=0))
            trg_bat_x = torch.cat(trg_bat_x, dim=0)
            trg_bat_x = trg_bat_x.to(device)
            _, trg_feat_list = model.feature_list(trg_bat_x)
            trg_style_feat, trg_content_feat = trg_feat_list[1], trg_feat_list[4]

            noisy_x = bat_x.clone()
            for i in range(len(noisy_x)):
                x = nn.Parameter(torch.unsqueeze(noisy_x[i], dim=0), requires_grad=True)
                optimizer = torch.optim.SGD([x], lr=args.adv_noise, momentum=0.9, weight_decay=5e-4, )
                optimizer = accelerator.prepare_optimizer(optimizer)
                style_feat_y, content_feat_y = torch.unsqueeze(trg_style_feat[i], dim=0), torch.unsqueeze(trg_content_feat[i], dim=0)
                for k in range(10):
                    _, feat_list = model.feature_list(x)
                    style_feat_pred, content_feat_pred = feat_list[1], feat_list[4]
                    loss = torch.mean(torch.square(style_feat_pred - style_feat_y)) + torch.mean(torch.square(content_feat_pred - content_feat_y))
                    optimizer.zero_grad()
                    # loss.backward()
                    accelerator.backward(loss)
                    optimizer.step()
                noisy_x[i] = torch.squeeze(x.detach(), dim=0)

            noisy_x = torch.clamp(noisy_x, min_pixel, max_pixel)
            unfrozen_module(model)
        else:
            raise NotImplementedError(f'not support noise type: {args.adv_type}')
    return noisy_x


def save_adv_data(bat_x, bat_class, bat_filename):
    for i in range(len(bat_x)):
        im_arr = (bat_x[i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        trg_dir = os.path.join(args.out_dir, bat_class[i])
        os.makedirs(trg_dir, exist_ok=True)
        Image.fromarray(im_arr).save(os.path.join(trg_dir, bat_filename[i]))


def main():
    logging_conf(args.log_file)
    log(args)
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    model, device = build_model()
    loss_func = nn.CrossEntropyLoss().to(device)
    log(model)

    dataset = build_dataset()
    log(dataset)

    min_pixel = 0.
    max_pixel = 1.
    clean_correct = 0
    noise_correct = 0
    processed_count = 0

    model.eval()
    for start_pos in range(0, len(dataset), args.batch_size):
        bat_x = []
        bat_y = []
        bat_class = []
        bat_filename = []
        for i in range(start_pos, min(start_pos + args.batch_size, len(dataset))):
            bat_x.append(torch.unsqueeze(dataset[i][0], dim=0))
            bat_y.append(dataset[i][1])
            bat_class.append(dataset.classes[dataset[i][1]])
            bat_filename.append(os.path.basename(dataset.imgs[i][0]))
        bat_x = torch.cat(bat_x, dim=0).to(device)
        bat_y = torch.tensor(bat_y).to(device)
        out = model(bat_x)
        pred_y = torch.max(out, dim=1)[1]
        clean_correct += torch.sum(pred_y == bat_y).item()

        noise_x = build_adv_data(bat_x, min_pixel, max_pixel, device, bat_y=bat_y, model=model, loss_func=loss_func, dataset=dataset)
        out = model(noise_x)
        noise_pred_y = torch.max(out, dim=1)[1]
        noise_correct += torch.sum(noise_pred_y == bat_y).item()

        save_adv_data(noise_x, bat_class, bat_filename)
        processed_count += len(bat_x)

        log(f'precessed: {processed_count} / {len(dataset)}')
        # break

    log(f'noise type : {args.adv_type}')
    log(f'clean data correct: {clean_correct}/{len(dataset)} {clean_correct / float(len(dataset))}')
    log(f'noise data correct: {noise_correct}/{len(dataset)} {noise_correct / float(len(dataset))}')

    
if __name__ == '__main__':
    main()

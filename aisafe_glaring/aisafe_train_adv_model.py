import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import accelerate
import os
from tqdm import tqdm

import models
from lib.dataset import AdvMixDataset

from dowdyboy_lib.log import logging_conf, log
from dowdyboy_lib.rand import set_seed


def do_nothing(x):
    return x


parser = argparse.ArgumentParser(description='train a resnet model with adv dataset')
# net config
parser.add_argument('--model-depth', type=int, default=34, help='model depth of resnet')
parser.add_argument('--img-size', type=int, default=32, help='input img size')
parser.add_argument('--num-classes', type=int, default=10, help='original class num')
# data config
parser.add_argument('--train-dir-list', type=str, nargs='*', required=True, help='Image Folders for train data, include different adv type')
parser.add_argument('--val-dir-list', type=str, nargs='*', required=True, help='Image Folders for val data, include different adv type')
parser.add_argument('--rand-aug', type=bool, default=False, help='is use rand aug with imagenet policy')
parser.add_argument('--adv-ratio', type=float, default=0.5, help='adv data ratio in dataset')
# optimizer config
parser.add_argument('--lr', type=float, default=0.01, help='sgd lr')
parser.add_argument('--momentum', type=float, default=0.9, help='sgd momentum')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='sgd weight decay')
# lr schedule config
parser.add_argument('--step-size', type=int, default=None, help='step lr size (epoch)')
parser.add_argument('--gamma', type=float, default=None, help='step lr gamma')
# train config
parser.add_argument('--batch-size', type=int, default=64, help='train batch size')
parser.add_argument('--num-workers', type=int, default=4, help='dataloader worker number')
parser.add_argument('--out-dir', type=str, required=True, help='store log and checkpoint')
parser.add_argument('--save-interval', type=int, default=10, help='how to save checkpoint')
parser.add_argument('--save-best', type=bool, default=True, help='is to save best checkpoint on val')
parser.add_argument('--epoch', type=int, default=150, help='num of epoch to train')
parser.add_argument('--seed', type=int, default=0, help='random seed value')

args = parser.parse_args()
accelerator = accelerate.Accelerator(mixed_precision='fp16', )


def build_model():
    if args.model_depth == 34:
        model = models.ResNet34(args.num_classes)
    else:
        raise NotImplementedError(f'not support pre net : {args.net_type}')
    model = accelerator.prepare_model(model)
    return model, accelerator.device


def build_data():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.45, 1.0), ),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment() if args.rand_aug else transforms.Lambda(do_nothing),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.CenterCrop(args.img_size, ),
        transforms.ToTensor(),
    ])
    train_dataset = AdvMixDataset(args.train_dir_list[0], args.train_dir_list[1:], is_train=True, transforms=train_transform, adv_ratio=args.adv_ratio, )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers, )
    train_loader = accelerator.prepare_data_loader(train_loader)
    val_dataset = AdvMixDataset(args.val_dir_list[0], args.val_dir_list[1:], is_train=False, transforms=test_transform, adv_ratio=args.adv_ratio, )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers, )
    val_loader = accelerator.prepare_data_loader(val_loader)
    return train_loader, train_dataset, val_loader, val_dataset


def build_optimizer(model):
    train_net_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, )
    train_net_optimizer = accelerator.prepare_optimizer(train_net_optimizer)
    if args.step_size is not None and args.gamma is not None:
        train_net_lr_scheduler = torch.optim.lr_scheduler.StepLR(train_net_optimizer, step_size=args.step_size, gamma=args.gamma, )
    else:
        train_net_lr_scheduler = None
    return train_net_optimizer, train_net_lr_scheduler


def metric_log(
        train_loss_list, train_correct_num_list, val_loss_list, val_correct_num_list,
        train_dataset, val_dataset,
        lr=None, writer=None, step=None, ):
    train_acc = float(sum(train_correct_num_list)) / len(train_dataset)
    train_loss = float(sum(train_loss_list)) / len(train_loss_list)
    log(f"train loss: {train_loss}, train acc: {train_acc}, lr: {lr}")

    val_acc = float(sum(val_correct_num_list)) / len(val_dataset)
    val_loss = float(sum(val_loss_list)) / len(val_loss_list)
    log(f"val loss: {val_loss}, val acc: {val_acc}, lr: {lr}")

    if writer is not None and step is not None:
        writer.add_scalar('epoch_train_loss', train_loss, global_step=step)
        writer.add_scalar('epoch_train_acc', train_acc, global_step=step)

        writer.add_scalar('epoch_val_loss', val_loss, global_step=step)
        writer.add_scalar('epoch_val_acc', val_acc, global_step=step)

        writer.add_scalar('lr', lr, global_step=step)

    return val_acc


def save_checkpoint(ep, best_acc, val_acc, model, optimizer, prefix=''):
    model_pth_dir = os.path.join(args.out_dir, 'checkpoints', 'model')
    optimizer_pth_dir = os.path.join(args.out_dir, 'checkpoints', 'optimizer')
    best_pth_dir = os.path.join(args.out_dir, 'checkpoints')
    os.makedirs(model_pth_dir, exist_ok=True)
    os.makedirs(optimizer_pth_dir, exist_ok=True)
    if ep % args.save_interval == 0:
        torch.save(model.state_dict(), os.path.join(model_pth_dir, f'{prefix}epoch_{ep}_model.pth'))
        torch.save(optimizer.state_dict(), os.path.join(optimizer_pth_dir, f'{prefix}epoch_{ep}_optimizer.pth'))
        log(f'success saved checkpoint for {prefix}epoch {ep}.')
    if args.save_best and val_acc > best_acc:
        for filename in os.listdir(best_pth_dir):
            if filename.startswith(f'best_model_{prefix}') or filename.startswith(f'best_optimizer_{prefix}'):
                os.remove(os.path.join(best_pth_dir, filename))
        torch.save(model.state_dict(), os.path.join(best_pth_dir, f'best_model_{prefix}epoch_{ep}.pth'))
        torch.save(optimizer.state_dict(), os.path.join(best_pth_dir, f'best_optimizer_{prefix}epoch_{ep}.pth'))
        log(f'best val acc update {prefix}epoch {ep} ({val_acc}), saved checkpoint.')
        return val_acc
    else:
        return best_acc


def main():
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    logging_conf(os.path.join(args.out_dir, 'train.log'))
    writer = SummaryWriter(os.path.join(args.out_dir, 'tf_logs'))
    log(args)

    train_loader, train_dataset, val_loader, val_dataset = build_data()
    log('train transform:')
    log(train_dataset.transforms)
    log('test transform:')
    log(val_dataset.transforms)
    log(f'train dataset size: {len(train_dataset)}')
    log(f'val dataset size: {len(val_dataset)}')

    model, device = build_model()
    loss_func = nn.CrossEntropyLoss().to(device)
    log(model)
    optimizer, lr_scheduler = build_optimizer(model)

    best_val_acc = 0.

    for ep in range(1, args.epoch + 1):
        log(f'======= epoch {ep} begin ========')
        model.train()
        train_correct_nums, train_losses = [], []
        tqdm_loader = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (bat_x, bat_im_y, _) in tqdm_loader:
            pred_y = model(bat_x)
            loss = loss_func(pred_y, bat_im_y)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            correct_count = torch.sum(torch.argmax(pred_y, dim=1) == bat_im_y).item()
            train_correct_nums.append(correct_count)
            train_losses.append(loss.item())

            tqdm_loader.set_description(f'Epoch [{ep}/{args.epoch}]')
            tqdm_loader.set_postfix(
                loss=loss.item(),
                acc=round(float(correct_count) / len(bat_x), 4),
            )

        model.eval()
        val_correct_nums, val_losses = [], []
        tqdm_loader = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, (bat_x, bat_im_y, _) in tqdm_loader:
            with torch.no_grad():
                pred_y = model(bat_x)
                loss = loss_func(pred_y, bat_im_y)

            correct_count = torch.sum(torch.argmax(pred_y, dim=1) == bat_im_y).item()
            val_correct_nums.append(correct_count)
            val_losses.append(loss.item())

            tqdm_loader.set_description(f'Epoch [{ep}/{args.epoch}]')
            tqdm_loader.set_postfix(
                loss=loss.item(),
                acc=round(float(correct_count) / len(bat_x), 4),
            )

        val_acc = metric_log(
            train_losses, train_correct_nums, val_losses, val_correct_nums,
            train_dataset, val_dataset,
            lr=optimizer.state_dict()['param_groups'][0]['lr'],
            writer=writer, step=ep,
        )

        best_val_acc = save_checkpoint(ep, best_val_acc, val_acc, model, optimizer, )

        if lr_scheduler is not None:
            lr_scheduler.step()

    return


if __name__ == '__main__':
    main()

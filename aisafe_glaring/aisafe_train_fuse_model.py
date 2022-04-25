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


parser = argparse.ArgumentParser(description='train a fuse model')
# net config
parser.add_argument('--net-type', type=str, required=True, help='resnet34')
parser.add_argument('--img-size', type=int, default=32, help='input img size')
parser.add_argument('--num-classes', type=int, default=10, help='original class num')
parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint file for pre net')
parser.add_argument('--head-sa-layer-num', type=int, default=4, help='diff head transformer layer number')
parser.add_argument('--head-type', type=str, default='v4', help='head version type: v3 , v4')
# data config
parser.add_argument('--train-dir-list', type=str, nargs='*', required=True, help='Image Folders for train data, include different adv type')
parser.add_argument('--val-dir-list', type=str, nargs='*', required=True, help='Image Folders for val data, include different adv type')
parser.add_argument('--rand-aug', type=bool, default=False, help='is use rand aug with imagenet policy')
parser.add_argument('--adv-ratio', type=float, default=0.5, help='adv data ratio in dataset')
# optimizer config
parser.add_argument('--lr', type=float, default=0.01, help='sgd lr')
parser.add_argument('--momentum', type=float, default=0.9, help='sgd momentum')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='sgd weight decay')
parser.add_argument('--lr-head', type=float, default=0.01, help='lr head for adv detect head')
parser.add_argument('--diff-loss-weight', type=float, default=1., help='loss weight 01 diff')
parser.add_argument('--im-loss-weight', type=float, default=1., help='loss weight image classify')
parser.add_argument('--im-backward-type', type=str, default='1-stage', help='image classify target net backward type : 1-stage, 2-stage')
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
    if args.net_type == 'resnet34':
        pre_net = models.ResNet34(args.num_classes)
        pre_net.load_state_dict(torch.load(args.checkpoint))
        train_net = models.ResNet34(args.num_classes)
        # head = models.GlaringDetectorHead(960, [1, 2, 3, 4])
        # head = models.GlaringDetectorHeadV2([64, 128, 256, 512], [1, 2, 3, 4], 960, )
        # head = models.GlaringDetectorHeadV3([64, 128, 256, 512], [1, 2, 3, 4], 4, )
        if args.head_type == 'v3':
            head = models.GlaringDetectorHeadV3([64, 128, 256, 512], [1, 2, 3, 4], args.head_sa_layer_num, )
        elif args.head_type == 'v4':
            head = models.GlaringDetectorHeadV4([64, 128, 256, 512], [1, 2, ], args.head_sa_layer_num, )
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError(f'not support pre net : {args.net_type}')
    model = models.GlaringNet(pre_net, train_net, head, )
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
    head_optimizer = torch.optim.SGD(model.head.parameters(), lr=args.lr_head, momentum=args.momentum, weight_decay=args.weight_decay, )
    train_net_optimizer = torch.optim.SGD(model.train_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, )
    head_optimizer = accelerator.prepare_optimizer(head_optimizer)
    train_net_optimizer = accelerator.prepare_optimizer(train_net_optimizer)
    if args.step_size is not None and args.gamma is not None:
        head_lr_scheduler = torch.optim.lr_scheduler.StepLR(head_optimizer, step_size=args.step_size, gamma=args.gamma, )
        train_net_lr_scheduler = torch.optim.lr_scheduler.StepLR(train_net_optimizer, step_size=args.step_size, gamma=args.gamma, )
    else:
        head_lr_scheduler = None
        train_net_lr_scheduler = None
    return head_optimizer, train_net_optimizer, head_lr_scheduler, train_net_lr_scheduler


def metric_log(
        train_diff_loss_list, train_diff_correct_num_list, val_diff_loss_list, val_diff_correct_num_list,
        train_im_loss_list, train_im_correct_num_list, val_im_loss_list, val_im_correct_num_list,
        train_dataset, val_dataset,
        lr=None, writer=None, step=None, ):
    train_diff_acc = float(sum(train_diff_correct_num_list)) / len(train_dataset)
    train_diff_loss = float(sum(train_diff_loss_list)) / len(train_diff_loss_list)
    log(f"train diff loss: {train_diff_loss}, train diff acc: {train_diff_acc}, lr: {lr}")

    train_im_acc = float(sum(train_im_correct_num_list)) / len(train_dataset)
    loss_arr = np.array(train_im_loss_list)
    train_im_loss = float(np.sum(loss_arr[:, 0])) / len(loss_arr)
    train_feat_loss = float(np.sum(loss_arr[:, 1])) / len(loss_arr)
    train_loss = float(np.sum(loss_arr[:, 2])) / len(loss_arr)
    log(f"train im loss: {train_im_loss}, train feat loss: {train_feat_loss}, train loss: {train_loss}, train im acc: {train_im_acc}, lr: {lr}")

    val_diff_acc = float(sum(val_diff_correct_num_list)) / len(val_dataset)
    val_diff_loss = float(sum(val_diff_loss_list)) / len(val_diff_loss_list)
    log(f"val diff loss: {val_diff_loss}, val diff acc: {val_diff_acc}, lr: {lr}")

    val_im_acc = float(sum(val_im_correct_num_list)) / len(val_dataset)
    loss_arr = np.array(val_im_loss_list)
    val_im_loss = float(np.sum(loss_arr[:, 0])) / len(loss_arr)
    val_feat_loss = float(np.sum(loss_arr[:, 1])) / len(loss_arr)
    val_loss = float(np.sum(loss_arr[:, 2])) / len(loss_arr)
    log(f"val im loss: {val_im_loss}, val feat loss: {val_feat_loss}, val loss: {val_loss}, val im acc: {val_im_acc}, lr: {lr}")

    if writer is not None and step is not None:
        writer.add_scalar('epoch_train_diff_loss', train_diff_loss, global_step=step)
        writer.add_scalar('epoch_train_diff_acc', train_diff_acc, global_step=step)
        writer.add_scalar('epoch_train_im_loss', train_im_loss, global_step=step)
        writer.add_scalar('epoch_train_feat_loss', train_feat_loss, global_step=step)
        writer.add_scalar('epoch_train_loss', train_loss, global_step=step)
        writer.add_scalar('epoch_train_im_acc', train_im_acc, global_step=step)

        writer.add_scalar('epoch_val_diff_loss', val_diff_loss, global_step=step)
        writer.add_scalar('epoch_val_diff_acc', val_diff_acc, global_step=step)
        writer.add_scalar('epoch_val_im_loss', val_im_loss, global_step=step)
        writer.add_scalar('epoch_val_feat_loss', val_feat_loss, global_step=step)
        writer.add_scalar('epoch_val_loss', val_loss, global_step=step)
        writer.add_scalar('epoch_val_im_acc', val_im_acc, global_step=step)

        writer.add_scalar('lr', lr, global_step=step)

    return val_diff_acc, val_im_acc


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
    loss_func_diff = nn.CrossEntropyLoss().to(device)
    loss_func_im = nn.CrossEntropyLoss().to(device)
    loss_func_feat = nn.SmoothL1Loss().to(device)
    log(model)
    head_optimizer, train_net_optimizer, head_lr_scheduler, train_net_lr_scheduler = build_optimizer(model)

    best_diff_val_acc = 0.
    best_im_val_acc = 0.

    for ep in range(1, args.epoch + 1):
        log(f'======= epoch {ep} begin ========')
        model.train()
        train_diff_correct_nums, train_diff_losses = [], []
        train_im_correct_nums, train_im_losses = [], []
        tqdm_loader = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (bat_x, bat_im_y, bat_diff_y) in tqdm_loader:
            pred_diff_y, diff_feat = model.forward_for_head(bat_x)
            loss_diff = loss_func_diff(pred_diff_y, bat_diff_y)
            head_optimizer.zero_grad()
            accelerator.backward(loss_diff)
            head_optimizer.step()

            diff_correct_count = torch.sum(torch.argmax(pred_diff_y, dim=1) == bat_diff_y).item()
            train_diff_correct_nums.append(diff_correct_count)
            train_diff_losses.append(loss_diff.item())

            if args.im_backward_type == '1-stage':
                model.frozen_head(True)
                diff_feat = diff_feat.detach().clone()
                pred_im_y, pred_diff_feat = model.forward_for_train_net(bat_x)
                loss_im = loss_func_im(pred_im_y, bat_im_y)
                loss_feat = loss_func_feat(pred_diff_feat, diff_feat)
                loss = args.im_loss_weight * loss_im + args.diff_loss_weight * loss_feat
                train_net_optimizer.zero_grad()
                accelerator.backward(loss)
                train_net_optimizer.step()
                model.frozen_head(False)
            elif args.im_backward_type == '2-stage':
                model.frozen_head(True)
                diff_feat = diff_feat.detach().clone()

                train_net_optimizer.zero_grad()
                pred_im_y, _ = model.forward_for_train_net(bat_x)
                loss_im = loss_func_im(pred_im_y, bat_im_y)
                loss_im = args.im_loss_weight * loss_im
                accelerator.backward(loss_im)
                train_net_optimizer.step()

                train_net_optimizer.zero_grad()
                _, pred_diff_feat = model.forward_for_train_net(bat_x)
                loss_feat = loss_func_feat(pred_diff_feat, diff_feat)
                loss_feat = args.diff_loss_weight * loss_feat
                accelerator.backward(loss_feat)
                train_net_optimizer.step()

                loss = loss_im + loss_feat
                model.frozen_head(False)
            else:
                raise NotImplementedError()

            im_correct_count = torch.sum(torch.argmax(pred_im_y, dim=1) == bat_im_y).item()
            train_im_correct_nums.append(im_correct_count)
            train_im_losses.append([loss_im.item(), loss_feat.item(), loss.item()])

            tqdm_loader.set_description(f'Epoch [{ep}/{args.epoch}]')
            tqdm_loader.set_postfix(
                loss_diff=loss_diff.item(), loss_im=loss_im.item(), loss_feat=loss_feat.item(), loss=loss.item(),
                acc_diff=round(float(diff_correct_count) / len(bat_x), 4),
                acc_im=round(float(im_correct_count) / len(bat_x), 4),
            )

        model.eval()
        val_diff_correct_nums, val_diff_losses = [], []
        val_im_correct_nums, val_im_losses = [], []
        tqdm_loader = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, (bat_x, bat_im_y, bat_diff_y) in tqdm_loader:
            with torch.no_grad():
                pred_diff_y, diff_feat = model.forward_for_head(bat_x)
                loss_diff = loss_func_diff(pred_diff_y, bat_diff_y)

            diff_correct_count = torch.sum(torch.argmax(pred_diff_y, dim=1) == bat_diff_y).item()
            val_diff_correct_nums.append(diff_correct_count)
            val_diff_losses.append(loss_diff.item())

            model.frozen_head(True)
            with torch.no_grad():
                diff_feat = diff_feat.detach().clone()
                pred_im_y, pred_diff_feat = model.forward_for_train_net(bat_x)
                loss_im = loss_func_im(pred_im_y, bat_im_y)
                loss_feat = loss_func_feat(pred_diff_feat, diff_feat)
                loss = args.im_loss_weight * loss_im + args.diff_loss_weight * loss_feat
            model.frozen_head(False)

            im_correct_count = torch.sum(torch.argmax(pred_im_y, dim=1) == bat_im_y).item()
            val_im_correct_nums.append(im_correct_count)
            val_im_losses.append([loss_im.item(), loss_feat.item(), loss.item()])

            tqdm_loader.set_description(f'Epoch [{ep}/{args.epoch}]')
            tqdm_loader.set_postfix(
                loss_diff=loss_diff.item(), loss_im=loss_im.item(), loss_feat=loss_feat.item(), loss=loss.item(),
                acc_diff=round(float(diff_correct_count) / len(bat_x), 4),
                acc_im=round(float(im_correct_count) / len(bat_x), 4),
            )

        val_diff_acc, val_im_acc = metric_log(
            train_diff_losses, train_diff_correct_nums, val_diff_losses, val_diff_correct_nums,
            train_im_losses, train_im_correct_nums, val_im_losses, val_im_correct_nums,
            train_dataset, val_dataset,
            lr=train_net_optimizer.state_dict()['param_groups'][0]['lr'],
            writer=writer, step=ep,
        )

        best_diff_val_acc = save_checkpoint(ep, best_diff_val_acc, val_diff_acc, model.head, head_optimizer, prefix='head-', )
        best_im_val_acc = save_checkpoint(ep, best_im_val_acc, val_im_acc, model.train_net, train_net_optimizer, prefix='train-net-', )

        if head_lr_scheduler is not None:
            head_lr_scheduler.step()
        if train_net_lr_scheduler is not None:
            train_net_lr_scheduler.step()

    # bat_x = torch.randn((2, 3, 32, 32)).to(device)
    # bat_diff_y = torch.zeros((2,), dtype=torch.uint8).to(device)
    # bat_im_y = torch.zeros((2,), dtype=torch.uint8).to(device)
    #
    # pred_diff_y, diff_feat = model.forward_for_head(bat_x)
    # loss_diff = loss_func_diff(pred_diff_y, bat_diff_y)
    # loss_diff.backward()
    #
    # model.frozen_head(True)
    # diff_feat = diff_feat.detach().clone()
    # pred_im_y, pred_diff_feat = model.forward_for_train_net(bat_x)
    # loss_im = loss_func_im(pred_im_y, bat_im_y)
    # loss_feat = loss_func_feat(pred_diff_feat, diff_feat)
    # loss = loss_im + loss_feat
    # loss.backward()
    # model.frozen_head(False)

    return


if __name__ == '__main__':
    main()

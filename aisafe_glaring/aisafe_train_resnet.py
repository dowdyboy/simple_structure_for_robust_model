import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import models

import argparse
from tqdm import tqdm
import logging
import os

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

parser = argparse.ArgumentParser(description='train a resnet model for aisafe dataset')
parser.add_argument('--model-depth', type=int, default=34, help='model depth of resnet')
parser.add_argument('--num-classes', type=int, default=10, help='to classify num')
parser.add_argument('--batch-size', type=int, default=64, help='batch size for data loader')
parser.add_argument('--img-size', type=int, default=32, help='input img size')
parser.add_argument('--rand-aug', type=bool, default=False, help='is use rand aug with imagenet policy')
parser.add_argument('--train-dir', type=str, required=True, help='Image Folder for train data')
parser.add_argument('--val-dir', type=str, required=True, help='Image Folder for val data')
parser.add_argument('--num-workers', type=int, default=4, help='dataloader worker number')
parser.add_argument('--lr', type=float, default=0.01, help='sgd lr')
parser.add_argument('--momentum', type=float, default=0.9, help='sgd momentum')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='sgd weight decay')
parser.add_argument('--step-size', type=int, default=None, help='step lr size (epoch)')
parser.add_argument('--gamma', type=float, default=None, help='step lr gamma')
parser.add_argument('--out-dir', type=str, default='./output', help='store log and checkpoint')
parser.add_argument('--save-interval', type=int, default=1, help='how to save checkpoint')
parser.add_argument('--save-best', type=bool, default=True, help='is to save best checkpoint on val')
parser.add_argument('--epoch', type=int, default=150, help='num of epoch to train')
parser.add_argument('--seed', type=int, default=0, help='random seed value')
args = parser.parse_args()


def do_nothing(x):
    return x


def set_seed(seed_value):
    import numpy as np
    import torch
    import random
    import os
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed_value)     # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）
    torch.backends.cudnn.deterministic = True


def log(txt):
    print(txt)
    logging.info(txt)


def build_dataloader():
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
    train_dataset = ImageFolder(args.train_dir, transform=train_transform, )
    val_dataset = ImageFolder(args.val_dir, transform=test_transform, )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, )
    return train_loader, val_loader, train_dataset, val_dataset


def build_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_depth == 34:
        model = models.ResNet34(args.num_classes)
    else:
        raise NotImplementedError
    model.to(device)
    return model, device


def build_optimizer(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, )
    if args.step_size is not None and args.gamma is not None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, )
    else:
        lr_scheduler = None
    return optimizer, lr_scheduler


def save_checkpoint(ep, best_acc, val_acc, model, optimizer):
    model_pth_dir = os.path.join(args.out_dir, 'checkpoints', 'model')
    optimizer_pth_dir = os.path.join(args.out_dir, 'checkpoints', 'optimizer')
    best_pth_dir = os.path.join(args.out_dir, 'checkpoints')
    os.makedirs(model_pth_dir, exist_ok=True)
    os.makedirs(optimizer_pth_dir, exist_ok=True)
    if ep % args.save_interval == 0:
        torch.save(model.state_dict(), os.path.join(model_pth_dir, f'epoch_{ep}_model.pth'))
        torch.save(optimizer.state_dict(), os.path.join(optimizer_pth_dir, f'epoch_{ep}_optimizer.pth'))
        log(f'success saved checkpoint for epoch {ep}.')
    if args.save_best and val_acc > best_acc:
        for filename in os.listdir(best_pth_dir):
            if filename.startswith('best_'):
                os.remove(os.path.join(best_pth_dir, filename))
        torch.save(model.state_dict(), os.path.join(best_pth_dir, f'best_model_epoch_{ep}.pth'))
        torch.save(optimizer.state_dict(), os.path.join(best_pth_dir, f'best_optimizer_epoch_{ep}.pth'))
        log(f'best val acc update epoch {ep} ({val_acc}), saved checkpoint.')
        return val_acc
    else:
        return best_acc


def main():
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.out_dir, 'train.log'), level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s', filemode='w')
    writer = SummaryWriter(os.path.join(args.out_dir, 'tf_logs'))
    log(args)

    train_loader, val_loader, train_dataset, val_dataset = build_dataloader()

    # train_dataset = val_dataset
    # train_loader = val_loader

    log('train transform:')
    log(train_dataset.transforms)
    log('val transform:')
    log(val_dataset.transforms)
    log(f'train dataset size: {len(train_dataset)}')
    log(f'val dataset size: {len(val_dataset)}')

    model, device = build_model()
    loss_func = nn.CrossEntropyLoss()
    loss_func = loss_func.to(device)
    log(model)
    optimizer, lr_scheduler = build_optimizer(model)
    log(f'optimizer: {optimizer}')
    log(f'lr_scheduler: {lr_scheduler}')

    if has_native_amp:
        amp_autocast = torch.cuda.amp.autocast
        log('Using native Torch AMP. Training in mixed precision.')

    best_val_acc = 0.
    train_iter_count = 0

    for ep in range(1, args.epoch + 1):
        log(f'======= epoch {ep} begin ========')
        model.train()
        epoch_correct_nums = []
        epoch_losses = []
        tqdm_loader = tqdm(enumerate(train_loader), total=len(train_loader))
        for bat_i, (bat_x, bat_y) in tqdm_loader:
            bat_x, bat_y = bat_x.to(device), bat_y.to(device)
            optimizer.zero_grad()
            if has_native_amp:
                with amp_autocast():
                    pred_y = model(bat_x)
                    loss = loss_func(pred_y, bat_y)
            else:
                pred_y = model(bat_x)
                loss = loss_func(pred_y, bat_y)
            loss.backward()
            optimizer.step()

            bat_correct_num = torch.sum(torch.argmax(pred_y, dim=1) == bat_y).item()
            epoch_losses.append(loss.item())
            epoch_correct_nums.append(bat_correct_num)

            tqdm_loader.set_description(f'Epoch [{ep}/{args.epoch}]')
            tqdm_loader.set_postfix(loss=loss.item(), acc=float(bat_correct_num) / len(bat_x))
            writer.add_scalar('train_iter_loss', loss.item(), global_step=train_iter_count)
            writer.add_scalar('train_iter_acc', float(bat_correct_num) / len(bat_x), global_step=train_iter_count)
            train_iter_count += 1
        train_loss = float(sum(epoch_losses)) / len(epoch_losses)
        train_acc = float(sum(epoch_correct_nums)) / len(train_dataset)
        log(f"train loss: {train_loss},train acc: {train_acc}, lr: {optimizer.state_dict()['param_groups'][0]['lr']}")
        writer.add_scalar('train_epoch_loss', train_loss, global_step=ep)
        writer.add_scalar('train_epoch_acc', train_acc, global_step=ep)
        writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], global_step=ep)

        model.eval()
        epoch_correct_nums = []
        epoch_losses = []
        tqdm_loader = tqdm(enumerate(val_loader), total=len(val_loader))
        for bat_i, (bat_x, bat_y) in tqdm_loader:
            bat_x, bat_y = bat_x.to(device), bat_y.to(device)
            if has_native_amp:
                with amp_autocast():
                    pred_y = model(bat_x)
            else:
                pred_y = model(bat_x)
            bat_correct_num = torch.sum(torch.argmax(pred_y, dim=1) == bat_y).item()
            epoch_losses.append(loss.item())
            epoch_correct_nums.append(bat_correct_num)

            tqdm_loader.set_description(f'Epoch [{ep}/{args.epoch}]')
            tqdm_loader.set_postfix(loss=loss.item(), acc=float(bat_correct_num) / len(bat_x))
        val_loss = float(sum(epoch_losses)) / len(epoch_losses)
        val_acc = float(sum(epoch_correct_nums)) / len(val_dataset)
        log(f"val loss: {val_loss},val acc: {val_acc}, lr: {optimizer.state_dict()['param_groups'][0]['lr']}")
        writer.add_scalar('val_epoch_loss', val_loss, global_step=ep)
        writer.add_scalar('val_epoch_acc', val_acc, global_step=ep)

        if lr_scheduler is not None:
            lr_scheduler.step()

        best_val_acc = save_checkpoint(ep, best_val_acc, val_acc, model, optimizer)


if __name__ == '__main__':
    main()


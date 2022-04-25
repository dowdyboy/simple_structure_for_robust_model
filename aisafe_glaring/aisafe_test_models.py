import argparse
from tqdm import tqdm
import accelerate
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import models
from lib.dataset import AdvMixDataset
from dowdyboy_lib.log import log, logging_conf

parser = argparse.ArgumentParser(description='test models')
parser.add_argument('--test-type', type=str, required=True, help='test type: im | diff')
parser.add_argument('--model-type', type=str, required=True, help='model type: resnet34')
parser.add_argument('--img-size', type=int, default=32, help='input img size')
parser.add_argument('--num-classes', type=int, default=10, help='original class num')
parser.add_argument('--checkpoint', type=str, nargs='*', required=True, help='model checkpoint file path')
parser.add_argument('--head-sa-layer-num', type=int, default=4, help='diff head transformer layer number')
parser.add_argument('--head-type', type=str, default='v4', help='head version type: v3 , v4')
parser.add_argument('--data-dirs', type=str, nargs='*', required=True, help='test data dir path, image folder format')
parser.add_argument('--adv-ratio', type=float, default=0.5, help='adv data ratio in dataset')
parser.add_argument('--log-file', type=str, default=None, help='log file path')
args = parser.parse_args()
accelerator = accelerate.Accelerator(mixed_precision='fp16', )


def build_model():
    if args.test_type == 'im':
        if args.model_type == 'resnet34':
            model = models.ResNet34(args.num_classes)
        else:
            raise NotImplementedError()
        model.load_state_dict(torch.load(args.checkpoint[0]))
    elif args.test_type == 'diff':
        if args.model_type == 'resnet34':
            pre_net = models.ResNet34(args.num_classes)
            pre_net.load_state_dict(torch.load(args.checkpoint[0]))
            # head = models.GlaringDetectorHead(960, [1, 2, 3, 4])
            # head = models.GlaringDetectorHeadV2([64, 128, 256, 512], [1, 2, 3, 4], 960, )
            # head = models.GlaringDetectorHeadV3([64, 128, 256, 512], [1, 2, 3, 4], 4, )
            if args.head_type == 'v3':
                head = models.GlaringDetectorHeadV3([64, 128, 256, 512], [1, 2, 3, 4], args.head_sa_layer_num, )
            elif args.head_type == 'v4':
                head = models.GlaringDetectorHeadV4([64, 128, 256, 512], [1, 2, ], args.head_sa_layer_num, )
            else:
                raise NotImplementedError()
            head.load_state_dict(torch.load(args.checkpoint[1]))
            model = models.GlaringNet(pre_net, pre_net, head, )
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    model = accelerator.prepare_model(model)
    return model, accelerator.device


def build_data():
    test_transform = transforms.Compose([
        transforms.CenterCrop(args.img_size, ),
        transforms.ToTensor(),
    ])
    if len(args.data_dirs) == 1:
        test_dataset = ImageFolder(args.data_dirs[0], transform=test_transform, )
    else:
        test_dataset = AdvMixDataset(args.data_dirs[0], args.data_dirs[1:], is_train=False, transforms=test_transform, adv_ratio=args.adv_ratio, )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, )
    test_loader = accelerator.prepare_data_loader(test_loader)
    return test_dataset, test_loader


def main():
    if args.log_file is not None:
        logging_conf(args.log_file)
    log(args)

    model, device = build_model()
    log(model)
    test_dataset, test_loader = build_data()
    log(test_dataset)

    model.eval()
    if isinstance(test_dataset, ImageFolder):
        all_correct_count = 0
        for i in tqdm(range(len(test_dataset)), total=len(test_dataset)):
            bat_x = torch.unsqueeze(test_dataset[i][0], dim=0).to(device)
            bat_y = torch.tensor(test_dataset[i][1], ).view((1, )).to(device) if args.test_type == 'im' else \
                torch.tensor(0, ).view((1, )).to(device)
            with torch.no_grad():
                pred_y = model(bat_x)
            correct_count = torch.sum(torch.argmax(pred_y, dim=1) == bat_y).item()
            all_correct_count += correct_count
        log(f'{all_correct_count} / {len(test_dataset)}  {float(all_correct_count) / len(test_dataset)}')
    elif isinstance(test_dataset, AdvMixDataset):
        all_correct_count = 0
        for i in tqdm(range(len(test_dataset)), total=len(test_dataset)):
            bat_x = torch.unsqueeze(test_dataset[i][0], dim=0).to(device)
            bat_y = torch.tensor(test_dataset[i][1], ).view((1, )).to(device) if args.test_type == 'im' else \
                torch.tensor(test_dataset[i][2], ).view((1, )).to(device)
            with torch.no_grad():
                pred_y = model(bat_x)
            correct_count = torch.sum(torch.argmax(pred_y, dim=1) == bat_y).item()
            all_correct_count += correct_count
        log(f'{all_correct_count} / {len(test_dataset)}  {float(all_correct_count) / len(test_dataset)}')
    else:
        raise NotImplementedError()

    # if args.test_type == 'im':
    # elif args.test_type == 'diff':

    return


if __name__ == '__main__':
    main()

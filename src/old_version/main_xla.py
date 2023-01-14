import torch_xla.distributed.parallel_loader as pl
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from intput.src.dataset import FGVC8_Dataset_MultiClass
from utils.train_xla import XLAModelTrainer, plot_line
from models.resnet import *
from datetime import datetime
import numpy as np
import argparse


src_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.dirname(src_path)
root_path = os.path.dirname(input_path)
output_path = os.path.join(root_path, 'working')

parser = argparse.ArgumentParser()
parser.add_argument('--kaggle', type=bool, default=False, help='if the script is run on kaggle')
parser.add_argument('--batch_size', type=int, default=48, help='size of each image batch')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--max_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--model', type=str, default='resnext', help='which model to use: resnext')
parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer： SGD, Adam')
parser.add_argument('--num_workers', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--chkpoint', type=str, default=None, help='checkpoint')
parser.add_argument('--print_acc_iter', type=int, default=10, help='print accuracy per n iteration')
parser.add_argument('--use_mixup', action='store_true', help='ob mix up verwendet werden')
parser.add_argument('--pretrained', action='store_true', help='whether to use pretrained model')
opt = parser.parse_args()

kaggle_prefix = 'fgvc8-roger10015-img-aug' if opt.kaggle else ''


now_time = datetime.now()
time_str = datetime.strftime(now_time, '%Y-%m-%d_%H-%M')
log_dir = os.path.join(output_path, 'log', time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, 'log.txt')
if os.path.exists(log_file):
    os.remove(log_file)

img_cls = {'complex': 0, 'frog_eye_leaf_spot': 1, 'powdery_mildew': 2, 'rust': 3, 'scab': 4}
num_cls = len(img_cls)

data_dir = os.path.join(input_path, kaggle_prefix, 'img_aug')

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.RandomErasing(),
    transforms.Normalize(norm_mean, norm_std)
])
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

train_txt_path = os.path.join(output_path, 'train.txt')
val_txt_path = os.path.join(output_path, 'val.txt')

train_dataset = FGVC8_Dataset_MultiClass(train_txt_path, num_cls, train_transform)
val_dataset = FGVC8_Dataset_MultiClass(val_txt_path, num_cls, val_transform)

def _run():
    device = xm.xla_device()
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),  # tell PyTorch how many devices (TPU cores) we are using for training
        rank=xm.get_ordinal(),  # tell PyTorch which device (core) we are on currently
        shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False)

    train_dl = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, sampler=train_sampler,
                          num_workers=opt.num_workers)
    val_dl = DataLoader(dataset=val_dataset, batch_size=int(opt.batch_size / 4), sampler=val_sampler,
                        num_workers=opt.num_workers)

    if opt.model == 'resnext50_32x4d':
        if opt.pretrained:
            model = resnext50_32x4d(pretrained=opt.pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_cls)
        else:
            model = resnext50_32x4d(num_classes=num_cls, pretrained=opt.pretrained)

    lr = opt.learning_rate * xm.xrt_world_size()
    if opt.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 2)

    if opt.chkpoint is not None and not opt.pretrained:
        checkpoint = torch.load(opt.chkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('~~~Model checkpoint loaded.~~~')
        if log_file is not None:
            with open(log_file, 'a') as f:
                print('~~~Model checkpoint loaded.~~~', file=f)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for k, v in optimizer.state.items():  # key is Parameter, val is a dict {key='momentum_buffer':tensor(...)}
                if 'momentum_buffer' not in v:
                    continue
                optimizer.state[k]['momentum_buffer'] = optimizer.state[k]['momentum_buffer'].to(device)
            print('~~~Optimizer checkpoint loaded.~~~')
            if log_file is not None:
                with open(log_file, 'a') as f:
                    print('~~~Optimizer checkpoint loaded.~~~', file=f)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    criterion = criterion.to(device)

    loss_rec = {'train': [], 'val': []}
    acc_rec = {'train': [], 'val': []}
    f1_rec = {'train': [], 'val': []}
    if opt.chkpoint is not None:
        best_f1 = checkpoint['best_f1']
        best_epoch = 0
    else:
        best_f1, best_epoch = 0, 0

    start_epoch = -1
    for epoch in range(start_epoch + 1, opt.max_epoch):
        if opt.use_mixup:
            mix_up = True if epoch > 2 else False
        else:
            mix_up = False
        xm.master_print('~~~Model training start. World Size: {}~~~'.format(xm.xrt_world_size()))
        if log_file is not None:
            with open(log_file, 'a') as f:
                xm.master_print('~~~Model training start. World Size: {}~~~'.format(xm.xrt_world_size()), fd=f)

        train_paraloader = pl.ParallelLoader(train_dl, [device])
        val_paraloader = pl.ParallelLoader(val_dl, [device])

        loss_train, acc_train, f1_train = XLAModelTrainer().train(train_paraloader.per_device_loader(device), model,
                                                                   criterion, optimizer, device, epoch,
                                                                   opt.max_epoch, num_cls, opt.print_acc_iter, mix_up, log_file)
        loss_val, acc_val, f1_val = XLAModelTrainer().valid(val_paraloader.per_device_loader(device), model,
                                                                   criterion, device, epoch,
                                                                   opt.max_epoch, num_cls, opt.print_acc_iter, log_file)

        scheduler.step()

        xm.master_print(
            "Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} Train F1:{:.4f} Valid F1:{:.4f} LR:{}".format(
                epoch + 1, opt.max_epoch, acc_train, acc_val, loss_train, loss_val, f1_train, f1_val,
                optimizer.param_groups[0]["lr"]))
        if log_file is not None:
            with open(log_file, 'a') as f:
                xm.master_print(
                    "Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} Train F1:{:.4f} Valid F1:{:.4f} LR:{}".format(
                        epoch + 1, opt.max_epoch, acc_train, acc_val, loss_train, loss_val, f1_train, f1_val,
                        optimizer.param_groups[0]["lr"]), fd=f)

        loss_rec["train"].append(loss_train), loss_rec["val"].append(loss_val)
        acc_rec["train"].append(acc_train), acc_rec["val"].append(acc_val)
        f1_rec["train"].append(f1_train), f1_rec["val"].append(f1_val)

        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["val"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["val"], mode="acc", out_dir=log_dir)
        plot_line(plt_x, f1_rec["train"], plt_x, f1_rec["val"], mode="f1", out_dir=log_dir)

        if best_f1 < f1_val:
            best_f1 = f1_val
            best_epoch = epoch + 1

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_f1': best_f1,
                'best_epoch': best_epoch,
            }

            checkpoint_path = os.path.join(log_dir, 'checkpoint_best.pkl')
            xm.save(checkpoint, checkpoint_path)
            xm.master_print('~~~ Best Model Saved. Best F1: {}, Best Epoch: {}. ~~~'.format(best_f1, best_epoch))
            if log_file is not None:
                with open(log_file, 'a') as f:
                    xm.master_print('~~~ Best Model Saved. Best F1: {}, Best Epoch: {}. ~~~'.format(best_f1, best_epoch), fd=f)

    xm.master_print(" done ~~~~ {}, best f1: {} in :{} epochs. ".format(datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M'),
                                                               best_f1, best_epoch))
    if log_file is not None:
        with open(log_file, 'a') as f:
            xm.master_print(" done ~~~~ {}, best f1: {} in :{} epochs. ".format(datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M'),
                                                                       best_f1, best_epoch), fd=f)

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%Y-%m-%d_%H-%M')
    xm.master_print(time_str)
    if log_file is not None:
        with open(log_file, 'a') as f:
            xm.master_print(time_str, fd=f)

if __name__ == '__main__':
    def _mp_fn(index, flags):

        a = _run()

    FLAGS = {}
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')

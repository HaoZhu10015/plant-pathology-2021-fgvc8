import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from intput.src.dataset import FGVC8_Dataset
from utils.label_smoothing import LabelSmoothingCrossEntropy
from models.effnetv2 import effnetv2_s
from models.resnet import resnext50_32x4d
from datetime import datetime
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
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
parser.add_argument('--model', type=str, default='effinetv2', help='which model to use: effinetv2, resnext')
parser.add_argument('--pretrained', type=bool, default=False, help='whether to use pretrained model')
parser.add_argument('--criterion', type=str, default='CR', help='which criterion to use: CR, LSCR')
parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer： SGD, Adam')
parser.add_argument('--num_workers', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--chkpoint', type=str, default=None, help='checkpoint')
parser.add_argument('--print_acc_iter', type=int, default=10, help='print accuracy per n iteration')
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


class FGVC8_ImageClassifier(pl.LightningModule):
    def __init__(self, num_cls, model, criterion, optimizer, lr, use_schedular=True, pretrained=False):
        super(FGVC8_ImageClassifier, self).__init__()
        self.num_cls = num_cls
        self.model_name = model
        self.criterion_name = criterion
        self.optimizer_name = optimizer
        self.lr = lr
        self.use_schedular = use_schedular
        self.pretrained = pretrained
        if opt.model == 'effinetv2':
            model = effnetv2_s(num_classes=num_cls)
        elif opt.model == 'resnext':
            if self.pretrained:
                model = resnext50_32x4d(pretrained=self.pretrained)
                model.fc = nn.Linear(model.fc.in_features, num_cls)
            else:
                model = resnext50_32x4d(num_cls=num_cls, pretrained=self.pretrained)

    def forward(self, x):
        # called with self(x)
        return self.model(x)

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        if self.optimizer_name == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        elif self.optimizer_name == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr)

        if self.use_schedular:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self(x)

        if self.criterion_name == 'CR':
            criterion = nn.CrossEntropyLoss()
        elif self.criterion_name == 'LSCR':
            criterion = LabelSmoothingCrossEntropy()

        loss = criterion(y_hat, y)

        conf_mat = np.zeros((self.num_cls, self.num_cls))
        _, predictions = torch.max(y_hat.data, 1)
        for j in range(len(y)):
            pred_i = predictions[j].cpu().numpy()
            cls_i = y[j].cpu().numpy()
            conf_mat[cls_i, pred_i] += 1
        acc = conf_mat.trace() / conf_mat.sum()

        log_values = {
            'train_loss': loss,
            'train_acc': acc
        }
        self.log_dict(log_values, prog_bar=True, on_epoch=True, sync_dist=True)

        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)

        conf_mat = np.zeros((self.num_cls, self.num_cls))
        _, predictions = torch.max(y_hat.data, 1)
        for j in range(len(y)):
            pred_i = predictions[j].cpu().numpy()
            cls_i = y[j].cpu().numpy()
            conf_mat[cls_i, pred_i] += 1
        acc = conf_mat.trace() / conf_mat.sum()

        log_values = {
            'val_loss': loss,
            'val_acc': acc
        }
        self.log_dict(log_values, prog_bar=True, on_epoch=True, sync_dist=True)


checkpoint_callback = ModelCheckpoint(
    dirpath=log_dir,
    filename='{epoch}-{val_acc:.2f}',
    monitor='val_acc',
    mode='max',
    save_top_k=1,
    save_weights_only=True,
    # every_n_val_epochs=1
    period=1
)

if __name__ == '__main__':
    img_cls = {'complex': 0, 'frog_eye_leaf_spot': 1, 'healthy': 2, 'powdery_mildew': 3, 'rust': 4, 'scab': 5}
    num_cls = len(img_cls)

    data_dir = os.path.join(input_path, kaggle_prefix, 'img_aug')

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    train_txt_path = os.path.join(output_path, 'train.txt')
    val_txt_path = os.path.join(output_path, 'val.txt')

    train_dataset = FGVC8_Dataset(train_txt_path, transform)
    val_dataset = FGVC8_Dataset(val_txt_path, transform)

    train_dl = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dl = DataLoader(dataset=val_dataset, batch_size=int(opt.batch_size / 4), num_workers=opt.num_workers)

    if opt.chkpoint is not None and not opt.pretrained:
        image_classifier = FGVC8_ImageClassifier(num_cls=num_cls, model=opt.model, criterion=opt.criterion,
                                                optimizer=opt.optimizer, lr=opt.learning_rate,
                                                use_schedular=True, pretrained=opt.pretrained).load_from_checkpoint(opt.chkpoint)
    else:
        image_classifier = FGVC8_ImageClassifier(num_cls=num_cls, model=opt.model, criterion=opt.criterion,
                                                optimizer=opt.optimizer, lr=opt.learning_rate,
                                                use_schedular=True, pretrained=opt.pretrained)

    trainer = pl.Trainer(tpu_cores=8, max_epochs=opt.max_epoch, callbacks=[checkpoint_callback])
    # trainer = pl.Trainer(max_epochs=opt.max_epoch, callbacks=[checkpoint_callback])
    trainer.fit(image_classifier, train_dl, val_dl)


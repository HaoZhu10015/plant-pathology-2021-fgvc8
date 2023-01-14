import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.effnetv2 import effnetv2_s
from models.mobilenetv3 import mobilenet_v3_large
from models.resnet import resnext50_32x4d, wide_resnet101_2
from models.CBAM.model_resnet import *
from models.resnext_wsl import *
import cv2
import argparse
import pandas as pd
import copy
from utils.ttach.wrappers import ClassificationTTAWrapper
from utils.ttach import aliases as tta

src_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.dirname(src_path)
root_path = os.path.dirname(input_path)
output_path = os.path.join(root_path, 'working')
data_path = os.path.join(input_path, 'plant-pathology-2021-fgvc8')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='effinetv2', help='which model to use: '
                                                                   'effinetv2'
                                                                   'mobilenetv3'
                                                                   'resnext50_32x4d'
                                                                   'wide_resnet101_2'
                                                                   'cbam_resnet101'
                                                                   'cbam_resnet50'
                                                                   'resnext101_32x16d')
parser.add_argument('--batch_size', type=int, default=2, help='size of each image batch')
parser.add_argument('--num_workers', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--BC1_chkpoint', type=str, default=None, help='checkpoint')
parser.add_argument('--FG_chkpoint', type=str, default=None, help='checkpoint')
parser.add_argument('--BC2_chkpoint', type=str, default=None, help='checkpoint')
parser.add_argument('--MC_chkpoint', type=str, default=None, help='checkpoint')
parser.add_argument('--resize_x', type=int, default=224, help='image resize size for x')
parser.add_argument('--resize_y', type=int, default=224, help='image resize size for y')
parser.add_argument('--use_tta', action='store_true', help='whether to use TTA')
parser.add_argument('--tta_crop_x', type=int, default=150, help='tta crop size for x')
parser.add_argument('--tta_crop_y', type=int, default=150, help='tta crop size for y')
parser.add_argument('--parallel', action='store_true', help='DataParallel')
opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TestDataset_S1(Dataset):
    def __init__(self, data_txt_path, transform=None):
        self.data_txt_path = data_txt_path
        self.transform = transform
        with open(self.data_txt_path, 'r') as f:
            self.data_path_list = f.readlines()
            self.data_path_list = [d.split()[0] for d in self.data_path_list]

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, item):
        img = cv2.imread(self.data_path_list[item])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)

        return img

class TestDataset_S2(Dataset):
    def __init__(self, data_txt_path, transform=None):
        self.data_txt_path = data_txt_path
        self.transform = transform
        with open(self.data_txt_path, 'r') as f:
            self.data_path_list = f.readlines()
            self.data_path_list = [d.split()[0] for d in self.data_path_list]

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, item):
        img = cv2.imread(self.data_path_list[item])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 15), -4, 128)
        if self.transform is not None:
            img = self.transform(img)

        return img


if __name__ == '__main__':
    test_dir_path = os.path.join(data_path, 'test_images')
    data_list_path = os.path.join(data_path, 'sample_submission.csv')
    test_data_df = pd.read_csv(data_list_path)
    submission_data_df = copy.deepcopy(test_data_df)

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((opt.resize_x, opt.resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # STEP 1: Binary Classification step_1: healthy or not
    BC1_img_cls = {'unhealthy': 0, 'healthy': 1}
    BC1_num_cls = len(BC1_img_cls)
    BC1_label_to_cls = {}
    for k, v in BC1_img_cls.items():
        BC1_label_to_cls[v] = k

    test_data_list = test_data_df.iloc[:, 0]
    test_txt_step1 = os.path.join(output_path, 'test_txt_step1.txt')
    if os.path.exists(test_txt_step1):
        os.remove(test_txt_step1)
    with open(test_txt_step1, 'w') as f:
        for i in test_data_list:
            f.write(os.path.join(test_dir_path, i) + ' \n')

    test_ds = TestDataset_S1(test_txt_step1, transform)
    test_dl = DataLoader(dataset=test_ds, batch_size=opt.batch_size, num_workers=opt.num_workers)

    if opt.model == 'effinetv2':
        model = effnetv2_s(num_classes=BC1_num_cls)
    elif opt.model == 'mobilenetv3':
        model = mobilenet_v3_large(num_classes=BC1_num_cls)
    elif opt.model == 'resnext50_32x4d':
        model = resnext50_32x4d(num_classes=BC1_num_cls)
    elif opt.model == 'wide_resnet101_2':
        model = wide_resnet101_2(num_classes=BC1_num_cls)
    elif opt.model == 'cbam_resnet101':
        model = ResidualNet('ImageNet', 101, BC1_num_cls, 'CBAM')
    elif opt.model == 'cbam_resnet50':
        model = ResidualNet('ImageNet', 50, BC1_num_cls, 'CBAM')
    elif opt.model == 'resnext101_32x16d':
        model = resnext101_32x16d_wsl(num_classes=BC1_num_cls)

    if opt.parallel:
        model = nn.DataParallel(model)
    if opt.BC1_chkpoint is not None:
        checkpoint = torch.load(opt.BC1_chkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    if opt.use_tta:
        model = ClassificationTTAWrapper(model, tta.five_crop_transform(opt.tta_crop_x, opt.tta_crop_y))
        print('TTA activated.')
    model.to(device)
    model.eval()

    pred_list = []
    for data in test_dl:
        inputs = data.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, predictions = torch.max(outputs.data, 1)

        pred_list += [BC1_label_to_cls[p] for p in list(predictions.cpu().numpy())]

    test_data_df.iloc[:, -1] = pred_list
    submission_data_df.iloc[:, -1] = pred_list

    # STEP 2: Mutiple Classification
    MC_img_cls = {'complex': 0, 'frog_eye_leaf_spot': 1, 'powdery_mildew': 2, 'rust': 3, 'scab': 4}
    MC_num_cls = len(MC_img_cls)
    MC_label_to_cls = {}
    for k, v in MC_img_cls.items():
        MC_label_to_cls[v] = k

    test_data_list = test_data_df.iloc[:, 0][test_data_df.iloc[:, 1] == 'unhealthy']
    test_txt_step2 = os.path.join(output_path, 'test_txt_step2.txt')
    if os.path.exists(test_txt_step2):
        os.remove(test_txt_step2)
    with open(test_txt_step2, 'w') as f:
        for i in test_data_list:
            f.write(os.path.join(test_dir_path, i) + ' \n')

    test_ds = TestDataset_S2(test_txt_step2, transform)
    test_dl = DataLoader(dataset=test_ds, batch_size=opt.batch_size, num_workers=opt.num_workers)

    if opt.model == 'effinetv2':
        model = effnetv2_s(num_classes=MC_num_cls)
    elif opt.model == 'mobilenetv3':
        model = mobilenet_v3_large(num_classes=MC_num_cls)
    elif opt.model == 'resnext50_32x4d':
        model = resnext50_32x4d(num_classes=MC_num_cls)
    elif opt.model == 'wide_resnet101_2':
        model = wide_resnet101_2(num_classes=MC_num_cls)
    elif opt.model == 'cbam_resnet101':
        model = ResidualNet('ImageNet', 101, MC_num_cls, 'CBAM')
    elif opt.model == 'cbam_resnet50':
        model = ResidualNet('ImageNet', 50, MC_num_cls, 'CBAM')
    elif opt.model == 'resnext101_32x16d':
        model = resnext101_32x16d_wsl(num_classes=MC_num_cls)

    if opt.parallel:
        model = nn.DataParallel(model)
    if opt.MC_chkpoint is not None:
        checkpoint = torch.load(opt.MC_chkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    if opt.use_tta:
        model = ClassificationTTAWrapper(model, tta.five_crop_transform(opt.tta_crop_x, opt.tta_crop_y))
        print('TTA activated.')
    model.to(device)
    model.eval()

    pred_list = []
    threshold = {0: 0.5,
                 1: 0.5,
                 2: 0.5,
                 3: 0.5,
                 4: 0.5}
    for data in test_dl:

        inputs = data.to(device)
        with torch.no_grad():
            outputs = model(inputs)

        predictions = np.array(torch.sigmoid(outputs.data).cpu().numpy() >= 0.5, dtype=int)

        for p in predictions:
            if p.sum() != 0:
                pred_list.append(' '.join([MC_label_to_cls[l] for l in np.where(p == 1)[0]]))
            else:
                pred_list.append('healthy')

    test_data_df.iloc[:, -1][test_data_df.iloc[:, -1] == 'unhealthy'] = pred_list
    submission_data_df.iloc[:, -1][submission_data_df.iloc[:, -1] == 'unhealthy'] = pred_list

    submission_file_path = os.path.join(output_path, 'submission.csv')
    if os.path.exists(submission_file_path):
        os.remove(submission_file_path)
    submission_data_df.to_csv(submission_file_path, index=None)
import os
import random
# random.seed(10)
import argparse

src_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.dirname(src_path)
root_path = os.path.dirname(input_path)
output_path = os.path.join(root_path, 'working')

parser = argparse.ArgumentParser()
parser.add_argument('--kaggle', action='store_true', help='if the script is run on kaggle')
parser.add_argument('--drop_prob', type=float, default=0.5, help='drop probability')
opt = parser.parse_args()

kaggle_prefix = 'fgvc8-roger10015-img-aug-bc2' if opt.kaggle else ''

if __name__ == '__main__':
    data_path = os.path.join(input_path, kaggle_prefix, 'img_aug')
    data_list = []
    for root, dirs, files in os.walk(data_path):
        label = root.split('/')[-1][0]

        for file in files:
            if label == '0' and random.random() > opt.drop_prob:
                continue
            if file.endswith('jpg'):
                img_path = os.path.join(root, file)
                data_list.append(img_path + ' ' + label + '\n')

    random.shuffle(data_list)
    split_idx = int(len(data_list) * 0.9)

    train_data = data_list[:split_idx]
    val_data = data_list[split_idx:]

    train_txt_pth = os.path.join(output_path, 'train.txt')
    if os.path.exists(train_txt_pth):
        os.remove(train_txt_pth)
    val_txt_pth = os.path.join(output_path, 'val.txt')
    if os.path.exists(val_txt_pth):
        os.remove(val_txt_pth)

    with open(train_txt_pth, 'w') as f:
        f.writelines(train_data)
    with open(val_txt_pth, 'w') as f:
        f.writelines(val_data)
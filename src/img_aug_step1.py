import os
import numpy as np
import cv2
import csv
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import random
import argparse

src_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.dirname(src_path)
data_path = os.path.join(input_path, 'plant-pathology-2021-fgvc8')
img_path = os.path.join(data_path, 'train_images')
root_path = os.path.dirname(input_path)
output_path = os.path.join(root_path, 'working')

parser = argparse.ArgumentParser()
parser.add_argument('--kaggle', action='store_true', help='if the script is run on kaggle')
parser.add_argument('--local', action='store_true', help='if save output to local desktop')
opt = parser.parse_args()

if __name__ == '__main__':
    img_cls = ['complex', 'frog_eye_leaf_spot', 'healthy', 'powdery_mildew', 'rust', 'scab']
    img_aug_cls = {'healthy': 3}  # class: fach

    if opt.kaggle:
        img_aug_save_path = os.path.join(output_path, 'img_aug')
    elif opt.local:
        img_aug_save_path = os.path.join('C:\\Users\\Public\\Desktop\\', 'img_aug')
    else:
        img_aug_save_path = os.path.join(input_path, 'img_aug')
    if not os.path.exists(img_aug_save_path):
        os.mkdir(img_aug_save_path)

    datagen = ImageDataGenerator(
        rotation_range=20,
        fill_mode='constant',
        height_shift_range=0.1,
        width_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.2
    )

    reader = csv.reader(open(os.path.join(data_path, 'train.csv')))
    next(reader)
    reader = list(reader)

    for row in tqdm(reader):
        label = row[-1]

        if len(label) > 0 and label.find(' ') == -1:
            img_file_path = os.path.join(img_path, row[0])
            img = cv2.imread(img_file_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (256, 256))  # !!!sehr wichtig fuer ImageDataGenerator.flow()
                # img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 15), -4, 128)

                if label not in img_aug_cls.keys():
                    save_pth = os.path.join(img_aug_save_path, str(0) + 'unhealthy')
                    if not os.path.exists(save_pth):
                        os.mkdir(save_pth)
                    cv2.imwrite(os.path.join(save_pth, row[0]), img)
                else:
                    save_pth = os.path.join(img_aug_save_path, str(1) + 'healthy')
                    if not os.path.exists(save_pth):
                        os.mkdir(save_pth)
                    cv2.imwrite(os.path.join(save_pth, row[0]), img)
                    img_gen = datagen.flow(
                        np.expand_dims(np.asarray(img), 0),
                        [label],
                        save_to_dir=save_pth,
                        save_format='jpg',
                        batch_size=1
                    )
                    n = 1
                    while True:
                        if n > img_aug_cls[label]:
                            break
                        next(img_gen)
                        n += 1
            else:
                print('Image {} not found.'.format(row[0]))

    data_list = []
    for root, dirs, files in os.walk(img_aug_save_path):
        for d in dirs:
            print(d + ': ' + str(len(os.listdir(os.path.join(root, d)))))

        for file in files:
            if file.endswith('.jpg'):
                label = list(os.path.basename(root))[0]
                data_list.append(file + ',' + label + '\n')

    train_csv_path = os.path.join(img_aug_save_path, 'train.csv')
    val_csv_path = os.path.join(img_aug_save_path, 'val.csv')
    for prefix in ['train', 'val']:
        if os.path.exists(eval('{}_csv_path'.format(prefix))):
            os.remove(eval('{}_csv_path'.format(prefix)))
        with open(eval('{}_csv_path'.format(prefix)), 'w') as f:
            f.write('image,label' + '\n')

    random.shuffle(data_list)
    data_split = int(len(data_list) * 0.9)

    for prefix in ['train', 'val']:
        with open(eval('{}_csv_path'.format(prefix)), 'a') as f:
            if prefix == 'train':
                f.writelines(data_list[:data_split])
            else:
                f.writelines(data_list[data_split:])
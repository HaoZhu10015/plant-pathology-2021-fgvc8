import os
import numpy as np
import cv2
import csv
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import argparse

src_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.dirname(src_path)
data_path = os.path.join(input_path, 'plant-pathology-2021-fgvc8')
img_path = os.path.join(data_path, 'train_images')
root_path = os.path.dirname(input_path)
output_path = os.path.join(root_path, 'working')

parser = argparse.ArgumentParser()
parser.add_argument('--kaggle', action='store_true', help='if the script is run on kaggle')
opt = parser.parse_args()

if __name__ == '__main__':
    img_cls = ['complex', 'frog_eye_leaf_spot', 'healthy', 'powdery_mildew', 'rust', 'scab']
    img_aug_fach = 3

    if opt.kaggle:
        img_aug_save_path = os.path.join(output_path, 'img_aug')
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

        if len(label) > 0 and label != 'healthy':
            img_file_path = os.path.join(img_path, row[0])
            img = cv2.imread(img_file_path)
            if img is not None:
                img = cv2.resize(img, (512, 512))  # !!!sehr wichtig fuer ImageDataGenerator.flow()
                img = np.asarray(img)

                if label.find(' ') == -1:
                    save_pth = os.path.join(img_aug_save_path, str(0) + 'simple_disease')
                    if not os.path.exists(save_pth):
                        os.mkdir(save_pth)
                    cv2.imwrite(os.path.join(save_pth, row[0]), img)
                else:
                    save_pth = os.path.join(img_aug_save_path, str(1) + 'complex_disease')
                    if not os.path.exists(save_pth):
                        os.mkdir(save_pth)
                    cv2.imwrite(os.path.join(save_pth, row[0]), img)
                    img_gen = datagen.flow(
                        np.expand_dims(img, 0),
                        [label],
                        save_to_dir=save_pth,
                        save_format='jpg',
                        batch_size=1
                    )
                    n = 1
                    while True:
                        if n > img_aug_fach:
                            break
                        next(img_gen)
                        n += 1
            else:
                print('Image {} not found.'.format(row[0]))
    for root, dirs, files in os.walk(img_aug_save_path):
        for d in dirs:
            print(d + ': ' + str(len(os.listdir(os.path.join(root, d)))))


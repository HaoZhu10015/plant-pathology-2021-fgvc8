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
    img_cls = {'complex': 0, 'frog_eye_leaf_spot': 1, 'powdery_mildew': 2, 'rust': 3, 'scab': 4}
    multiclass_aug_fach = 3
    img_aug_cls = {'complex': 1, 'powdery_mildew': 3, 'rust': 1}  # class: fach

    if opt.kaggle:
        save_path = os.path.join(output_path, 'img_aug')
    else:
        save_path = os.path.join(input_path, 'img_aug')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

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
        prefix = [0, 0, 0, 0, 0]
        if row[-1] != 'healthy':
            for l in row[-1].split():
                prefix[img_cls[l]] += 1
        prefix = ''.join(list(map(str, prefix)))

        img_file_path = os.path.join(img_path, row[0])
        img = cv2.imread(img_file_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            img = cv2.resize(img, (560, 560))
            cv2.imwrite(os.path.join(save_path, prefix + '_' + row[0]), img)

            if row[-1].find(' ') != -1:
                fach = multiclass_aug_fach
            elif row[-1] in img_aug_cls:
                fach = img_aug_cls[row[-1]]
            else:
                continue

            img_gen = datagen.flow(
                np.expand_dims(np.asarray(img), 0),
                [row[-1]],
                save_to_dir=save_path,
                save_prefix=prefix,
                save_format='jpg',
                batch_size=1
            )
            n = 1
            while True:
                if n > fach:
                    break
                next(img_gen)
                n += 1
        else:
            print('Image {} not found.'.format(row[0]))

    print(len(os.listdir(save_path)))


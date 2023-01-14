import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
from sklearn.preprocessing import LabelEncoder


src_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.dirname(src_path)
data_path = os.path.join(input_path, 'plant-pathology-2021-fgvc8')
img_path = os.path.join(data_path, 'train_images')
root_path = os.path.dirname(input_path)
output_path = os.path.join(root_path, 'working')

if __name__ == '__main__':
    data_stat_log_pth = os.path.join(output_path, 'data_statistic.txt')

    reader = csv.reader(open(os.path.join(data_path, 'train.csv')))
    next(reader)

    inputs = []
    labels = []

    for row in reader:
        label = row[-1]
        if len(label) > 0 and label.find(' ') == -1:
            img_file_path = os.path.join(img_path, row[0])
            # img = cv2.imread(img_file_path)
            # img = cv2.resize(img, (640, 480))  # !!!sehr wichtig fuer ImageDataGenerator.flow()
            img = 1  # Wegen des kleinen RAM meines MacBooks
            if img is not None:
                img_arr = np.asarray(img)
                inputs.append(img_arr)
                labels.append(label)
            else:
                print('Image {} not found.'.format(row[0]))

    inputs = np.asarray(inputs)
    labels = np.asarray(labels)

    encoder = LabelEncoder()
    encoder.fit(labels)
    print(encoder.classes_)
    with open(data_stat_log_pth, 'w') as f:
        print(encoder.classes_, '\n', file=f)

    num_cls = len(encoder.classes_)

    encoded_labels = encoder.transform(labels)

    counts = np.bincount(encoded_labels)
    print(counts)
    with open(data_stat_log_pth, 'a') as f:
        print(counts, '\n', file=f)

    percentages = 100 * counts / sum(counts)
    print(percentages)
    with open(data_stat_log_pth, 'a') as f:
        print(percentages, '\n', file=f)

    fig, ax = plt.subplots()
    plt.bar(list(range(num_cls)), percentages)
    ax.set_xticklabels([''] + list(encoder.classes_))
    ax.set_ylabel('Percentage')
    plt.savefig(os.path.join(output_path, 'data_statistic.png'))
import argparse
import os
import time

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from tabulate import tabulate

from utils import rgb2ind, ind2rgb
from metrics import CM
from predict_config import pad_to

def printing(classes, cm, columns, plot_cm):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # Plot confusion matrices
    if plot_cm:
        cms = [cm, cm / cm.sum(axis=0, keepdims=True), cm / cm.sum(axis=1, keepdims=True)]
        titles = [
            'Original',
            'What classes are responsible for each classification',
            'How each class has been classified'
        ]
        for t, confusion_matrix in enumerate(cms):
            fig, ax = plt.subplots()
            ax.matshow(confusion_matrix)
            ax.set_title(titles[t])
            ax.set_xlabel('Predicted class')
            ax.set_ylabel('True class')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xticks(list(range(len(classes))))
            ax.set_yticks(list(range(len(classes))))
            ax.set_xticklabels(classes, rotation='vertical')
            ax.set_yticklabels(classes)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    c = round(confusion_matrix[j, i], 2)
                    ax.text(i, j, str(c), va='center', ha='center')
            cm_file = (f'results/cm{str(t)}_{timestr}')
            plt.savefig(cm_file, bbox_inches='tight')
            #plt.show()

    metrics_file = (f'results/metrics_{timestr}.txt')
    # Compute metrics
    f = open(metrics_file, 'w')
    table = []

    table.append(['Accuracy', np.diag(cm).sum() / cm.sum()])
    cm_nobg = np.copy(cm)
    cm_nobg[0] = 0
    table.append(['Accuracy no bg', np.diag(cm_nobg).sum() / cm_nobg.sum()])
    for i, c in enumerate(classes):
        if c == 'Background':
            continue
        table.append(
            [c] + [column[i] for column in columns[:-1]] + [np.array([metrics[i] for metrics in columns[-1]]) / sum(
                [metrics[i] for metrics in columns[-1]])])
    table.append(['Mean'] + [np.nanmean(column) for column in columns[:-1]] + [
        np.array(np.sum(columns[-1], axis=1)) / np.sum(columns[-1], axis=1).sum()])
    table.append(['Mean no bg'] + [np.nanmean(column[1:]) for column in columns[:-1]] + [
        np.array(np.sum(columns[-1][1:], axis=1)) / np.sum(columns[-1][1:], axis=1).sum()])
    f.write(
        tabulate(table, headers=['Class', 'Class accuracy', 'Recall', 'Precision', 'F1', 'JI','TP, FP, TN, FN']))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for evaluation metrics on dataset folder')
    parser.add_argument('--root_path', '-p', help='Path to the dataset', default='./Data/')
    parser.add_argument('--plot', action='store_true', help='Plot confusion matrix flag')

    args = parser.parse_args()

    classes = ['Background', 'Walls', 'Glass_walls', 'Railings', 'Doors', 'Sliding_doors', 'Windows', 'Stairs_all']
    metric = CM(len(classes))

    subdirs = list()
    for d in os.listdir(args.root_path):
        if os.path.isdir(os.path.join(args.root_path, d)):
            subdirs.append(d)
    print('Data samples: ', subdirs)
    pbar = tqdm(subdirs, "Processing")
    for subdir in pbar:
        subpath = os.path.join(args.root_path, subdir)
        mask_name = ''
        prediction_name = ''
        dir_files = os.listdir(subpath)
        for f in dir_files:
            ending = f.split('.')[-1]

            if ending == 'png' and 'result' in f:
                prediction_name = f
                pbar.set_description(f'Found: {prediction_name} file')
            elif f == 'mask.png':
                mask_name = f
                pbar.set_description(f'Found: {f} file')
                
        if prediction_name == '':
            print(f'Sample: {subdir} - No prediction file - skip')
            continue
        elif mask_name == '':
            print(f'Sample: {subdir} - No mask file - skip')
            continue
  
        mask, _ = pad_to(ind2rgb(cv2.imread(os.path.join(subpath, mask_name), cv2.IMREAD_GRAYSCALE).astype(np.uint8)), 32)
        prediction, _ = pad_to(np.array(Image.open(os.path.join(subpath, prediction_name)).convert("RGB")), 32)
        if mask.shape != prediction.shape:
            h, w = mask.shape[:2]
            prediction = cv2.resize(prediction, dsize=(h, w), interpolation=cv2.INTER_CUBIC)
        
        metric.update_state(rgb2ind(mask.numpy()), rgb2ind(prediction.numpy()))

    printing(classes, metric.result().numpy(), metric.cm_metrics(), args.plot)
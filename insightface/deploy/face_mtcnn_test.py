import face_model
import argparse
import cv2
import glob
import sys
import numpy as np
import os
import time
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r100-ii/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)
start = time.time()
cnt = 0
'''
lr_path = '/home/jiaheng/Desktop/GitHub/IJB-C-1/LR'
for dir_path, dirs, files in os.walk(lr_path):
    if len(dir_path.split('/')) == 8:
        cnt += 1
        print('Time:', time.time() - start)
        print('No.', cnt)
        print(dir_path)
        print('--------------------------------')
    test_img_list = []
    test_filename_list = []
    for file_name in files:
        img_path = os.path.join(dir_path, file_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        resized_img = cv2.resize(img, (224, 224))
        test_img_list.append(resized_img)
        test_filename_list.append(file_name)
    if len(test_img_list) == 0:
        continue
    test_img_list = np.array(test_img_list, copy=False, dtype=np.float32)
    scores = model.predict(test_img_list, batch_size=1, verbose=1)
    highest_score_index = scores.argmax()
    highest_score_img = test_img_list[highest_score_index]
    highest_score_filename = test_filename_list[highest_score_index]
    highest_dir = dir_path.replace('LR', 'LR_highest')
    os.makedirs(highest_dir, exist_ok=True)
    cv2.imwrite(os.path.join(highest_dir, highest_score_filename), highest_score_img)
'''
hr_path = '/home/jiaheng/Desktop/GitHub/IJB-C-1/GT'
for img_path in glob.glob(hr_path + "/*/img/*.jpg", recursive=True):
    cnt += 1
    print('Time:', time.time() - start)
    print('No.', cnt)
    print(img_path)
    print('--------------------------------')
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    if model.get_input(img) is None:
        continue
    print(os.path.dirname(img_path))
    highest_dir = os.path.dirname(img_path).replace('GT', 'GT_mtcnn')
    os.makedirs(highest_dir, exist_ok=True)
    highest_score_filename = img_path.split('/')[-1]
    print(highest_score_filename)
    cv2.imwrite(os.path.join(highest_dir, highest_score_filename), img)
print('Extract the images with highest score successfully!')


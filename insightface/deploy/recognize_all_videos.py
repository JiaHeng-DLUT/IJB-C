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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

def get_embedding(img_path):
    img_cv2 = cv2.imread(img_path)
    img = model.get_input(img_cv2)
    if img is None:
        return None
    return model.get_feature(img)

def predict_id(f1):
    min_dist = 1e9
    id_predicted = 0
    for i in range(len(template_embedding_list)):
        f2 = template_embedding_list[i]
        dist = np.sum(np.square(f1 - f2))
        if (dist < min_dist):
            min_dist = dist
            id_predicted = template_id_list[i]
    return id_predicted

template_embedding_list = []
template_id_list = []
'''
hr_path = '/home/jiaheng/Desktop/GitHub/IJB-C-1/GT_highest_mtcnn'
for img_path in glob.glob(hr_path + "/*/img/*.jpg", recursive=True):
    print(img_path.split('/'))
    id = img_path.split('/')[-3]
    f2 = get_embedding(img_path)
    if f2 is not None:
        template_embedding_list.append(f2)
        template_id_list.append(id)
    print('Num of template embeddings:', len(template_embedding_list))
'''

base_dir = '/home/jiaheng/Desktop/GitHub/IJB-C-1/GT'
f = open('template_fq.txt','r')
out = f.readlines()
for line in out:
    _ = line.split(' ')
    # print(_)
    id = _[0]
    _ = _[1].split('/')
    filename = _[-1]
    # print(filename)
    filename = filename.replace('txt', 'jpg').rstrip('\n')
    img_path = os.path.join(base_dir, str(id)) + '/img' + '/' + filename
    print(img_path)
    # img_cv2 = cv2.imread(img_path)
    # print(img_cv2.shape)
    # cv2.imshow('', img_cv2)
    # cv2.waitKey(0)
    f2 = get_embedding(img_path)
    if f2 is not None:
        template_embedding_list.append(f2)
        template_id_list.append(id)
    print('Num of template embeddings:', len(template_embedding_list))
f.close()

cnt_all = 0
cnt = 0
correct = 0
'''
lr_path = '/home/jiaheng/Desktop/GitHub/IJB-C-1/LR_deblur'
for img_path in glob.glob(lr_path + "/*/*/*.jpg", recursive=True):
    _ = img_path.split('/')
    # print(_)
    id = _[-3]
    video_id = _[-2]
    if video_id == 'img':
        continue
'''
base_dir = '/home/jiaheng/Desktop/GitHub/IJB-C-1/LR'
f = open('lr.txt','r')
out = f.readlines()
for line in out:
    _ = line.split(' ')
    # print(_)
    id = _[0]
    img_path = _[1].rstrip("\n")
    print(img_path)
    print("True ID:", id)
    cnt_all += 1
    print('Num of all imgaes:', cnt_all)
    f1 = get_embedding(img_path)
    if f1 is None:
        continue
    cnt += 1
    print('Num of valid images:', cnt)
    id_predicted = predict_id(f1)
    print("Predicted ID:", id_predicted)
    if id_predicted == id:
        correct += 1
    accuracy = correct / cnt
    print('Correct:', correct)
    print('Accuracy:', accuracy)    
    print('Time:', time.time() - start)
    print('--------------------------------------')
f.close()
print('Num of template embeddings:', len(template_embedding_list))

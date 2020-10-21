import face_model
import argparse
import cv2
import sys
import numpy as np
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
'''
img = cv2.imread('Tom_Hanks_54745.png')
print(img.shape)
img = model.get_input(img)
print(img.shape)
f1 = model.get_feature(img)
img = cv2.imread('Tom_Hanks_54745.png')
img = model.get_input(img)
f2 = model.get_feature(img)
dist = np.sum(np.square(f1-f2))
print(dist)
sim = np.dot(f1, f2.T)
print(sim)
#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)
'''
start = time.time()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
num_workers = 4

def collate_fn(x):
    # print(x)
    return x

# template
template_dataset = datasets.ImageFolder('../../template')
# print(type(template_dataset))
# print(type(template_dataset.class_to_idx))
# print(template_dataset.class_to_idx)
# print(type(template_dataset.class_to_idx.items()))
template_dataset.idx_to_class = { i: c for c, i in template_dataset.class_to_idx.items() }
template_loader = DataLoader(template_dataset, batch_size=1, collate_fn=collate_fn, num_workers=num_workers)
test_dataset = datasets.ImageFolder('../../test')
test_dataset.idx_to_class = { i: c for c, i in test_dataset.class_to_idx.items() }
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, num_workers=num_workers)
template_embedding_list = []
template_id_list = []

def recognize(f1):
    min_dist = 1e9
    id_predicted = 0
    if len(template_embedding_list) > 0:
        for i in range(len(template_embedding_list)):
            f2 = template_embedding_list[i]
            dist = np.sum(np.square(f1-f2))
            if (dist < min_dist):
                min_dist = dist
                id_predicted = template_id_list[i]
        return id_predicted
    cnt = 0
    for batch in template_loader:
        print(batch)
        '''
        if cnt >= 100:
            break
        '''
        cnt += 1
        for (img_PIL, i) in batch:
            img_cv2 = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
            img_cv2 = cv2.resize(img_cv2, (112, 112))
            img = model.get_input(img_cv2)
            if img is None:
                continue
            f2 = model.get_feature(img)
            template_embedding_list.append(f2)
            template_id_list.append(template_dataset.idx_to_class[i])
            dist = np.sum(np.square(f1-f2))
            if (dist < min_dist):
                min_dist = dist
                id_predicted = template_dataset.idx_to_class[i]
    return id_predicted

cnt = 0
correct = 0
for batch in test_loader:
    cnt += 1
    print("No.", cnt)
    for (img_PIL, i) in batch:
        print("ID:", test_dataset.idx_to_class[i])
        img_cv2 = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
        img_cv2 = cv2.resize(img_cv2, (112, 112))
        print(img_cv2.shape)
        img = model.get_input(img_cv2)
        print(img.shape)
        if img is None:
            cnt -= 1
            continue
        f1 = model.get_feature(img)
        id_predicted = recognize(f1)
        print("Predicted ID:", id_predicted)
        id_true = test_dataset.idx_to_class[i]
        if id_predicted == id_true:
            correct += 1
    accuracy = correct / cnt
    print('Accuracy:', accuracy)    
    print('Time:', time.time() - start)
    print('--------------------------------------')

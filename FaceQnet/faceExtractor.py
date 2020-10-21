import cv2
import numpy as np
import os
import time
# === DON'T DELETE ===
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# === DON'T DELETE ===
from keras.models import load_model

model = load_model('./FaceQnet_v1.h5')
print("Load [FaceQnet_v1.h5] successfully!")
start = time.time()
cnt = 0

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
    
"""
hr_path = '/home/jiaheng/Desktop/GitHub/IJB-C-1/GT_mtcnn'
for dir_path, dirs, files in os.walk(hr_path):
    if len(dir_path.split('/')) == 9 and dir_path.split('/')[-1] == 'img':
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
            #resized_img = cv2.resize(img, (224, 224))
            resized_img = img
            test_img_list.append(resized_img)
            test_filename_list.append(file_name)
        if len(test_img_list) == 0:
            continue
        test_img_list = np.array(test_img_list, copy=False, dtype=np.float32)
        scores = model.predict(test_img_list, batch_size=1, verbose=1)
        highest_score_index = scores.argmax()
        highest_score_img = test_img_list[highest_score_index]
        highest_score_filename = test_filename_list[highest_score_index]
        highest_dir = dir_path.replace('GT', 'GT_highest')
        os.makedirs(highest_dir, exist_ok=True)
        cv2.imwrite(os.path.join(highest_dir, highest_score_filename), highest_score_img)
"""
print('Extract the images with highest score successfully!')

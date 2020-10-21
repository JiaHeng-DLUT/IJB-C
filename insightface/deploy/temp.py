import os
import cv2
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
    img_cv2 = cv2.imread(img_path)
    # print(img_cv2.shape)
    # cv2.imshow('', img_cv2)
    # cv2.waitKey(0)
    break

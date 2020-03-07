from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os.path as osp
from models import EfficientDet
from datasets import get_augumentation
from utils import EFFICIENTDET

import cv2
import copy
import time
import numpy as np
from PIL import Image
import scipy.io as sio


def get_data():
    subset = 'val'
    if subset is 'val':
        wider_face = sio.loadmat(
            '../WIDER_FACE/wider_face_split/wider_face_val.mat')
    else:
        wider_face = sio.loadmat(
            '../WIDER_FACE/wider_face_split/wider_face_test.mat')
    event_list = wider_face['event_list']
    file_list = wider_face['file_list']
    del wider_face

    imgs_path = os.path.join(
        '../WIDER_FACE/', 'WIDER_{}'.format(subset), 'images')
    save_path = 'eval_tools/efficientdet-d1_{}'.format(subset)

    return event_list, file_list, imgs_path, save_path


class Detect(object):
    """
        dir_name: Folder or image_file
    """

    def __init__(self, weights, num_class=21, network='efficientdet-d1', size_image=(512, 512)):
        super(Detect,  self).__init__()
        self.weights = weights
        self.size_image = size_image
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else 'cpu')
        self.transform = get_augumentation(phase='test',width=size_image[0], height=size_image[1])
        if(self.weights is not None):
            print('Load pretrained Model')
            checkpoint = torch.load(
                self.weights, map_location=lambda storage, loc: storage)
            num_class = checkpoint['num_class']
            network = checkpoint['network']

        self.model = EfficientDet(num_classes=num_class,
                     network=network,
                     W_bifpn=EFFICIENTDET[network]['W_bifpn'],
                     D_bifpn=EFFICIENTDET[network]['D_bifpn'],
                     D_class=EFFICIENTDET[network]['D_class'],
                     is_training=False,
                     threshold=0.055
                     )

        if(self.weights is not None):
            state_dict = checkpoint['state_dict']
            self.model.load_state_dict(state_dict)
        self.model = self.model.cuda()
        self.model.eval()

    def process(self, file_name=None, img=None):
        if file_name is not None:
            img = cv2.imread(file_name)
        origin_img = copy.deepcopy(img)
        augmentation = self.transform(image=img)
        img = augmentation['image']
        img = img.to(self.device)
        img = img.unsqueeze(0)

        with torch.no_grad():
            scores, classification, transformed_anchors = self.model(img)
            bboxes = list()
            labels = list()
            bbox_scores = list()
            colors = list()
            for j in range(scores.shape[0]):
                bbox = transformed_anchors[[j], :][0].data.cpu().numpy()
                x1 = int(bbox[0]*origin_img.shape[1]/self.size_image[1])
                y1 = int(bbox[1]*origin_img.shape[0]/self.size_image[0])
                x2 = int(bbox[2]*origin_img.shape[1]/self.size_image[1])
                y2 = int(bbox[3]*origin_img.shape[0]/self.size_image[0])
                bboxes.append([x1, y1, x2, y2])
                label_name = 'face' if int(classification[[j]]) == 0 else 'not recognized'
                labels.append(label_name)

                
                score = np.around(
                    scores[[j]].cpu().numpy(), decimals=3)
                bbox_scores.append(float(score))

            return bboxes, labels, bbox_scores


if __name__ == '__main__':
    weight = './saved/weights/afterVOC_WF_efficientdet-d1_58.pth'
    detect = Detect(weights=weight,size_image=(1024,1024))
    event_list, file_list, imgs_path, save_path = get_data()
    
    counter = 0
    for index, event in enumerate(event_list):
        filelist = file_list[index][0]
        path = os.path.join(save_path, event[0][0])
        if not os.path.exists(path):
            os.makedirs(path)

        for num, file in enumerate(filelist):
            im_name = file[0][0]#.encode('utf-8')
            in_file = os.path.join(imgs_path, event[0][0], im_name[:] + '.jpg')
            #img = cv2.imread(in_file)
            img = Image.open(in_file)
            if img.mode == 'L':
                img = img.convert('RGB')
            img = np.array(img)

            counter += 1

            t1 = time.time()
            bboxes, labels, bbox_scores = detect.process(img=img)
            t2 = time.time()
            print('Detect %04d th image costs %.4f' % (counter, t2 - t1))

            fout = open(osp.join(save_path, event[0][
                        0], im_name + '.txt'), 'w')
            fout.write('{:s}\n'.format(event[0][0] + '/' + im_name + '.jpg'))
            fout.write('{:d}\n'.format(len(bbox_scores)))
            for i in range(len(bboxes)):
                xmin = bboxes[i][0]
                ymin = bboxes[i][1]
                xmax = bboxes[i][2]
                ymax = bboxes[i][3]
                score = bbox_scores[i]
                fout.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                           format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))
            fout.close()
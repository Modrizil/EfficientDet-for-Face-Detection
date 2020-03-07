import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

# VOC_CLASSES = (  # always index 0
#     'aeroplane', 'bicycle', 'bird', 'boat',
#     'bottle', 'bus', 'car', 'cat', 'chair',
#     'cow', 'diningtable', 'dog', 'horse',
#     'motorbike', 'person', 'pottedplant',
#     'sheep', 'sofa', 'train', 'tvmonitor')
WF_CLASSES = (
    'background','face'
)

# note: if you used our download scripts, this should be right
# VOC_ROOT = osp.join('/home/toandm2', "data/VOCdevkit/")
WF_ROOT = osp.join('./widerface/train/')

class WFAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(WF_CLASSES, range(len(WF_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, labels, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        name = 'face'
        if len(labels) == 0:
            return None
        for idx, label in enumerate(labels):
            if(label[2]<=0 or label[3]<=0):
                continue
            bndbox = []
            bndbox.append(label[0])
            bndbox.append(label[1])
            bndbox.append(label[0] + label[2])
            bndbox.append(label[1] + label[3])

            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)

            res += [bndbox]
        
        if len(res) == 0:
            return None

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class WFDetection(data.Dataset):
    """VOC Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 transform=None, target_transform=WFAnnotationTransform(),
                 dataset_name='WIDERFACE'):
        self.root = root # ./widerface/train/
        #self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join(self.root, 'label_clean.txt')
        txt_path = self._annopath
        # self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        # self.ids = list()
        # for (year, name) in image_sets:
        #     rootpath = osp.join(self.root, 'VOC' + year)
        #     for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        #         self.ids.append((rootpath, line.strip()))
        self.imgs_path = []
        self.words = []
        f = open(self._annopath,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label_clean.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

    def __getitem__(self, index):
        # img_id = self.ids[index]

        # target = ET.parse(self._annopath % img_id).getroot()
        # img = cv2.imread(self._imgpath % img_id)
        img = cv2.imread(self.imgs_path[index])
        height, width, channels = img.shape

        labels = self.words[index]

        if self.target_transform is not None:
            target = self.target_transform(labels, width, height)
        if target == None:
            print('No.{}  no target:{}'.format(index, self.imgs_path[index]))
            return None
        
        target = np.array(target)
        bbox = target[:, :4]
        labels = target[:, 4]
        if self.transform is not None:
            annotation = {'image': img, 'bboxes': bbox, 'category_id': labels}
            augmentation = self.transform(**annotation)
            img = augmentation['image']
            bbox = augmentation['bboxes']
            labels = augmentation['category_id']
        return {'image': img, 'bboxes': bbox, 'category_id': labels} 

    def __len__(self):
        return len(self.imgs_path)
    def __num_class__(self):
        return len(WF_CLASSES)
    def label_to_name(self, label):
        return WF_CLASSES[label]
    
    

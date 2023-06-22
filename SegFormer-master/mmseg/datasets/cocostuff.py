from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp

@DATASETS.register_module()
class CocoStuff(CustomDataset):
    """Coco Stuff dataset.
    """
    # d={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
    # nclass = len(d)#80类
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    # CLASSES=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18,
    #          19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    #          37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52
    #     , 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72
    #     , 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    # CLASSES = [str(i) for i in CLASSES]

    # random generated color
    PALETTE = [[167, 200, 7],
               [127, 228, 215],
               [26, 135, 248],
               [238, 73, 166],
               [91, 210, 215],
               [122, 20, 236],
               [234, 173, 35],
               [34, 98, 46],
               [115, 11, 206],
               [52, 251, 238],
               [209, 156, 236],
               [239, 10, 0],
               [26, 122, 36],
               [162, 181, 66],
               [26, 64, 22],
               [46, 226, 200],
               [89, 176, 6],
               [103, 36, 32],
               [74, 89, 159],
               [250, 215, 25],
               [57, 246, 82],
               [51, 156, 111],
               [139, 114, 219],
               [65, 208, 253],
               [33, 184, 119],
               [230, 239, 58],
               [176, 141, 158],
               [21, 29, 31],
               [135, 133, 163],
               [152, 241, 248],
               [253, 54, 7],
               [231, 86, 229],
               [179, 220, 46],
               [155, 217, 185],
               [58, 251, 190],
               [40, 201, 63],
               [236, 52, 220],
               [71, 203, 170],
               [96, 56, 41],
               [252, 231, 125],
               [255, 60, 100],
               [11, 172, 184],
               [127, 46, 248],
               [1, 105, 163],
               [191, 218, 95],
               [87, 160, 119],
               [149, 223, 79],
               [216, 180, 245],
               [58, 226, 163],
               [11, 43, 118],
               [20, 23, 100],
               [71, 222, 109],
               [124, 197, 150],
               [38, 106, 43],
               [115, 73, 156],
               [113, 110, 50],
               [94, 2, 184],
               [163, 168, 155],
               [83, 39, 145],
               [150, 169, 81],
               [134, 25, 2],
               [145, 49, 138],
               [46, 27, 209],
               [145, 187, 117],
               [197, 9, 211],
               [179, 12, 118],
               [107, 241, 133],
               [255, 176, 224],
               [49, 56, 217],
               [10, 227, 177],
               [152, 117, 25],
               [139, 76, 23],
               [53, 191, 10],
               [14, 244, 90],
               [247, 94, 189],
               [202, 160, 149],
               [24, 31, 150],
               [164, 236, 24],
               [47, 10, 204],
               [84, 187, 44],
               ]



    def __init__(self,split, **kwargs):
        super(CocoStuff, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',split=split,
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
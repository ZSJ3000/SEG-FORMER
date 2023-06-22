import os
import random
from tkinter import _flatten
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

random.seed(0)

json_path = r"E:\DESKTOP\ARVR\Week3_segformer\SegFormer-master\tools\data\COCO\annotations\instances_val2017.json"
img_path = r"E:\DESKTOP\ARVR\Week3_segformer\SegFormer-master\tools\data\COCO\val2017"

# random pallette
PALETTE = [[0,0,0],
         [167, 200, 7],
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
pallette = list(_flatten(PALETTE))

# load coco data
coco = COCO(annotation_file=json_path)

# get all image index info
ids = list(sorted(coco.imgs.keys()))
print("number of images: {}".format(len(ids)))

# get all coco class labels
coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])

n=0
m=0
# 遍历前三张图像
for img_id in ids[:]:
    # 获取对应图像id的所有annotations idx信息

    ann_ids = coco.getAnnIds(imgIds=img_id)
    # 根据annotations idx信息获取所有标注信息
    targets = coco.loadAnns(ann_ids)

    # get image file name
    path = coco.loadImgs(img_id)[0]['file_name']
    # read image
    img = Image.open(os.path.join(img_path, path)).convert('RGB')
    img_w, img_h = img.size

    masks = []
    cats = []
    for target in targets:
        cats.append(target["category_id"])  # get object class id
        polygons = target["segmentation"]   # get object polygons
        rles = coco_mask.frPyObjects(polygons, img_h, img_w)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = mask.any(axis=2)
        masks.append(mask)

    cats = np.array(cats, dtype=np.int32)

    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((0, img_h, img_w), dtype=np.uint8)

    # merge all instance masks into a single segmentation map
    # with its corresponding categories
    if (masks * cats[:, None, None]).size != 0:
        target = (masks * cats[:, None, None]).max(axis=0)
    else:
        # target = (masks * cats[:, None, None]).any(axis=0)
        os.remove(r"E:\DESKTOP\ARVR\Week3_segformer\SegFormer-master\tools\data\COCO\val2017\{}".format(path))
        m+=1
        continue


    # discard overlapping instances
    target[masks.sum(0) > 1] = 255
    target = Image.fromarray(target.astype(np.uint8))

    target.putpalette(pallette)
    target = target.convert('RGB')
    target.save(r"E:\DESKTOP\ARVR\Week3_segformer\SegFormer-master\tools\data\COCO\valnanno\{}".format(path))
    n+=1
    print("成功转换第{}/118287张图片".format(n))
print("有{}张图片没有标注".format(m))

class VOC:
    #图片分割使用，正常色盘
    PALETTE = [[0,0,0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                    [0, 64, 128]]

    CLASSES= ['background','aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

    # VIDEO的色盘，与正常色盘对称
    PALETTE1  = [[0,0,0],[0, 0, 128], [0, 128, 0], [0, 128, 128],
                     [128, 0, 0], [128, 0, 128], [128, 128, 0], [128, 128, 128],
                     [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192],
                     [128, 0, 64], [128, 0, 192], [128, 128, 64], [128, 128, 192],
                     [0, 64, 0], [0, 64, 128], [0, 192, 0], [0, 192, 128],
                     [128, 64, 0]]

    CLASSES1 = ['background','aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
    
    model="VOC_iter_10000.pth"
    config="segformer.b0.512x512.ade.160k.py"
    palette="voc"

class CITYSCAPES:
    # 图片分割使用，正常色盘
    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]
    # VIDEO的色盘，与正常色盘对称
    CLASSES1 = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE1 =[[128, 64, 128], [232, 35, 244], [70, 70, 70], [156, 102, 102],
              [153, 153, 190], [153, 153, 153], [30, 170, 250], [0, 220, 220],
              [35, 142, 107], [152, 251, 152], [180, 130, 70], [60, 20, 220],
              [0, 0, 255], [142, 0, 0], [70, 0, 0], [100, 60, 0],
              [100, 80, 0], [230, 0, 0], [32, 11, 119]]
    model = "CITY_iter_160K.pth"
    config = "segformer.b0.512x1024.city.160k.py"
    palette="cityscapes"




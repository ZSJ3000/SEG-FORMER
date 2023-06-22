from PIL import Image
import numpy as np
import os

bacepath = r"E:\DESKTOP\ARVR\Week3_segformer\SegFormer-master\tools\data\COCO\valnanno"  # 需要转化的文件夹路径，jpg和png都能一起批量转化（24转8）
savepath = r'E:\DESKTOP\ARVR\Week3_segformer\SegFormer-master\tools\data\COCO\valnanno8'  # 保存地路径
f_n = os.listdir(bacepath)
for n in f_n:
    imdir = bacepath + '\\' + n
    img = Image.open(imdir).convert('P')
    img.save(savepath + '\\' + n.split('.')[0] + '.png') # 转换后的进行保存

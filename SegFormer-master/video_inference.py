import os
from argparse import ArgumentParser
import mmcv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette
from tqdm import tqdm

import shutil
import warnings
warnings.filterwarnings("ignore") # 忽略警告


def show_result_pyplot(model, img, result, palette=None, fig_size=(15, 10), out_file=None):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, palette=palette, show=False, out_file=out_file)
    # plt.figure(figsize=fig_size)
    # plt.imshow(mmcv.bgr2rgb(img))
    # plt.show()
    return img

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)


def main():
    parser = ArgumentParser()
    #default=0表示摄像头
    parser.add_argument('--source', default=r"E:\DESKTOP\ARVR\Week3_segformer\SegFormer-master\TEST VIDEO\5.gif", help='Image source')
    parser.add_argument('--config', default=r"E:\DESKTOP\ARVR\Week3_segformer\SegFormer-master\local_configs\segformer\B0\segformer.b0.512x512.ade.160k.py",
                        help='Config file')
    parser.add_argument('--checkpoint', default=r"E:\DESKTOP\ARVR\Week3_segformer\SegFormer-master\tools\rs128\iter_10000.pth",
                        help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='voc',
        help='Color palette used for segmentation map')
    args = parser.parse_args()
    if not os.path.exists('results'):
        os.mkdir('results')
    save_path = 'results/'
    del_file(save_path)
    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    camera = cv2.VideoCapture(args.source)
    mean_time = 0
    cnt = 0
    while True:
        ret_val, img = camera.read()
        w, h, _ = img.shape
        img = cv2.resize(img, (1024, 1024))
        current_time = cv2.getTickCount()
        # test a single image
        result = inference_segmentor(model, img)
        # show the results
        out_file = save_path+'%s.jpg' % cnt
        result = show_result_pyplot(model, img, result, get_palette(args.palette))
        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05

        cv2.putText(result, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                    (400, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

        # 标签中每个RGB颜色的值
        VOC_COLORMAP = [[128, 0, 0], [0, 128, 0], [128, 128, 0],
                        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                        [0, 64, 128]]
        # 标签其标注的类别
        VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                       'diningtable', 'dog', 'horse', 'motorbike', 'person',
                       'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

        for i in range(len(VOC_CLASSES)):
            cv2.putText(result,VOC_CLASSES[i], (0, 20+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, VOC_COLORMAP[i], 2)


        result = cv2.resize(result, (h, w))
        cv2.imshow('img', result)
        cv2.imwrite(out_file, result)
        cnt += 1
        ch = cv2.waitKey(1)

        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break
    camera.release()
    # videowriter.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()




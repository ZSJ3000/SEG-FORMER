import tkinter as tk
from tkinter import filedialog
import random

import torch
from PIL import Image, ImageTk, ImageSequence
import time
import os
from argparse import ArgumentParser
import mmcv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import front
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette
from tqdm import tqdm
import shutil
import keyboard
from palette import *


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


def main(f_path,video_save_path,DATA):
    parser = ArgumentParser()
    #default=0表示摄像头
    parser.add_argument('--source', default=f_path, help='Image source')
    parser.add_argument('--config', default=r"E:\DESKTOP\ARVR\Week3_segformer\SegFormer-master\local_configs\segformer\B0\{}".format(DATA.config),
                        help='Config file')
    parser.add_argument('--checkpoint', default=r"E:\DESKTOP\ARVR\Week3_segformer\SegFormer-master\tools\rs128\{}".format(DATA.model),
                        help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default=DATA.palette,
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
        if f_path==0:
            if keyboard.is_pressed("Enter"):
                break
        if ret_val==False:
            camera.release()
            cv2.destroyAllWindows()
            break
        w, h, _ = img.shape
        img = cv2.resize(img, (640, 480))
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

        for i in range(len(DATA.CLASSES1)):
            cv2.putText(result,DATA.CLASSES1[i], (0, 20+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, DATA.PALETTE1[i], thickness=2)

        result = cv2.resize(result, (h, w))
        # cv2.imshow('img', result)
        cv2.imwrite(out_file, result)
        cnt += 1
        # ch = cv2.waitKey(1)

    print("video2jpg finished")
    img0 = cv2.imread('results/%d.jpg' % 1)
    fps = 30
    size = img0.shape
    size = list(size[0:2])
    size[0],size[1]=size[1],size[0]
    size=tuple(size)

    videowriter = cv2.VideoWriter(video_save_path,
                                  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, frameSize=size)
    num = len(os.listdir('results'))
    for i in tqdm(range(1, num)):
        img = cv2.imread('results/%d.jpg' % i)
        videowriter.write(img)
    videowriter.release()
    shutil.rmtree('results')
    os.mkdir('results')
    print("jpg2video finished")

    # videowriter.release()





def PIC_SEG(img_root,save_mask_root,DATA):

    def label2image(pred):
        # pred: [320,480]
        colormap = torch.tensor(DATA.PALETTE, device="cpu", dtype=int)

        return (colormap[pred, :]).data.cpu().numpy()

    config_file = r"local_configs\segformer\B0\{}".format(DATA.config)
    checkpoint_file = r"tools\rs128\{}".format(DATA.model)
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    img_root = img_root + "/"
    save_mask_root =save_mask_root +"/"
    img_names = os.listdir(img_root)
    for img_name in tqdm(img_names):
        # test a single image
        img = img_root + img_name
        result = inference_segmentor(model, img)[0]
        result = label2image(result)
        result = (np.uint8(result))
        for i in range(len(DATA.CLASSES)):
            cv2.putText(result, DATA.CLASSES[i], (0, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, DATA.PALETTE[i], 2)
        result = Image.fromarray(result)
        result.save(save_mask_root + img_name)
    print("执行完毕")



def B1():
    tk.messagebox.showinfo( "MY-SEG", "如有疑问请联系E-mail: 1509435185@qq.com")
def B2():

    A=tk.messagebox.askokcancel( "MY-SEG", "选择你想使用的视频")
    if A==True:
        root1 = tk.Tk()
        root1.title('Choose the video')
        # 选择视频文件
        root1.withdraw()
        f_path = filedialog.askopenfilename()
        B=tk.messagebox.askokcancel("MY-SEG", "选择视频保存位置")
        if (B==True) and (f_path!=None):
            video_save_path = filedialog.asksaveasfilename()
            C = tk.messagebox.askokcancel("MY-SEG", "是否要选择基于CITYSPACES的160K的训练模型")
            if C==True:
                DATA=CITYSCAPES
            else:
                DATA=VOC
            # 视频投入模型
            main(f_path, video_save_path, DATA)
    root1.destroy()


def B3():
    A=tk.messagebox.askokcancel( "MY-SEG", "请打开摄像头")
    if A==True:
        root2 = tk.Tk()
        root2.title('Select the location to save the video')
        root2.withdraw()
        f_path=0
        B=tk.messagebox.askokcancel( "MY-SEG", "选择视频保存位置")
        if B==True:
            video_save_path = filedialog.asksaveasfilename()
            tk.messagebox.showinfo("MY-SEG", "按下回车键停止")
            C = tk.messagebox.askokcancel("MY-SEG", "是否要选择基于CITYSPACES的160K的训练模型")
            if C == True:
                DATA = CITYSCAPES
            else:
                DATA = VOC
            # 视频投入模型
            main(f_path, video_save_path,DATA)
    root2.destroy()

def B4():
    A=tk.messagebox.askokcancel( "MY-SEG", "选择你想分割的图片所在的文件夹")
    if A == True:
        root3 = tk.Tk()
        root3.title('Choose the file')
         #选择视频文件
        root3.withdraw()
        img_root = filedialog.askdirectory()
        B = tk.messagebox.askokcancel("MY-SEG", "选择分割后的标签图片要保存的文件夹")
        if (B == True) and (img_root != None):
            save_mask_root = filedialog.askdirectory()
            C = tk.messagebox.askokcancel("MY-SEG", "是否要选择基于CITYSPACES的160K的训练模型")
            if C == True:
                DATA = CITYSCAPES
            else:
                DATA = VOC
            # 视频投入模型
            PIC_SEG(img_root, save_mask_root,DATA)
    root3.destroy()




def pick(event):
    while 1:
        im = Image.open(gif_address+gif_list[gif_num]+'.gif')
        # GIF图片流的迭代器
        iter = ImageSequence.Iterator(im)
        #frame就是gif的每一帧，转换一下格式就能显示了
        for frame in iter:
            #将frame放大至窗口大小
            frame = frame.resize((window_width,window_height),Image.ANTIALIAS)
            pic=ImageTk.PhotoImage(frame)
            canvas.create_image(window_width/2,window_height/2,image=pic)
            time.sleep(0.05)
            root.update_idletasks()  #刷新
            root.update()

def on_closing():
    if tk.messagebox.askokcancel("MY-SEG", "是否退出窗口？"):
        root.destroy()


#初始化窗体
window_width = 900
window_height = 600
root = tk.Tk()
root.geometry(str(window_width)+'x'+str(window_height))
root.title('MY-SEG')
root.configure(background='#DEEBF7')
root.resizable(False, False)
B1 = tk.Button(root, text ="关于MY-SEG", command = B1, height = 1, width = 20, bg = '#DEEBF7', relief='groove')
B2 = tk.Button(root, text ="选择视频", command = B2, height = 1, width = 8, bg = '#DEEBF7')
B3 = tk.Button(root, text ="调用摄像头", command = B3, height = 1, width = 8, bg = '#DEEBF7')
B4 = tk.Button(root, text ="图片标签分割", command = B4, height = 1, width = 12, bg = '#DEEBF7')

B1.pack()
B2.pack()
B2.place(x=0, y=0)
B3.pack()
B3.place(x=68, y=0)
B4.pack()
B4.place(x=136, y=0)


canvas = tk.Canvas(root,width=window_width, height=window_height,bg='#DEEBF7')
canvas.pack()
gif_list = ['IT男','爱心狗','大白','流泪狗','西瓜','幽灵','纸盒狗','音乐猫',"自行车"]
gif_num = random.randint(0,8)
gif_address ='E:/DESKTOP/ARVR/Week3_segformer/SegFormer-master/GIF/'
canvas.bind("<Enter>",pick)


#退出部分
root.protocol('WM_DELETE_WINDOW', on_closing)
root.mainloop()

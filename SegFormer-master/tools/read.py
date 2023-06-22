import os

import pickle



file = open(r"E:\DESKTOP\ARVR\Week3_segformer\SegFormer-master\tools\work_dirs\res.pkl","rb")

content = pickle.load(file)

print(content)

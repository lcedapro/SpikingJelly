import numpy as np
from PIL import Image
 
# 打开图片
img_PB_L0 = Image.open('PB_L0.png')
img_PB_L1 = Image.open('PB_L1.png')
img_PB_L2 = Image.open('PB_L2.png')
img_PB_L3 = Image.open('PB_L3.png')
img_PB_L4 = Image.open('PB_L4.png')
img_PB_L5 = Image.open('PB_L5.png')
img_SJ_L0 = Image.open('SJ_L0.png')
img_SJ_L1 = Image.open('SJ_L1.png')
img_SJ_L2 = Image.open('SJ_L2.png')
img_SJ_L3 = Image.open('SJ_L3.png')
img_SJ_L4 = Image.open('SJ_L4.png')
img_SJ_L5 = Image.open('SJ_L5.png')

# 转化为ndarray对象
arr_PB_L0 = np.array(img_PB_L0)
arr_PB_L1 = np.array(img_PB_L1)
arr_PB_L2 = np.array(img_PB_L2)
arr_PB_L3 = np.array(img_PB_L3)
arr_PB_L4 = np.array(img_PB_L4)
arr_PB_L5 = np.array(img_PB_L5)
arr_SJ_L0 = np.array(img_SJ_L0)
arr_SJ_L1 = np.array(img_SJ_L1)
arr_SJ_L2 = np.array(img_SJ_L2)
arr_SJ_L3 = np.array(img_SJ_L3)
arr_SJ_L4 = np.array(img_SJ_L4)
arr_SJ_L5 = np.array(img_SJ_L5)
 
arr_pb = np.concatenate((arr_PB_L0, arr_PB_L1, arr_PB_L2, arr_PB_L3, arr_PB_L4, arr_PB_L5), axis = 1) # 横向拼接
arr_sj = np.concatenate((arr_SJ_L0, arr_SJ_L1, arr_SJ_L2, arr_SJ_L3, arr_SJ_L4, arr_SJ_L5), axis = 1) # 横向拼接
arr = np.concatenate((arr_pb, arr_sj), axis = 0) # 纵向拼接

# 生成图片
img = Image.fromarray(arr)
 
# 保存图片
img.save('merge2.png')
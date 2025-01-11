# 由于原始数据过于稀疏，0占据了很大一部分，因此可以把大于等于1的像素值全部当作1来处理
# 这样，输入的数据就只占1个bit，可以大大减少数据量

# 补充以下程序，实现将root_dir下的所有.npz文件中的数据修改为0或1的数据，并保存到root_dir1目录下。root_dir1和root_dir的目录结构和文件名应保持不变，
# 例如源文件为'.\train\0\user01_fluorescent_0_154.npz'，则目标文件为'.\train_1bit\0\user01_fluorescent_0_154.npz'

# 修改数据的参考代码：
# arr_modified = np.where(arr >= 1, 1, 0)
# arr_modified = arr_modified.astype(np.uint8)

import numpy as np
import os

# 指定源目录和目标目录
root_dir = './train_original'
root_dir1 = './train'

# 指定裁剪的时间步长度
TARGET_T = 16

# 确保目标目录存在，如果不存在则创建
if not os.path.exists(root_dir1):
    os.makedirs(root_dir1)

# 遍历源目录下的所有文件和文件夹
for subdir, dirs, files in os.walk(root_dir):
    # 构建对应的目标目录路径
    subdir1 = subdir.replace(root_dir, root_dir1)
    if not os.path.exists(subdir1):
        os.makedirs(subdir1)
    
    # 遍历所有文件
    for file in files:
        # 检查文件是否为.npz文件
        if file.endswith('.npz'):
            # 构建完整的文件路径
            file_path = os.path.join(subdir, file)
            file_path1 = os.path.join(subdir1, file)
            
            # 加载.npz文件
            with np.load(file_path) as data:
                # 获取'frames'键对应的数组
                arr = data['frames']

                # 裁剪到 TARGET_T 帧，如果源数据长度小于 TARGET_T，则补齐
                source_t = arr.shape[0]
                if source_t >= TARGET_T:
                    # 裁剪到 TARGET_T 帧
                    arr_cropped = arr[:TARGET_T]
                else:
                    # 补齐到 TARGET_T 帧
                    padding = np.repeat(arr[-1:], TARGET_T - source_t, axis=0)  # 重复最后一帧
                    arr_cropped = np.concatenate([arr, padding], axis=0)  # 拼接

                # 修改数据，将大于等于1的像素值设为1，否则为0
                arr_modified = np.where(arr_cropped == 0, 0, 1)
                arr_modified = arr_modified.astype(np.int8)

                # 保存修改后的数据到新的.npz文件，保持键名为'frames'
                np.savez_compressed(file_path1, frames=arr_modified)

                print("frames modified: ", file_path1)

print("数据处理完成，已保存到", root_dir1)
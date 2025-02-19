'''
与CustomImageDataset的不同: 
CustomImageDataset保留全部事件，__getitem__返回(target_t, 2, H, W)
CustomImageDataset0只取正事件或负事件，__getitem__返回(target_t, 1, H, W)
'''

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import random

class CustomImageDataset0(Dataset):
    def __init__(self, root_dir, transform=None, target_t=8, expand_factor=1, random_en=True, num_crops_per_video=5):
        """
        Args:
            root_dir (string): Directory with all the images organized in subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_t: 目标时间帧数 (T=8)
            expand_factor: 帧扩展倍数 (e.g., 2表示扩展到16帧)
            random_en: 是否启用随机裁剪
            num_crops_per_video: 每个视频裁剪的样本数量
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)  # All subfolders are class labels
        self.image_paths = []
        self.target_t = target_t
        self.expand_factor = expand_factor
        self.random_en = random_en
        self.num_crops_per_video = num_crops_per_video  # 每个视频裁剪的样本数量
        
        # Collect all .npz file paths and their corresponding labels
        for class_label in self.classes:
            class_dir = os.path.join(root_dir, class_label)
            if not os.path.isdir(class_dir):
                continue
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.npz'):
                    self.image_paths.append((os.path.join(class_dir, file_name), int(class_label)))
    
    def __len__(self):
        # 数据集长度 = 视频数量 × 每个视频的裁剪样本数量
        return len(self.image_paths) * self.num_crops_per_video
    
    def __getitem__(self, idx):
        # 计算视频索引和裁剪索引
        video_idx = idx // self.num_crops_per_video
        crop_idx = idx % self.num_crops_per_video
        
        img_path, label = self.image_paths[video_idx]
        data = np.load(img_path)  # Load .npz file
        
        # Assuming .npz contains image data in a standard key like 'arr_0'
        img_data = data['frames']  # Adjust this if the key is different
        
        # If you need to apply any transformations to numpy arrays (e.g., normalization), do it here.
        if self.transform:
            img_data = self.transform(img_data)
        
        # Convert the numpy array to a tensor
        img_tensor = torch.tensor(img_data, dtype=torch.float32)

        # 确保 T 维度 >= target_t
        t, c, h, w = img_tensor.shape
        if t < self.target_t:
            raise ValueError(f"Time dimension {t} is smaller than target {self.target_t}.")

        # 在 T 轴上随机裁剪 target_t 帧
        start_idx = 0
        if self.random_en:
            start_idx = random.randint(0, t - self.target_t)
        img_tensor_cropped = img_tensor[start_idx:start_idx + self.target_t]  #裁剪后的 img shape: (4, C, H=64, W=64)

        # 扩展帧：重复原始帧扩展倍数
        expanded_img_tensor = img_tensor_cropped.repeat(self.expand_factor, 1, 1, 1)

        return expanded_img_tensor[:,0:1,:,:], label

if __name__ == '__main__':

    # 设置训练集和测试集的目录
    train_dir = './duration_1000/train'
    test_dir = './duration_1000/test'

    # 创建训练集和测试集的数据集实例
    train_dataset = CustomImageDataset(root_dir=train_dir, target_t=1, expand_factor=1, random_en=True, num_crops_per_video=5)
    test_dataset = CustomImageDataset(root_dir=test_dir, target_t=1, expand_factor=1, random_en=False, num_crops_per_video=5)

    # 创建训练集和测试集的DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    # 示例：打印前20个数据的shape和label
    count = 0
    for images, labels in train_loader:
        for i in range(images.size(0)):
            print(f"Training Image {count + i + 1}: Shape: {images[i].shape}, Label: {labels[i].item()}")
            if count + i + 1 >= 20:
                break
        count += images.size(0)
        if count >= 20:
            break

    count = 0
    for images, labels in test_loader:
        for i in range(images.size(0)):
            print(f"Testing Image {count + i + 1}: Shape: {images[i].shape}, Label: {labels[i].item()}")
            if count + i + 1 >= 20:
                break
        count += images.size(0)
        if count >= 20:
            break

# https://kimi.moonshot.cn/chat/cuq1c154335ccdjmc94g
# 帮我看看这个PYTORCH自定义数据集代码。我写的代码包含一个随机裁剪功能，用于将数据集中的图片流裁剪成指定的时间帧数。
# 但是，如果这样裁剪，那么数据集中有160个视频，裁剪出来的(target_t,C,H,W)数组也只有160个。这样会导致一个视频的大部分时间都没有覆盖到。
# 你能帮我修改成160个视频，裁剪出较多的(target_t,C,H,W)数组吗？

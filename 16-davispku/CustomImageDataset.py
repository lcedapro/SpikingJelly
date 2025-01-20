import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import random

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_t=8, expand_factor=1, random_en=True):
        """
        Args:
            root_dir (string): Directory with all the images organized in subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_t: 目标时间帧数 (T=4)
            expand_factor: 帧扩展倍数 (e.g., 2表示扩展到8帧)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)  # All subfolders are class labels
        self.image_paths = []
        self.target_t = target_t
        self.expand_factor = expand_factor
        self.random_en = random_en
        
        # Collect all .npz file paths and their corresponding labels
        for class_label in self.classes:
            class_dir = os.path.join(root_dir, class_label)
            if not os.path.isdir(class_dir):
                continue
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.npz'):
                    self.image_paths.append((os.path.join(class_dir, file_name), int(class_label)))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
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

        return expanded_img_tensor, label

if __name__ == '__main__':

    # 设置训练集和测试集的目录
    train_dir = './duration_1000/train'
    test_dir = './duration_1000/test'

    # 创建训练集和测试集的数据集实例
    train_dataset = CustomImageDataset(root_dir=train_dir, target_t=8, expand_factor=1, random_en=True)
    test_dataset = CustomImageDataset(root_dir=test_dir, target_t=8, expand_factor=1, random_en=False)

    # 创建训练集和测试集的DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(len(train_loader))
    print(len(train_loader))

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

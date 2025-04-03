import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from torchvision.transforms import functional as F

class CustomStaticDataset(Dataset):
    def __init__(self, root_dir, transform=None, expand_factor=4):
        """
        Args:
            root_dir (string): Directory with all the images organized in subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
            expand_factor: 帧扩展倍数 (e.g., 2表示扩展到16帧)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)  # All subfolders are class labels
        self.image_paths = []
        self.expand_factor = expand_factor
        
        # Collect all .npy file paths and their corresponding labels
        for class_label in self.classes:
            class_dir = os.path.join(root_dir, class_label)
            if not os.path.isdir(class_dir):
                continue
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.png'):
                    self.image_paths.append((os.path.join(class_dir, file_name), int(class_label)))
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        # img_data = np.load(img_path)  # Load .npy file
        # Load .png file and convert to numpy array
        img_data = np.array(Image.open(img_path).convert('L'))
        img_data = np.where(img_data != 0, 1, 0).astype(np.uint8).transpose(1,  0)
        img_data = np.expand_dims(img_data, axis=0)
        # print(img_data.shape)
        # If you need to apply any transformations to numpy arrays (e.g., normalization), do it here.
        if self.transform:
            img_data = self.transform(img_data)
        
        # Convert the numpy array to a tensor
        # img_tensor = torch.tensor(img_data, dtype=torch.float32)

        # 扩展帧：重复原始帧扩展倍数
        expanded_img_data = np.expand_dims(img_data, axis=0).repeat(self.expand_factor, axis=0)

        return torch.tensor(expanded_img_data, dtype=torch.float32), label

def test_transform(img_data : np.ndarray):
    # 在这里添加你的 transform 操作，例如归一化、裁剪等

    # 1. Rotation（旋转）: 随机旋转角度（-20到20度）
    angle = random.uniform(-10, 10)
    # angle = 0

    # 2. Translation（平移）: 10像素的随机水平（x）和垂直（y）平移
    tx = random.randint(-8, 8)
    ty = random.randint(-8, 8)
    # tx = 0
    # ty = 0
    translate = [tx, ty]

    # 3. Scaling（缩放）: 随机缩放因子（0.8到1.2）
    scale = random.uniform(0.9, 1.2)
    # scale = 1

    # 4. Shear（错切）: 20度的随机水平（x）和垂直（y）错切
    sx = random.uniform(-2, 2)
    sy = random.uniform(-2, 2)
    # sx = 0
    # sy = 0
    shear = [sx, sy]

    img_data_tensor = torch.tensor(img_data, dtype=torch.float32)

    # Apply the transformations
    img_data_tensor = F.affine(img_data_tensor, angle, translate, scale, shear, interpolation=F.InterpolationMode.NEAREST, fill=[0])

    return img_data_tensor.numpy()

if __name__ == '__main__':

    # 设置训练集和测试集的目录
    train_dir = './temporary_datasets/duration_2000_0306'
    test_dir = './temporary_datasets/duration_2000_0306'

    # 创建训练集和测试集的数据集实例
    train_dataset = CustomStaticDataset(root_dir=train_dir, expand_factor=4)
    test_dataset = CustomStaticDataset(root_dir=test_dir, expand_factor=4)

    # 创建训练集和测试集的DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    # # 打印一个批次的数据
    # for data, labels in train_loader:
    #     print("Data shape:", data.shape)  # 打印数据形状
    #     print("Labels:", labels)         # 打印标签
    #     break
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

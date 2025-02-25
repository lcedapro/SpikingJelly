import numpy as np

def numpy_maxpool2d(X, pool_size, stride):
    n, m = X.shape
    out_n = (n - pool_size) // stride + 1
    out_m = (m - pool_size) // stride + 1
    out = np.zeros((out_n, out_m))

    for i in range(out_n):
        for j in range(out_m):
            out[i][j] = np.max(X[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size])
    return out

# Compare numpy_maxpool2d with torch.nn.functional.max_pool2d
if __name__ == "__main__":
            # 在输入图像中截取对应池化窗口的区域
            # 使用切片操作获取池化窗口内的最大值
    image = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    # 返回池化后的输出特征图
    # numpy_maxpool2d
    pooling_size = 2
    stride = 2
    print(numpy_maxpool2d(image, pooling_size, stride))
    # torch.nn.functional.max_pool2d
    import torch
    image = torch.tensor(image)
    print(torch.nn.functional.max_pool2d(image.unsqueeze(0).unsqueeze(0), pooling_size, stride))
    


    
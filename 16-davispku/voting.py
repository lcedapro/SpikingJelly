# 用于PAIBox解码的VotingLayer

# 写一个python脚本，将输入的numpy数组的元素按照每隔固定的列数的形式相加。输出数组的第i列第j行元素是输入数组的第i列第j*2行和第i列第j*2+1行元素之和
# 示例输入：
# [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
# 示例输出：
# [[3,7],[11,15],[19,23],[27,31]]


import numpy as np

def voting(arr, step):
    if len(arr.shape) == 1:
        cols = arr.shape
        output = np.zeros((cols // step ))

        for i in range(cols // step):
            for j in range(step):
                output[i] += arr[i * step + j]

    elif len(arr.shape) == 2:
        rows, cols = arr.shape
        output = np.zeros((rows, cols // step ))

        for i in range(rows):
            for j in range(cols // step):
                for k in range(step):
                    output[i, j] += arr[i, j * step + k]
    
    elif len(arr.shape) == 3:
        depth, rows, cols = arr.shape
        output = np.zeros((depth, rows, cols // step))

        for i in range(depth):
            for j in range(rows):
                for k in range(cols // step):
                    for l in range(step):
                        output[i, j, k] += arr[i, j, k * step + l]
    else:
        raise ValueError("Input array must be 1D, 2D, or 3D")
    
    # output datatype = input datatype
    output = output.astype(arr.dtype)
    return output

if __name__ == '__main__':
    input_arr = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    output_arr = voting(input_arr, 2)
    print(output_arr)

    # 3D
    input_arr = np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],[[17,18,19,20],[21,22,23,24],[25,26,27,28],[29,30,31,32]]])
    output_arr = voting(input_arr, 2)
    print(output_arr)

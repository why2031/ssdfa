import numpy as np

# 加载 .npy 文件，allow_pickle=True 适用于存储了非纯数组数据（例如字典）的情况
data = np.load('mnist_conv.npy', allow_pickle=True)

# 如果数据是字典，可以查看字典的键值：
if isinstance(data, dict):
    for key, value in data.items():
        print(f"{key}: {value}")
else:
    print(data)

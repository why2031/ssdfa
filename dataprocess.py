import torch
import torch.utils.data as Data
import random
import numpy as np
import h5py

'''
Split Data:
    Interictal_Train: Interictal_Validation = 8: 2
    Preictal_Train: Preictal_Validation = 8: 2
'''
def split_dataset(dataset, rate):
        # e.g. dimension: X * 23 * 5120
        # shape should be X
    # rate is a float number: rate = train_data / whole_data
    # split number should be in int type
    train_split_inter = int(dataset['interictal'].shape[0] * rate)
    train_split_pre = int(dataset['preictal'].shape[0] * rate)

    # create an array in range of X
    random_idx_inter = np.arange(dataset['interictal'].shape[0])
    random_idx_pre = np.arange(dataset['preictal'].shape[0])

    # set random seed
    # consistency 一致性
    random.seed(508)
    random.shuffle(random_idx_inter)
    random.shuffle(random_idx_pre)

    # split
    # train_part_inter取的是random_idx_inter中前train_split_inter（不包含）个元素
    train_part_inter = random_idx_inter[:train_split_inter]
    train_part_pre = random_idx_pre[:train_split_pre]
    val_part_inter = random_idx_inter[train_split_inter:]
    val_part_pre = random_idx_pre[train_split_pre:]

    return train_part_inter, train_part_pre, val_part_inter, val_part_pre

'''
Training Set Generator
'''
# still needs to be revised?
def train_generator(dataset, batch_size, pre_part, inter_part, rate):

    # initialize an index from 0 to 5120 with step of rate
    resample_idx = np.arange(0, 5120, rate)

        # e.g. X * 23 * 5120
        # dataset[0].shape = [23, 5120]
        # dataset[0].shape[0] = [23], dataset[0].shape[1] = [5120]
    data_shape = dataset['interictal'][0].shape

        # e.g. dimension should be [batch_size, 23, int(5120 / rate)]
        # 5120 / rate = len(resample_idx)
    X = np.zeros((batch_size, data_shape[0], int(data_shape[1] / rate)))
    y = np.zeros((batch_size, ))

    # inter_samples: pre_samples = half_batch_size: half_batch_size = 1: 1
    half_batch = int(batch_size / 2)
    inter_samples = random.sample(range(len(inter_part)), half_batch)
    pre_samples = random.sample(range(len(pre_part)), half_batch)

    # The following code lines have the same functionality to get the index
    # inter_idxs = list(inter_part[inter_samples])
    # pre_idxs = list(pre_part[pre_samples])
    pre_idxs = list(pre_samples)
    inter_idxs = list(inter_samples)

    # sort the index
    inter_idxs.sort()
    pre_idxs.sort()

        # e.g. dataset = X * 23 * 5120
        # X1 = len(inter_idxs) * 23 * 5120
    X1 = inter_part[inter_idxs, :, :]
    X[:half_batch] = X1[:, :, resample_idx]
    X2 = pre_part[pre_idxs, :, :]
    X[half_batch:] = X2[:, :, resample_idx]

    # indices that >= half_batch are from preictal(1)
    y[half_batch:] = 1

    # X added another dimension
    X = X.reshape(X.shape + (1, ))
    # y = to_categorical(y) # to one-hot vector?

    return X, y

'''
Validation Set Generator:
    Similar To Training Set Generator
    No Sampling!
'''
def val_generator(dataset, inter_part, pre_part, rate):
    resample_idx = np.arange(0, 5120, rate)

    inter_idxs = list(inter_part)
    pre_idxs = list(pre_part)

    inter_idxs.sort()
    pre_idxs.sort()

    X1 = dataset['interictal'][inter_idxs, :, :]
    X2 = dataset['preictal'][pre_idxs, :, :]

    X1 = X1[:, :, resample_idx]
    X2 = X2[:, :, resample_idx]

    X = np.concatenate((X1, X2), axis=0)
    X = X.reshape(X.shape + (1, ))

    y1 = np.zeros((len(inter_idxs), ))
    y2 = np.ones((len(pre_idxs), ))

    y = np.concatenate((y1, y2), axis=0)
    # y = to_categorical(y)

    return X, y

'''
Overlap
'''
def overlap_data(data, overlap_rate, divide):
    # re-construct
        # e.g. data = N * 23 * 5120
        # new = (N*divide) * 23 * (5120/divide)
    new = np.zeros((len(data) * divide, 23, int(5120 / divide)))

    # len(new) = len(data) * divide
        # e.g. range(N*divide)
    for i in range(len(new)):
        new[i] = data[int(i / divide), :, int(5120 / divide) * (i % divide) : int(5120 / divide) * (i % divide + 1)]

    # round means that 0.5 becomes 1 and 0.49 becomes 0
    slide_num = round((1 - overlap_rate) * divide)

    new_num = int((len(new) - divide) / slide_num + 1)

    new_sample = np.zeros((new_num, 23, 5120))

    for i in range(new_num):
        tmp = np.zeros((23, 1))
        for j in range(divide):
            tmp = np.concatenate((tmp, new[i * slide_num + j]), axis=1)
        tmp_sample = np.delete(tmp, 0, axis=1)
        new_sample[i] = tmp_sample

    return new_sample

'''
Process
'''
def process_eeg(sessionIDs):
    # create an array to iterate
    # sessionIDs = ['Pretrain09_s1_06']

    saved_paths = {}

    for sessionID in sessionIDs:
        metrics = np.zeros((1, 3))  # no use

        data_path = 'E:/BCI/Data/' + sessionID + '.h5'
        print(f'data_path: {data_path}')

        dataset = h5py.File(data_path, 'r')

        training_rate = 0.8  # add a configuration to store these info?

        train_part_inter, train_part_pre, val_part_inter, val_part_pre = split_dataset(dataset, training_rate)

        val_inter_num = len(val_part_inter)  # never used
        val_pre_num = len(val_part_pre)  # never used

        class_weights = {0: 1., 1: 1.}  # balance?

        resample_rate = 1

        print('********************Preictal********************')
        print(dataset['preictal'].shape[0])
        val_pre_idxs = list(val_part_pre)  # preictal验证部分的索引
        val_pre_idxs = sorted(val_pre_idxs)
        train_samples_pre = np.zeros((1, 23, 5120))  # the below 'delete' operation deletes here

        # 这里从validation集中取间隔的索引再次构成训练样本！data augmentation
        for i in range(len(val_pre_idxs)):
            print(f'Iteration {i + 1} : {len(val_pre_idxs)}')
            data_idxs = []  # new array

            if i < len(val_pre_idxs) - 1:
                data_interval = val_pre_idxs[i + 1] - val_pre_idxs[i]  # calculate the interval in idxs
                print(f'Interval: {data_interval}')  # My addition
                if data_interval > 1:
                    for j in range(data_interval):
                        if j > 0:
                            data_idxs.append(val_pre_idxs[i] + j)

                    train_samples_beforeol = dataset['preictal'][data_idxs, :, :]  # 'beforeol' means before overlap?
                    # 并不是对所有的部分都进行overlap，而是对连续的数据进行overlap，因此要计算interval！！！
                    train_samples_pre = np.concatenate(
                        (train_samples_pre, overlap_data(train_samples_beforeol, overlap_rate=0.95, divide=20)), axis=0)
                    print(f'Current Shape: {train_samples_pre.shape}')  # why this is in the iteration？
                print()

        train_samples_pre = np.delete(train_samples_pre, 0, axis=0)  # see above explanation
        print(f'Final Shape: {train_samples_pre.shape}')
        print('********************Preictal********************')
        print()

        print('********************Inter********************')
        print(dataset['interictal'].shape[0])
        val_inter_idxs = list(val_part_inter)  # interictal验证部分的索引
        val_inter_idxs = sorted(val_inter_idxs)
        train_samples_inter = np.zeros((1, 23, 5120))  # the below 'delete' operation deletes here

        # 这里从validation集中取间隔的索引再次构成训练样本！data augmentation
        for i in range(len(val_inter_idxs)):
            print(f'Iteration {i + 1} : {len(val_inter_idxs)}')
            data_idxs = []  # new array

            if i < len(val_inter_idxs) - 1:
                data_interval = val_inter_idxs[i + 1] - val_inter_idxs[i]  # calculate the interval in idxs
                print(f'Interval: {data_interval}')  # My addition
                if data_interval > 1:
                    for j in range(data_interval):
                        if j > 0:
                            data_idxs.append(val_inter_idxs[i] + j)

                    train_samples_beforeol_inter = dataset['interictal'][data_idxs, :, :]  # 'beforeol' means before overlap?
                    # 并不是对所有的部分都进行overlap，而是对连续的数据进行overlap，因此要计算interval！！！
                    train_samples_inter = np.concatenate(
                        (train_samples_inter, overlap_data(train_samples_beforeol_inter, overlap_rate=0.9, divide=20)),
                        axis=0)
                    print(f'Current Shape: {train_samples_inter.shape}')
                print()

        train_samples_inter = np.delete(train_samples_inter, 0, axis=0)  # see above explanation
        print(f'Final Shape: {train_samples_inter.shape}')
        print('********************Inter********************')

        max_batch_size = min(train_samples_inter.shape[0], train_samples_pre.shape[0])
        temp_batch_size = 1
        while temp_batch_size * 2 <= max_batch_size:
            temp_batch_size *= 2

        print(f'Max Batch Size: {temp_batch_size}')

        # 生成训练集
        # 由于经过overlap后preictal部分size以倍数形式增大，因而half_batchsize接近于interictal的数据点
        X_train, y_train = train_generator(dataset, temp_batch_size, train_samples_pre, train_samples_inter, resample_rate)

        # 生成验证集（为统一，这里全部改为val）
        X_val, y_val = val_generator(dataset, val_part_inter, val_part_pre, resample_rate)

        # calculate Mean and Std used for data normalization
        data_mean = np.mean(np.concatenate((X_train, X_val), axis=0), axis=(0, 1, 2))
        data_std = np.std(np.concatenate((X_train, X_val), axis=0), axis=(0, 1, 2))

        X_train = torch.from_numpy((X_train - data_mean) / data_std)

        y_train = torch.from_numpy(y_train)
        y_train = y_train.long()

        X_val = torch.from_numpy((X_val - data_mean) / data_std)

        y_val = torch.from_numpy(y_val)
        y_val = y_val.long()

        print(f'X_train shape: {X_train.shape}')
        print(f'y_train shape: {y_train.shape}')
        print(f'X_val shape: {X_val.shape}')
        print(f'y_val shape: {y_val.shape}')
        print()

        train_dataset = Data.TensorDataset(X_train, y_train)
        val_dataset = Data.TensorDataset(X_val, y_val)

        train_dataset_path = f'E:/BCI/VPT_DFA/src/processed2/{sessionID}_train.pt'
        val_dataset_path = f'E:/BCI/VPT_DFA/src/processed2/{sessionID}_val.pt'

        torch.save(train_dataset, train_dataset_path)
        torch.save(val_dataset, val_dataset_path)

        saved_paths[sessionID] = {
            'train_dataset': train_dataset_path,
            'val_dataset': val_dataset_path
        }

    return saved_paths

sessionIDs = ['01_s1', '01_s2',
              '04_s1', '04_s2',
              '05_s1', '05_s2',
              '07_s1', '07_s2',
              '09_s1', '09_s2',
              '22_s1', '22_s2']

# process_eeg(sessionIDs)
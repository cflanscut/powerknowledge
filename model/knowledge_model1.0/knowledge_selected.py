import sys
import numpy as np
import itertools
import csv
sys.path.append('data/')


def index_read_data(index, data):
    select_data = np.zeros(len(index))
    i = 0
    for id in index:
        select_data[i] = data[id]
        i += 1
    return select_data


x = [
    [0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 0, 1, 1, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
]
x = np.array(x)
for i in range(x.shape[0]):  # i为第几个样本
    record = {}
    x_nonzero = np.nonzero(x[i, :])
    x_nonzero = x_nonzero[0].tolist()
    feature_len = len(x_nonzero)
    for k in range(1, feature_len + 1):  # k为选几个特征
        for c in itertools.combinations(x_nonzero, k):  # c为选的特征的index
            record[c] = []
            i_data = index_read_data(c, x[i, :])
            for j in range(x.shape[0]):
                j_data = index_read_data(c, x[j, :])
                if (i_data == j_data).all():  # 如果j样本读取出来的数据和i相同，那么就记录
                    record[c].append(j)
    # record = sorted(record.items(), key=lambda x: (len(x[1]), len(x[0])))  # 按照同类最少原则
    with open('test.csv', 'w', newline='') as op:
        writer = csv.writer(op)
        writer.writerows(record.items())

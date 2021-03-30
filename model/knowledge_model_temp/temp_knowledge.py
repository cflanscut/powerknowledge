import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import pandas as pd
import json
sys.path.append('data/')
from read_PLAID_data import read_source_data, read_index


def vibration_attenuation(data):
    # to do
    # 通过拟合振荡衰减函数，计算损失来判断
    flag = 0
    return flag


def attenuation(data):
    # to do
    # 通过拟合衰减函数，计算损失来判断
    flag = 0
    return flag


def cos_attenuation(data):
    # to do
    # 暂态后跟稳定有个差
    flag = 0
    return flag


def times_detector(data, f, fs):
    Ns = fs / f
    k = 1
    for i in range(1, 7):
        N = int(Ns / i)
        data1 = np.array(data[0:N])
        data2 = np.array(data[N + 1:2 * N])
        pp_count = abs(np.argmax(data1) - np.argmax(data2))
        if abs(pp_count) < Ns / 25:
            k = i
    return k


def MSE(y, y_pred):
    y_diff = y - y_pred
    y_diff_square = np.power(y_diff, 2)
    return np.sum(y_diff_square) / len(y_diff)


def attenuation_f(t, a, b, c, k, fi, f):
    return a * np.sin(k * 2 * np.pi * f * t + fi) * np.exp(-c * t) + b


def fit_attenuation(y):
    lenth = len(y)
    t = np.linspace(0, (lenth - 1) / 30000, lenth)
    f = 60
    k = times_detector(y, f, 30000)
    fi = 0
    a = 1
    b = 0
    c = 100
    iterations = 1000
    updata_rate = 1
    a_adagrad = 0
    b_adagrad = 0
    c_adagrad = 0
    k_adagrad = 0
    fi_adagrad = 0
    eps = 0.000000001

    for i in range(iterations):
        sin_r = np.sin(k * 2 * np.pi * f * t + fi) * np.exp(-c * t)
        cos_r = np.cos(k * 2 * np.pi * f * t + fi) * np.exp(-c * t)
        y_pred = a * sin_r + b
        y_diff = y - y_pred
        a_diff = 2 * np.sum(y_diff * -sin_r) / lenth
        b_diff = 2 * np.sum(y_diff * -1) / lenth
        c_diff = 2 * np.sum(y_diff * a * sin_r * t) / lenth
        k_diff = 2 * np.sum(y_diff * -a * cos_r * 2 * np.pi * f * t) / lenth
        fi_diff = 2 * np.sum(y_diff * -a * cos_r) / lenth

        a_adagrad += a_diff**2
        b_adagrad += b_diff**2
        c_adagrad += c_diff**2
        k_adagrad += k_diff**2
        fi_adagrad += fi_diff**2

        a = a - 2 * updata_rate * a_diff / (np.sqrt(a_adagrad + eps))
        b = b - updata_rate * b_diff / (np.sqrt(b_adagrad + eps))
        c = c - 10 * updata_rate * c_diff / (np.sqrt(c_adagrad + eps))
        k = k - updata_rate * k_diff / (np.sqrt(k_adagrad + eps))
        fi = fi - 5 * updata_rate * fi_diff / (np.sqrt(fi_adagrad + eps))

        if a < 0:
            a = -a
            fi += np.pi

        if k < 0:
            k = 0

        if k > 11:
            k = 11

        if fi > 2 * np.pi:
            fi -= 2 * np.pi
        if fi < 0:
            fi += 2 * np.pi

        y_pred = attenuation_f(t, a, b, c, k, fi, 60)
        mse = MSE(y, y_pred) / abs(a)
        if mse < 1e-5:
            break
    return a, b, c, fi, k, mse


def fit_attenuation2(y):
    lenth = len(y)
    t = np.linspace(0, (lenth - 1) / 30000, lenth)
    b = 1
    c = 1
    b_adagrad = 0
    c_adagrad = 0
    eps = 0.00000001
    learning_rate = 1
    iterations = 1000

    for i in range(iterations):
        y_diff = y - b * np.exp(-c * t)
        b_diff = 2 * np.sum(y_diff * -np.exp(-c * t)) / lenth
        c_diff = 2 * np.sum(y_diff * b * np.exp(-c * t) * t) / length

        b_adagrad += b_diff**2
        c_adagrad += c_diff**2

        b = b - learning_rate * b_diff / np.sqrt(b_adagrad + eps)
        c = c - 10 * learning_rate * c_diff / np.sqrt(c_adagrad + eps)

        mse = MSE(y, b * np.exp(-c * t))
        if abs(mse / b) < 0.01:
            break
    return b, c


def fit_cosin(y):
    lenth = len(y)
    t = np.linspace(0, (lenth - 1) / 30000, lenth)
    f = 60
    a = 1
    k = 1
    fi = 0
    a_adagrad = 0
    k_adagrad = 0
    fi_adagrad = 0
    eps = 0.00000001
    learning_rate = 0.1
    iterations = 1000

    for i in range(iterations):
        sin_r = np.sin(k * 2 * np.pi * f * t + fi)
        cos_r = np.cos(k * 2 * np.pi * f * t + fi)

        y_diff = y - a * sin_r
        a_diff = 2 * np.sum(y_diff * -sin_r) / lenth
        k_diff = 2 * np.sum(y_diff * -a * cos_r * 2 * np.pi * f * t) / lenth
        fi_diff = 2 * np.sum(y_diff * -a * cos_r) / lenth

        a_adagrad += a_diff**2
        k_adagrad += k_diff**2
        fi_adagrad += fi_diff**2

        a = a - 10 * learning_rate * a_diff / np.sqrt(a_adagrad + eps)
        k = k - learning_rate * k_diff / np.sqrt(k_adagrad + eps)
        fi = fi - learning_rate * fi_diff / np.sqrt(fi_adagrad + eps)

        mse = MSE(y, a * sin_r)
        if abs(mse / a) < 0.001:
            break
    return a, k, fi


def fit_cosin_puls_attenuation(y):
    lenth = len(y)
    t = np.linspace(0, (lenth - 1) / 30000, lenth)
    f = 60

    b, c = fit_attenuation2(y)
    a, k, fi = fit_cosin(y - b2 * np.exp(-c2 * t))

    a_adagrad = 0
    b_adagrad = 0
    c_adagrad = 0
    k_adagrad = 0
    fi_adagrad = 0
    eps = 0.00000001
    learning_rate = 0.01
    iterations = 1000

    for i in range(iterations):
        sin_r = np.sin(k * 2 * np.pi * f * t + fi)
        cos_r = np.cos(k * 2 * np.pi * f * t + fi)

        y_diff = y - (a * sin_r + b * np.exp(-c * t))
        a_diff = 2 * np.sum(y_diff * -sin_r) / lenth
        b_diff = 2 * np.sum(y_diff * -np.exp(-c * t)) / lenth
        c_diff = 2 * np.sum(y_diff * b * np.exp(-c * t) * t) / length
        k_diff = 2 * np.sum(y_diff * -a * cos_r * 2 * np.pi * f * t) / lenth
        fi_diff = 2 * np.sum(y_diff * -a * cos_r) / lenth

        a_adagrad += a_diff**2
        b_adagrad += b_diff**2
        c_adagrad += c_diff**2
        k_adagrad += k_diff**2
        fi_adagrad += fi_diff**2

        a = a - learning_rate * a_diff / np.sqrt(a_adagrad + eps)
        b = b - learning_rate * b_diff / np.sqrt(b_adagrad + eps)
        c = c - learning_rate * c_diff / np.sqrt(c_adagrad + eps)
        k = k - learning_rate * k_diff / np.sqrt(k_adagrad + eps)
        fi = fi - learning_rate * fi_diff / np.sqrt(fi_adagrad + eps)

        mse = MSE(y, a * sin_r + b * np.exp(-c * t)) / np.sqrt(a**2 + b**2)
        if mse < 0.01:
            break
    return a, b, c, fi, k, mse


with open(
        '/home/chaofan/powerknowledge/data/source/metadata_submetered2.1.json',
        'r',
        encoding='utf8') as load_meta:
    meta = json.load(load_meta)

total_start_time = time.time()
source_dir = 'data/source/submetered_new_pured/source/'
file_list = os.listdir(source_dir)
length = 1500
t = np.linspace(0, (length - 1) / 30000, length)
outputs = []
type_index = read_index('type')
type_selected = ['Air Conditioner']
type_skiped = ['Compact Fluorescent Lamp', 'Fridge']

for key in type_index.keys():
    print(key)
    csv_dir = type_index[key]
    csv_dir = list(map(str, csv_dir))
    # if key not in type_selected:
    #     continue
    # if key in type_skiped:
    #     continue
    for i, file in enumerate(csv_dir):
        start_time = time.time()
        file += '.csv'
        if file not in file_list:
            continue
        file_dir = source_dir + file
        Switch_V, Switch_I = read_source_data(file_dir,
                                              offset=0,
                                              length=length)
        Stable_V, Stable_I = read_source_data(file_dir,
                                              offset=length,
                                              length=length)
        I_diff = np.array(Switch_I) - np.array(Stable_I)
        a, b, c, fi, k, mse = fit_attenuation(I_diff)
        b2, c2 = fit_attenuation2(I_diff)
        a1, b1, c1, fi1, k1, mse1 = fit_cosin_puls_attenuation(I_diff)

        plt.figure(figsize=(8, 6))
        ax1 = plt.subplot(511)
        plt.plot(I_diff)
        ax2 = plt.subplot(512)
        plt.plot(attenuation_f(t, a, b, c, k, fi, 60))
        ax3 = plt.subplot(513)
        plt.plot(b2 * np.exp(-c2 * t))
        ax4 = plt.subplot(514)
        plt.plot(I_diff - b2 * np.exp(-c2 * t))
        ax5 = plt.subplot(515)
        plt.plot(a1 * np.sin(k1 * 2 * np.pi * 60 * t + fi1) +
                 b1 * np.exp(-c1 * t))
        # print('attenuation:%.5f,attenuation1:%.5f' % (mse, mse1))
        # plt.show()

        plt.suptitle(file + '(' + meta[file[0:-4]]['appliance']['type'] + '-' +
                     meta[file[0:-4]]['appliance']['status'] + ')')
        plt.savefig('model/knowledge_model_temp/diff_jpg/' + file[0:-4] +
                    '.jpg',
                    dpi=600)
        plt.close()
        a = round(a, 5)
        b = round(b, 5)
        c = round(c, 5)
        k = round(k, 1)
        fi = round(fi, 5)
        mse = round(mse, 5)
        a1 = round(a1, 5)
        b1 = round(b1, 5)
        c1 = round(c1, 5)
        k1 = round(k1, 1)
        fi1 = round(fi1, 5)
        mse1 = round(mse1, 5)
        op = {}
        op['file'] = file
        op['a'] = a
        op['b'] = b
        op['c'] = c
        op['k'] = k
        op['fi'] = fi
        op['mse'] = mse
        op['a1'] = a1
        op['b1'] = b1
        op['c1'] = c1
        op['k1'] = k1
        op['fi1'] = fi1
        op['mse1'] = mse1
        label = meta[file[0:-4]]['appliance']['type']
        op['type'] = label
        outputs.append(op)
        print('[%03d/%03d]:%.3fs' %
              (i, len(csv_dir), time.time() - start_time))

pd.DataFrame(outputs).to_csv('model/knowledge_model_temp/fit_result.csv')
print('finished! total cost:%.3fs' % (time.time() - total_start_time))

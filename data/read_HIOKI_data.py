import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def read_source_data(file_dir, length='default', offset=0):
    soucre_data = pd.read_csv(file_dir,
                              names=["U", "I"],
                              skiprows=9,
                              usecols=[1, 2])
    if length == 'default':
        length = soucre_data.shape[0] - offset - 1
    return soucre_data['U'][offset:offset + length -
                            1], soucre_data['I'][offset:offset + length - 1]


# test
# Temp_U, Temp_I = read_source_data(
#     'data/source/HIOKI/source/DDCsx_20201126_170239.csv', length=800)
# print()


def find_temp_start(guide_feature, threshold):
    dir = '/home/chaofan/powerknowledge/data/source/HIOKI/process/'
    start_record = {}
    csv_list = os.listdir(dir)
    for i, file in enumerate(csv_list):
        if '.csv' not in file:
            continue
        value0 = 0
        data = pd.read_csv(os.path.join(dir, file))
        data = data[guide_feature]
        for j, value in enumerate(data):
            if abs(value - value0) >= threshold:
                start_record[file] = j
                break
            else:
                value0 = value
    return start_record


# test
# t = find_temp_start('P', 5)
# print()


def find_temp_start_pricisely(guide_feature, threshold):
    source_dir = '/home/chaofan/powerknowledge/data/source/HIOKI/source/'
    start_index = find_temp_start(guide_feature, threshold)
    start_record = {}
    for file, start_period in start_index.items():
        if start_period > 0:
            start_period -= 1
            file_dir = source_dir + file
            Voltage, Current = read_source_data(file_dir,
                                                offset=start_period * 400,
                                                length=1000)
            start_i = 0
            value0 = 0
            consecutive_count = 0
            sign = 0
            for i, value in enumerate(Current):
                diff = value - value0
                if np.sign(diff) == sign and sign != 0:
                    if consecutive_count == 0:
                        start_i = start_period * 400 + i
                    consecutive_count += 1
                else:
                    consecutive_count = 0
                if consecutive_count > 5:
                    if start_i > 100:
                        start_i -= 100
                    start_record[file] = start_i
                    break
                value0 = value
                sign = np.sign(diff)
    return start_record


# test
# source_dir = 'data/source/HIOKI/source/'
# new_dir = 'data/source/HIOKI/source_pured/'
# start_index = find_temp_start_pricisely('P', 5)
# count = 0

# for file, start_row in start_index.items():
#     count += 1
#     df = pd.read_csv(source_dir + file, skiprows=start_row)
#     df.to_csv(new_dir + file, index=False)
#     print('dealing...:%03d/%03d' % (count, len(start_index)))


def bar(pos):
    pos = int(pos)
    bar.ax.clear()
    if pos + bar.N > len(bar.x):
        n = len(bar.x) - pos
    else:
        n = bar.N
    X = bar.x[pos:pos + n]
    Y = bar.y[pos:pos + n]
    bar.ax.plot(X, Y)  # 相当于每次触发重新画

    bar.ax.xaxis.set_ticks([])
    # bar.ax.yaxis.set_ticks([])


def slider_plot(data):
    fig, bar.ax = plt.subplots(figsize=(10, 6))
    bar.x = np.linspace(0, len(data) - 1, len(data))
    bar.y = np.array(data)
    bar.N = 3000
    barpos = plt.axes([0.18, 0.05, 0.55, 0.03],
                      facecolor="skyblue")  # [左下角在画板的位置（横坐标纵坐标），图大小（长宽）]
    slider = Slider(barpos, 'Barpos', 0, len(bar.x) - bar.N,
                    valinit=0)  # barpos放置滑条容器，用axes实例；0最小值；后面接着最大值；valinit初始值
    slider.on_changed(
        bar)  # on_changed是滑动条方法，用于绑定滑动条值改变的事件，就是滑动条改变了就调用这个函数，会自动传滑动条当前值进去
    bar(0)
    plt.show()
    plt.close()


# Temp_U, Temp_I = read_source_data(
#     'data/source/HIOKI/source_pured/DDClyk_20210402_111655.csv', length=4000)
# slider_plot(Temp_I)


def find_stable_start(guide_feature,
                      threshold=5,
                      consecutive_count=3,
                      stable_rate=0.05):
    dir = '/home/chaofan/powerknowledge/data/source/HIOKI/process/'
    start_record = {}
    csv_list = os.listdir(dir)
    for i, file in enumerate(csv_list):
        if '.csv' not in file:
            continue
        data = pd.read_csv(os.path.join(dir, file))
        data = data[guide_feature]
        value0 = 0
        count = 0
        start_record[file] = []
        threshold = 5
        for j, value in enumerate(data):
            if abs(value / (value0 + 0.0001) - 1) <= stable_rate:
                if abs(value) < threshold:
                    value0 = value
                    continue
                count += 1
                value0 = value
                if count >= consecutive_count:
                    start_record[file].append(j)
                    threshold = abs(value0 * 1.1)
                    count = 0
            else:
                count = 0
                value0 = value
    return start_record


record = find_stable_start('P', 5)
print()

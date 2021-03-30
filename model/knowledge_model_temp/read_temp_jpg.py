import sys
import json
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append('data/')
from read_PLAID_data import read_source_data, find_temp_start, find_temp_start_pricisely

start_index = find_temp_start_pricisely('S', 20)
source_dir = 'data/source/submetered_new/'

length1 = 3000
t1 = range(length1)
length2 = 3000
t2 = range(length2)
with open(
        '/home/chaofan/powerknowledge/data/source/metadata_submetered2.1.json',
        'r',
        encoding='utf8') as load_meta:
    meta = json.load(load_meta)
total_len = len(start_index)
count = 0

# for file, start_period in start_index.items():
#     plt.figure()
#     plt.cla()
#     if start_period > 0:
#         start_period -= 1
#     count += 1
#     file_dir = source_dir + file
#     Voltage, Current = read_source_data(file_dir,
#                                         offset=start_period * 500,
#                                         length=length1)
#     Stable_V, Stable_I = read_source_data(file_dir,
#                                           offset=(start_period + 10) * 500,
#                                           length=length2)
#     tem = np.array(Current) - np.array(Stable_I)

#     ax1 = plt.subplot(321)
#     plt.plot(t1, Voltage)
#     plt.grid(alpha=0.5, linestyle='-.')
#     plt.title('swith on')
#     plt.ylabel('Instantaneous voltage')

#     ax2 = plt.subplot(323)
#     plt.plot(t1, Current)
#     plt.grid(alpha=0.5, linestyle='-.')
#     plt.xlabel('Sampling points with 30kHz')
#     plt.ylabel('Instantaneous current')

#     ax3 = plt.subplot(322)
#     plt.plot(t2, Stable_V)
#     plt.grid(alpha=0.5, linestyle='-.')
#     plt.title('stable')
#     # plt.ylabel('Instantaneous voltage')

#     ax4 = plt.subplot(324)
#     plt.plot(t2, Stable_I)
#     plt.grid(alpha=0.5, linestyle='-.')
#     plt.xlabel('Sampling points with 30kHz')
#     # plt.ylabel('Instantaneous current')

#     ax5 = plt.subplot(325)
#     plt.plot(t2, tem)
#     plt.grid(alpha=0.5, linestyle='-.')
#     # plt.xlabel('Sampling points with 30kHz')

#     plt.suptitle(file + '(' + meta[file[0:-4]]['appliance']['type'] + '-' +
#                  meta[file[0:-4]]['appliance']['status'] + ')')
#     plt.savefig('model/knowledge_model_temp/jpg/total/' + file[0:-4] + '.jpg',
#                 dpi=600)

#     plt.close()
#     print('dealing...:%03d/%03d' % (count, total_len))

for file, start_point in start_index.items():
    plt.figure(figsize=(8, 6))
    plt.cla()
    count += 1
    file_dir = source_dir + file
    Voltage, Current = read_source_data(file_dir,
                                        offset=start_point,
                                        length=length1)
    Stable_V, Stable_I = read_source_data(file_dir,
                                          offset=start_point + 3000,
                                          length=length2)
    tem = np.array(Current) - np.array(Stable_I)

    ax1 = plt.subplot(411)
    plt.plot(t1, Voltage)
    plt.grid(alpha=0.5, linestyle='-.')
    plt.title('Voltage')
    plt.xticks([])

    ax2 = plt.subplot(412)
    plt.plot(t1, Current)
    plt.grid(alpha=0.5, linestyle='-.')
    plt.title('Instantaneous Current')
    plt.xticks([])

    ax4 = plt.subplot(413)
    plt.plot(t2, Stable_I)
    plt.grid(alpha=0.5, linestyle='-.')
    plt.title('Stable Current')
    plt.xticks([])
    
    ax5 = plt.subplot(414)
    plt.plot(t2, tem)
    plt.grid(alpha=0.5, linestyle='-.')
    plt.title('Difference Current')
    plt.xlabel('Sampling points with 30kHz')

    plt.suptitle(file + '(' + meta[file[0:-4]]['appliance']['type'] + '-' +
                 meta[file[0:-4]]['appliance']['status'] + ')')
    plt.savefig('model/knowledge_model_temp/jpg/total/' + file[0:-4] + '.jpg',
                dpi=600)

    plt.close()
    print('dealing...:%03d/%03d' % (count, total_len))
import sys
import matplotlib.pyplot as plt
import json
sys.path.append('data/')
from read_PLAID_data import read_source_data, find_temp_start

start_index = find_temp_start('P', 5)
source_dir = 'data/source/submetered_new/'

length1 = 3000
t1 = range(length1)
length2 = 1500
t2 = range(length2)
with open(
        '/home/chaofan/powerknowledge/data/source/metadata_submetered2.1.json',
        'r',
        encoding='utf8') as load_meta:
    meta = json.load(load_meta)
total_len = len(start_index)
count = 0

for file, start_cir in start_index.items():
    plt.figure()
    plt.cla()
    if start_cir > 0:
        start_cir -= 1
    count += 1
    file_dir = source_dir + file
    Voltage, Current = read_source_data(file_dir,
                                        offset=start_cir * 500,
                                        length=length1)
    Stable_V, Stable_I = read_source_data(file_dir,
                                          offset=(start_cir + 10) * 500,
                                          length=length2)

    ax1 = plt.subplot(221)
    plt.plot(t1, Voltage)
    plt.grid(alpha=0.5, linestyle='-.')
    plt.title('swith on')
    plt.ylabel('Instantaneous voltage')

    ax2 = plt.subplot(223)
    plt.plot(t1, Current)
    plt.grid(alpha=0.5, linestyle='-.')
    plt.xlabel('Sampling points with 30kHz')
    plt.ylabel('Instantaneous current')

    ax3 = plt.subplot(222)
    plt.plot(t2, Stable_V)
    plt.grid(alpha=0.5, linestyle='-.')
    plt.title('stable')
    # plt.ylabel('Instantaneous voltage')

    ax2 = plt.subplot(224)
    plt.plot(t2, Stable_I)
    plt.grid(alpha=0.5, linestyle='-.')
    plt.xlabel('Sampling points with 30kHz')
    # plt.ylabel('Instantaneous current')

    plt.suptitle(file + '(' + meta[file[0:-4]]['appliance']['type'] + ')')
    plt.savefig('model/knowledge_model_temp/jpg/' + file[0:-4] + '.jpg',
                dpi=300)

    plt.close()
    print('dealing...:%03d/%03d' % (count, total_len))

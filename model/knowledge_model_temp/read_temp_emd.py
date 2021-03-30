import sys
import json
import matplotlib
import numpy as np
import os
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs

import matplotlib.pyplot as plt
sys.path.append('data/')
from read_PLAID_data import read_source_data

source_dir = 'data/source/submetered_new_pured/source/'

with open(
        '/home/chaofan/powerknowledge/data/source/metadata_submetered2.1.json',
        'r',
        encoding='utf8') as load_meta:
    meta = json.load(load_meta)

length1 = 3000
length2 = 3000
t = range(3000)
csv_dir = os.listdir(source_dir)
for file in csv_dir:
    file_dir = source_dir + file
    Switch_V, Switch_I = read_source_data(file_dir, offset=0, length=length1)
    Stable_V, Stable_I = read_source_data(file_dir,
                                          offset=3000,
                                          length=length2)
    tem = np.array(Switch_I) - np.array(Stable_I)

    decomposer = EMD(tem, n_imfs=3)
    imfs = decomposer.decompose()
    plot_imfs(tem, imfs, t)

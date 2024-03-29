import sys
import json
import os
sys.path.append('data/')
from read_PLAID_data import read_source_data, slider_plot

source_dir = 'data/source/submetered_new_pured/source/'
file_list = os.listdir(source_dir)
# with open('model/find_result.txt', 'r') as f:
#     file_list = []
#     for line in f.readlines():
#         line = line.strip('\n')
#         line = line.split(':')
#         file = line[0][1:-1]
#         if '.csv' in file:
#             file_list.append(file)
file_dir = source_dir + '1670.csv'
Switch_V, Switch_I = read_source_data(file_dir)
slider_plot(Switch_I)
with open(
        '/home/chaofan/powerknowledge/data/source/metadata_submetered2.1.json',
        'r',
        encoding='utf8') as load_meta:
    meta = json.load(load_meta)
count = 0

for i, file in enumerate(file_list):
    file_dir = source_dir + file
    Switch_V, Switch_I = read_source_data(file_dir)
    print(file + '(' + meta[file[0:-4]]['appliance']['type'] + '-' +
          meta[file[0:-4]]['appliance']['status'] + ')')
    slider_plot(Switch_I)

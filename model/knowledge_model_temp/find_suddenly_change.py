import sys
import os
import numpy as np
sys.path.append('data/')
from read_PLAID_data import read_source_data, read_index

source_dir = 'data/source/submetered_new_pured/source/'
type_index = read_index('type')
skip_list = type_index['Compact Fluorescent Lamp'] + type_index['Laptop']
csv_dir = os.listdir(source_dir)
record = {}
for file in csv_dir:
    if int(file[0:-4]) in skip_list:
        continue
    file_dir = source_dir + file
    Switch_V, Switch_I = read_source_data(file_dir, offset=0, length=1500)
    Total_V, Total_I = read_source_data(file_dir, offset=0)
    I_max = np.max(abs(Switch_I))
    # slide_arr = np.zeros(10)
    value0 = 0
    # Threshold = np.sum(slide_arr) / len(slide_arr)
    Threshold = 0.01256 * I_max
    # diff = 0
    slide_width = 10

    for i, value in enumerate(Total_I):
        if i == 0:
            value0 = value
            continue

        if abs(value - value0) > 10 * Threshold:
            if abs(np.average(Total_I[i+1:i+6])-np.average(Total_I[i-5:i]))> 10 * Threshold:
                if file not in record.keys():
                    record[file] = []
                    record[file].append(i)
                    value0 = value
                    continue
                record[file].append(i)

        if abs(value) > I_max:
            I_max = abs(value)
            Threshold = 0.01256 * I_max
        # diff = abs(value - value0)
        # if diff == 0:
        #     continue
        # slide_arr = np.delete(slide_arr, 0, axis=0)
        # slide_arr = np.insert(slide_arr, slide_arr.size, diff)
        # Threshold = np.sum(slide_arr) / len(slide_arr)
        value0 = value

with open('model/find_result.txt', 'w') as f:
    for key in record:
        f.write('\n')
        f.writelines('"' + str(key) + '": ' + str(record[key]))

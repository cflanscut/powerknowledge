import sys
import pandas as pd

sys.path.append('data/')
from read_PLAID_data import read_source_data, find_temp_start_pricisely

source_dir = 'data/source/submetered_new/'
new_dir = 'data/source/submetered_new_pured/'
start_index = find_temp_start_pricisely('P', 5)
count = 0
for file, start_row in start_index.items():
    count += 1
    df = pd.read_csv(source_dir + file, skiprows=start_row)
    df.to_csv(new_dir + file)
    print('dealing...:%03d/%03d' % (count, len(start_index)))

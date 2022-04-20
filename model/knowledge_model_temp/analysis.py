import pandas as pd
# import sys
import json
# sys.path.append('data/')
# from read_PLAID_data import read_index
fit_result = pd.read_csv('model/knowledge_model_temp/fit_result_new.csv',
                         skiprows=1)
x = fit_result.iloc[:, 2:14]
y = fit_result.iloc[:, 14]
fit_result = fit_result.values.tolist()
# fit_result = sorted(fit_result, key=(lambda x: x[6]))

with open(
        '/home/chaofan/powerknowledge/data/source/metadata_submetered2.1.json',
        'r',
        encoding='utf8') as load_meta:
    meta = json.load(load_meta)

for fr in fit_result:
    # if fr[6] >= 0.0015:
    #     break
    label = meta[fr[0][0:-4]]['appliance']['type']
    fr.append(label)

pd.DataFrame(fit_result).to_csv('model/knowledge_model_temp/fit_result.csv')

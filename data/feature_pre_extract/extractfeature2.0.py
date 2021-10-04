from features import Features
import os
import pandas as pd
import json
import numpy as np

power_frequency = 60
sampling_frequency = 30000

path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
source_dir = 'source/submetered_new'
process_dir = 'data/source/submetered_process2.1/'
meta_path = path + '/source/metadata_submetered2.0.json'
csv_dir = os.listdir(os.path.join(path, source_dir))


def get_dic_value_by_k(dict_, k):
    if k in dict_:
        return dict_[k]
    else:
        for v in dict_.values():
            if isinstance(v, dict):
                value = get_dic_value_by_k(v, k)
                if value is not None:
                    return value
        return None


feature = Features(sampling_frequency=sampling_frequency,
                   is_fft=True,
                   eval_per=1 / 60)
with open(meta_path) as data_file:
    meta = json.load(data_file)
load_transformer = {'I': 0, 'R': 1, 'NL': 0}
for i, file in enumerate(csv_dir):
    soucre_data = pd.read_csv(os.path.join(path, source_dir, file),
                              names=["I", "U"])
    feature(soucre_data['I'], soucre_data['U'])
    load = get_dic_value_by_k(meta[file[0:-4]], "load")
    load = load_transformer[load]
    is_heat = get_dic_value_by_k(meta[file[0:-4]], "is_heat")
    is_cold = get_dic_value_by_k(meta[file[0:-4]], "is_cool")
    is_light = get_dic_value_by_k(meta[file[0:-4]], "is_light")
    is_rotate = get_dic_value_by_k(meta[file[0:-4]], "is_rotate")
    data_len = len(feature.data_i_mean_list)
    dataframe = pd.concat(
        [
            pd.DataFrame({'load': data_len * [load]}),
            pd.DataFrame({'is_heat': data_len * [is_heat]}),
            pd.DataFrame({'is_cold': data_len * [is_cold]}),
            pd.DataFrame({'is_light': data_len * [is_light]}),
            pd.DataFrame({'is_rotate': data_len * [is_rotate]}),
            pd.DataFrame({'i_mean': feature.data_i_mean_list}),
            pd.DataFrame({'i_pp': feature.data_i_pp_list}),
            pd.DataFrame({'i_rms': feature.data_i_rms_list}),
            pd.DataFrame({'i_wave_factor': feature.data_i_wave_factor_list}),
            pd.DataFrame({'i_pp_rms': feature.data_i_pp_rms_list}),
            pd.DataFrame({'i_thd': feature.data_i_thd_list}),
            # pd.DataFrame({'u_mean': feature.data_u_mean_list}),
            # pd.DataFrame({'u_pp': feature.data_u_pp_list}),
            # pd.DataFrame({'u_rms': feature.data_u_rms_list}),
            pd.DataFrame({'P': feature.P_list}),
            pd.DataFrame({'Q': feature.Q_list}),
            pd.DataFrame({'S': feature.S_list}),
            pd.DataFrame({'P_F': feature.P_F_list}),
        ],
        axis=1)

    u_hm = np.array(feature.u_i_fft_list['U_hm']).transpose()
    u_hp = np.array(feature.u_i_fft_list['U_hp']).transpose()
    i_hm = np.array(feature.u_i_fft_list['I_hm']).transpose()
    i_hp = np.array(feature.u_i_fft_list['I_hp']).transpose()
    z_hm = np.array(feature.u_i_fft_list['Z_hm']).transpose()
    z_hp = np.array(feature.u_i_fft_list['Z_hp']).transpose()

    for times in range(1, u_hm.shape[0]):
        uhm = pd.DataFrame({'u_hm{}'.format(times): u_hm[times, :]})
        uhp = pd.DataFrame({'u_hp{}'.format(times): u_hp[times, :]})
        ihm = pd.DataFrame({'i_hm{}'.format(times): i_hm[times, :]})
        ihp = pd.DataFrame({'i_hp{}'.format(times): i_hp[times, :]})
        zhm = pd.DataFrame({'z_hm{}'.format(times): z_hm[times, :]})
        zhp = pd.DataFrame({'z_hp{}'.format(times): z_hp[times, :]})
        dataframe = pd.concat(
            [
                dataframe,
                # uhm,
                # uhp,
                ihm,
                ihp,
                zhm,
                zhp
            ],
            axis=1)

    dataframe.to_csv(process_dir + file, index=True, sep=',')
    print('正在处理第{}个数据'.format(i))

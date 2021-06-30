from features import Features
import os
import pandas as pd
import numpy as np

source_dir = 'data/source/HIOKI_xinpei/raw'  # 修改
process_dir = 'data/source/HIOKI_xinpei/process/total'
csv_dir = os.listdir(source_dir)
feature = Features(
    sampling_frequency=20000,  # 修改
    power_frequency=50,
    is_fft=True,
    eval_per=1 / 50,  # 修改
    is_wavelet=True,
    wt_level=6)
for i, file in enumerate(csv_dir):
    # if file != '1.csv':
    #     continue
    if '.csv' not in file:
        continue
    soucre_data = pd.read_csv(os.path.join(source_dir, file),
                              names=["U", "I"],
                              skiprows=9,
                              usecols=[1, 2])  # 修改

    feature(soucre_data['I'], soucre_data['U'])

    dataframe = pd.concat(
        [
            pd.DataFrame({'i_mean': feature.data_i_mean_list}),
            pd.DataFrame({'i_pp': feature.data_i_pp_list}),
            pd.DataFrame({'i_rms': feature.data_i_rms_list}),
            pd.DataFrame({'i_wave_factor': feature.data_i_wave_factor_list}),
            pd.DataFrame({'i_pp_rms': feature.data_i_pp_rms_list}),
            pd.DataFrame({'i_thd': feature.data_i_thd_list}),
            pd.DataFrame({'pure_thd': feature.data_pure_thd_list}),
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
    for t in range(2, u_hm.shape[0]):
        i_hm[t, :] = np.round(i_hm[t, :] / (i_hm[1, :]+0.000001), 5)
        z_hm[t, :] = np.round(z_hm[t, :] / (z_hm[1, :]+0.000001), 5)

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

#!/home/chaofan/anaconda3/bin/python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2021/01/09 15:14:27
@Author      :chenxinpei
@version      :1.0
'''
import numpy as np
from base_features import BaseFeatures
from expert_features import ExpertFeatures


class Features:
    """
    基础特征类，继承于FeaturesPar类，用于计算一段较长数据，对长数据进行分段从而得到一系列特征序列
    """
    def __init__(self,
                 sampling_frequency,
                 eval_per=0.02,
                 use_periods=1,
                 is_fft=False,
                 is_wavelet=False,
                 wt_level=0,
                 wt_name='db3'):
        """BaseFeatures类的初始化

        :param sampling_frequency: 采样频率，当要进行傅里叶变换即is_fft为True时需指定
        :param eval_per: 每多长时间进行一次计算，默认为每0.02s计算一次
        :param use_periods: 每次计算使用多少个周期，默认为1周期
        :param is_fft: 是否进行傅里叶计算，True时结果在属性data_fft中
        :param is_wavelet: 是否需要进行小波变换相关计算，True时结果在属性data_wt中
        :param wt_level: 小波变换层数，当要进行小波变换时需指定
        :param wt_name: 小波名称，默认为db3，可修改
        """
        self.sampling_frequency = sampling_frequency
        self.power_frequency = 60  # 电源频率
        self.num_per_periods = int(self.sampling_frequency /
                                   self.power_frequency)
        self.is_fft = is_fft
        self.is_wavelet = is_wavelet
        if self.is_wavelet:
            if wt_level == 0:
                # 若要做小波变换，则需确定小波种类和分析的阶数，否则抛出错误
                raise Exception('请确定小波变换阶数（wt_level）的值，必要时修改小波种类，以便进行小波计算')
            else:
                self.wt_level = wt_level
                self.wt_name = wt_name
        self.eval_per = eval_per
        self.use_periods = use_periods
        self.__base_feature = BaseFeatures(is_fft, False, sampling_frequency,
                                           wt_level, wt_name)
        self.__expert_feature = ExpertFeatures(is_fft, sampling_frequency)

    def __call__(self, data_i, data_u=None):
        """获取各种特征序列

        :param data_i: 电流数据
        :param data_u: 电压数据，当不输入电压数据时，只计算电流数据
        :return:
        """
        data_i = np.array(data_i)
        self.__data_i = data_i  # 原始电流数据
        self.__data_i_mean_list = []  # 电流数据平均值列表
        self.__data_i_pp_list = []  # 电流数据峰峰值列表
        self.__data_i_rms_list = []  # 电流数据有效值列表
        self.__data_i_wave_factor_list = []  # 电流数据波形因数列表
        self.__data_i_pp_rms_list = []  # 电流数据峰均比列表
        self.__data_i_thd_list = []
        self.__data_fft_list = {"hm": [], "hp": []}
        self.__data_wt_whole = None  # 电流数据的小波分析
        # 整段数据切成数组，并通过BaseFeatures类依次append
        self.cut_data_i = self.__cut_data(data_i)  # 分段后的电流数据
        for i in range(len(self.cut_data_i)):
            self.__base_feature(self.cut_data_i[i])
            self.__data_i_mean_list.append(self.__base_feature.mean)
            self.__data_i_pp_list.append(self.__base_feature.pp)
            self.__data_i_rms_list.append(self.__base_feature.rms)
            self.__data_i_wave_factor_list.append(
                self.__base_feature.wave_factor)
            self.__data_i_pp_rms_list.append(self.__base_feature.pp_rms)
            if (self.is_fft and (self.__base_feature.fft is not None)
                    and (data_u is not None)):
                self.__data_fft_list["hm"].append(
                    self.__base_feature.fft["hm"])
                self.__data_fft_list["hp"].append(
                    self.__base_feature.fft["hp"])
                self.__data_i_thd_list.append(self.__base_feature.thd)
        if self.is_wavelet:
            self.__data_wt_whole = BaseFeatures.get_wt_data(
                self.__data_i, self.wt_name, self.wt_level)

        self.__data_u = None
        self.__data_u_mean_list = []  # 电压数据平均值列表
        self.__data_u_pp_list = []  # 电压数据峰峰值列表
        self.__data_u_rms_list = []  # 电压数据有效值列表
        self.__data_u_wave_factor_list = []  # 电压数据波形因数列表
        self.__data_u_pp_rms_list = []  # 电压数据峰均比列表
        self.__P_list = []  # 有功功率列表
        self.__S_list = []  # 视在功率列表
        self.__Q_list = []  # 无功功率列表
        self.__P_F_list = []  # 功率因数列表
        self.__u_i_fft_list = {
            "U_hm": [],
            "U_hp": [],
            "I_hm": [],
            "I_hp": [],
            "Z_hm": [],
            "Z_hp": []
        }  # 电压电流FFT结果列表
        if data_u is not None:
            data_u = np.array(data_u)
            self.data_p = data_u * data_i
            self.__data_u = data_u  # 原始电压数据
            self.cut_data_u = self.__cut_data(data_u)  # 分段后的电压数据
            for i in range(len(self.cut_data_u)):
                self.__base_feature(self.cut_data_u[i])
                self.__data_u_mean_list.append(self.__base_feature.mean)
                self.__data_u_pp_list.append(self.__base_feature.pp)
                self.__data_u_rms_list.append(self.__base_feature.rms)
                self.__expert_feature(self.cut_data_u[i], self.cut_data_i[i])
                self.__P_list.append(self.__expert_feature.P)
                self.__S_list.append(self.__expert_feature.S)
                self.__Q_list.append(self.__expert_feature.Q)
                self.__P_F_list.append(self.__expert_feature.P_F)
                if self.is_fft and self.__expert_feature.u_i_fft:
                    self.__u_i_fft_list["U_hm"].append(
                        self.__expert_feature.u_i_fft["U_hm"])
                    self.__u_i_fft_list["U_hp"].append(
                        self.__expert_feature.u_i_fft["U_hp"])
                    self.__u_i_fft_list["I_hm"].append(
                        self.__expert_feature.u_i_fft["I_hm"])
                    self.__u_i_fft_list["I_hp"].append(
                        self.__expert_feature.u_i_fft["I_hp"])
                    self.__u_i_fft_list["Z_hm"].append(
                        self.__expert_feature.u_i_fft["Z_hm"])
                    self.__u_i_fft_list["Z_hp"].append(
                        self.__expert_feature.u_i_fft["Z_hp"])
        return self

    def __cut_data(self, data):
        """对一段数据分为几段相同长度数据

        :param data: 要进行分段的数据
        :return: 分段后的数据
        """
        data = np.array(data)
        cut_data = []
        eval_per_num = int(self.sampling_frequency * self.eval_per)  # 采样间隔
        used_num = int(self.num_per_periods * self.use_periods)  # 采样长度
        for i in range(int((len(data) - used_num) / eval_per_num) + 1):
            cut_data.append(data[i * eval_per_num:i * eval_per_num + used_num])
        return cut_data

    @property
    def data_i(self):
        """设置属性只读"""
        return self.__data_i

    @property
    def data_i_mean_list(self):
        """设置属性只读"""
        return self.__data_i_mean_list

    @property
    def data_i_pp_list(self):
        """设置属性只读"""
        return self.__data_i_pp_list

    @property
    def data_i_rms_list(self):
        """设置属性只读"""
        return self.__data_i_rms_list

    @property
    def data_i_wave_factor_list(self):
        """设置属性只读"""
        return self.__data_i_wave_factor_list

    @property
    def data_i_pp_rms_list(self):
        """设置属性只读"""
        return self.__data_i_pp_rms_list

    @property
    def data_wt_whole(self):
        """设置属性只读"""
        return self.__data_wt_whole

    @property
    def data_fft_list(self):
        """设置属性只读"""
        return self.__data_fft_list

    @property
    def data_i_thd_list(self):
        """设置属性只读"""
        return self.__data_i_thd_list

    @property
    def data_u(self):
        """设置属性只读"""
        return self.__data_u

    @property
    def data_u_mean_list(self):
        """设置属性只读"""
        return self.__data_u_mean_list

    @property
    def data_u_pp_list(self):
        """设置属性只读"""
        return self.__data_u_pp_list

    @property
    def data_u_rms_list(self):
        """设置属性只读"""
        return self.__data_u_rms_list

    @property
    def data_u_wave_factor_list(self):
        """设置属性只读"""
        return self.__data_u_wave_factor_list

    @property
    def data_u_pp_rms_list(self):
        """设置属性只读"""
        return self.__data_u_pp_rms_list

    @property
    def P_list(self):
        """设置属性只读"""
        return self.__P_list

    @property
    def S_list(self):
        """设置属性只读"""
        return self.__S_list

    @property
    def Q_list(self):
        """设置属性只读"""
        return self.__Q_list

    @property
    def P_F_list(self):
        """设置属性只读"""
        return self.__P_F_list

    @property
    def u_i_fft_list(self):
        """设置属性只读"""
        return self.__u_i_fft_list

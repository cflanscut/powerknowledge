import numpy as np
from base_features import BaseFeatures


class ExpertFeatures:
    """
    专家特征类，用于计算电压电流组合特征
    """
    def __init__(self, is_fft=False, sampling_frequency=0):
        """ExpertFeatures类的初始化

        :param is_fft: 是否进行傅里叶计算，True时结果在属性data_fft中
        :param sampling_frequency: 采样频率，当要进行傅里叶变换即is_fft为True时需指定
        """
        self.power_frequency = 60  # 电源频率
        self.__is_fft = is_fft
        if self.__is_fft:
            if sampling_frequency == 0:
                # 若要做fft，则需确定采样频率的值，否则抛出错误
                raise Exception('请确定采样频率（sampling_frequency）的值，以便进行FFT计算')
            else:
                self.sampling_frequency = sampling_frequency

    def __call__(self, data_u, data_i):
        """更新实例的特征

        :param data_u: 要计算的原始电压数据
        :param data_i: 要计算的原始电流数据
        :return: 更新完各特征的实例
        """
        data_u = np.array(data_u)
        data_i = np.array(data_i)
        self.data_p = data_u * data_i
        self.__source_data_u = data_u  # 原始电压数据
        self.__source_data_i = data_i  # 原始电流数据
        self.__P = self.get_P(data_u, data_i)  # 计算有功功率
        self.__S = self.get_S(data_u, data_i)  # 计算视在功率
        self.__Q = np.sqrt(np.square(self.S) - np.square(self.P))  # 计算无功功率
        self.__P_F = self.get_factor(data_u, data_i)  # 计算功率因数
        self.__u_i_fft = None
        if self.__is_fft:
            self.__u_i_fft = self.get_ui_harmonic(data_u, data_i,
                                                  self.sampling_frequency,
                                                  self.power_frequency)
        return self

    @staticmethod
    def get_P(u_data, i_data):
        """计算有功功率

        :param u_data: 原始电压数据
        :param i_data: 原始电流数据
        :return: 有功功率
        """
        return np.mean(i_data * u_data)

    @staticmethod
    def get_S(u_data, i_data):
        """计算视在功率

        :param u_data: 原始电压数据
        :param i_data: 原始电流数据
        :return: 视在功率
        """
        return BaseFeatures.get_rms(i_data) * BaseFeatures.get_rms(u_data)

    @staticmethod
    def get_Q(u_data, i_data):
        """计算无功功率

        :param u_data: 原始电压数据
        :param i_data: 原始电流数据
        :return: 无功功率
        """
        s = ExpertFeatures.get_S(u_data, i_data)
        p = ExpertFeatures.get_P(u_data, i_data)
        return np.sqrt(s * s - p * p)

    @staticmethod
    def get_factor(u_data, i_data):
        """计算功率因数

        :param u_data: 原始电压数据
        :param i_data: 原始电流数据
        :return: 功率因数
        """
        return ExpertFeatures.get_P(i_data, u_data) / ExpertFeatures.get_S(
            i_data, u_data)

    @staticmethod
    def get_ui_harmonic(u_data, i_data, sampling_frequency, power_frequency):
        """对原始电压电流数据进行傅里叶变换，结果以电压基波为基准

        :param u_data: 原始电压波形
        :param i_data: 原始电流波形
        :param sampling_frequency: 采样频率
        :param power_frequency: 电源频率
        :return: 傅里叶计算结果
        """
        u_data = np.array(u_data)
        i_data = np.array(i_data)
        freq, u_hm, u_hp = BaseFeatures.fft_to_harmonic(
            u_data, sampling_frequency, power_frequency)
        freq, i_hm, i_hp = BaseFeatures.fft_to_harmonic(
            i_data, sampling_frequency, power_frequency)
        # 改为以电压基波为基准，即电压基波相位为0
        for i in range(1, 32):
            if i_hm[i] != 0:
                i_hp[i] -= i * u_hp[1]
            while i_hp[i] < 0:
                i_hp[i] += 360
            while i_hp[i] > 360:
                i_hp[i] -= 360
        for i in range(2, 32):
            if u_hm[i] != 0:
                u_hp[i] -= i * u_hp[1]
            while u_hp[i] < 0:
                u_hp[i] += 360
            while u_hp[i] >= 360:
                u_hp[i] -= 360
        u_hp[1] = 0
        # 将电流对齐到电压操作
        harmonic = {
            "freq": freq,
            "U_hm": u_hm,
            "U_hp": u_hp,
            "I_hm": i_hm,
            "I_hp": i_hp,
            "Z_hm": np.true_divide(u_hm, i_hm),
            "Z_hp": u_hp - i_hp
        }
        return harmonic

    @property
    def source_data_u(self):
        return self.__source_data_u

    @property
    def source_data_i(self):
        return self.__source_data_i

    @property
    def P(self):
        return self.__P

    @property
    def S(self):
        return self.__S

    @property
    def Q(self):
        return self.__Q

    @property
    def P_F(self):
        return self.__P_F

    @property
    def u_i_fft(self):
        return self.__u_i_fft

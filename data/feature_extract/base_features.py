import numpy as np
import pywt


class BaseFeatures:
    """
    基础特征的父类，包含常见基础特征的计算方法，用于对整段数据进行单次计算
    """
    def __init__(self,
                 is_fft=False,
                 is_wavelet=False,
                 sampling_frequency=0,
                 wt_level=0,
                 wt_name='db3'):
        """FeaturesPar类的初始化

        :param is_fft: 是否进行傅里叶计算，True时结果在属性data_fft中
        :param is_wavelet: 是否需要进行小波变换相关计算，True时结果在属性data_wt中
        :param sampling_frequency: 采样频率，当要进行傅里叶变换即is_fft为True时需指定
        :param wt_level: 小波变换层数，当要进行小波变换时需指定
        :param wt_name: 小波名称，默认为db3，可修改
        """
        self.power_frequency = 60  # 电源频率
        self.__is_fft = is_fft
        self.__is_wavelet = is_wavelet
        if self.__is_fft:
            if sampling_frequency == 0:
                # 若要做fft，则需确定采样频率的值，否则抛出错误
                raise Exception('请确定采样频率（sampling_frequency）的值，以便进行FFT计算')
            else:
                self.sampling_frequency = sampling_frequency
        if self.__is_wavelet:
            if wt_level == 0:
                # 若要做小波变换，则需确定小波种类和分析的阶数，否则抛出错误
                raise Exception('请确定小波变换阶数（wt_level）的值，必要时修改小波种类，以便进行小波计算')
            else:
                self.wt_level = wt_level
                self.wt_name = wt_name

    def __call__(self, data):
        """更新实例的特征

        :param data: 要进行计算的数据
        :return: 更新后的实例
        """
        data = np.array(data)
        self.__data_source = data  # 原始数据
        self.__data_mean = self.get_mean(data)  # 数据平均值
        self.__data_absmean = self.get_absmean(data)  # 数据绝对均值
        self.__data_pp = self.get_p_p_value(data)  # 数据峰峰值
        self.__data_rms = self.get_rms(data)  # 数据有效值
        self.__data_wave_factor = self.get_wave_factor(data)  # 计算波形因数
        self.__data_pp_rms = self.get_pp_rms(data)  # 计算峰均比
        self.__data_fft = None
        self.__data_wt = None
        self.__data_thd = None
        if self.__is_wavelet:
            self.__data_wt = self.get_wt_data(data, self.wt_name,
                                              self.wt_level)
        if self.__is_fft:
            self.__data_fft = {
                "freq": (self.fft_to_harmonic(data, self.sampling_frequency,
                                              self.power_frequency))[0],
                "hm": (self.fft_to_harmonic(data, self.sampling_frequency,
                                            self.power_frequency))[1],
                "hp": (self.fft_to_harmonic(data, self.sampling_frequency,
                                            self.power_frequency))[2]
            }
            self.__data_thd = np.mean(np.square(
                (self.__data_fft["hm"])[2:])) / (self.__data_fft["hm"])[1]
        return self

    @property
    def source(self):
        """设置属性只读"""
        return self.__data_source

    @property
    def mean(self):
        """设置属性只读"""
        return self.__data_mean

    @property
    def absmean(self):
        return self.__data_absmean

    @property
    def pp(self):
        """设置属性只读"""
        return self.__data_pp

    @property
    def rms(self):
        """设置属性只读"""
        return self.__data_rms

    @property
    def wave_factor(self):
        """设置属性只读"""
        return self.__data_wave_factor

    @property
    def pp_rms(self):
        """设置属性只读"""
        return self.__data_pp_rms

    @property
    def wt(self):
        """设置属性只读"""
        return self.__data_wt

    @property
    def fft(self):
        """设置属性只读"""
        return self.__data_fft

    @property
    def thd(self):
        """设置属性只读"""
        return self.__data_thd

    @staticmethod
    def get_mean(data):
        """计算数据平均值

        :param data: 要进行计算平均值的数据
        :return: 平均值
        """
        data = np.array(data).reshape(-1)
        return np.mean(data)

    @staticmethod
    def get_absmean(data):
        """计算数据绝对均值

        :param data: 要进行计算绝对均值的数据
        :return: 绝对均值
        """
        data = np.array(data).reshape(-1)
        return np.mean(np.abs(data))

    @staticmethod
    def get_p_p_value(data):
        """计算峰峰值

        :param data: 要进行计算峰峰值的数据
        :return: 峰峰值
        """
        data = np.array(data).reshape(-1)
        return np.max(data) - np.min(data)

    @staticmethod
    def get_rms(data):
        """计算有效值

        :param data: 要进行有效值计算的数据
        :return: 有效值
        """
        square_data = np.square(data)
        return np.sqrt(np.mean(square_data))

    @staticmethod
    def get_wave_factor(data):
        """计算波形因数

        :param data: 要进行波形因数计算的数据
        :return: 波形因数
        """
        return BaseFeatures.get_rms(data) / np.mean(np.abs(data))

    @staticmethod
    def get_pp_rms(data):
        """计算均峰比

        :param data: 要进行均峰比计算的数据
        :return: 均峰比
        """
        return BaseFeatures.get_p_p_value(data) / BaseFeatures.get_rms(data)

    @staticmethod
    def fft_to_harmonic(data, sampling_frequency, power_frequency):
        """傅里叶计算

        :param data: 要进行傅里叶计算的数据
        :param sampling_frequency: 数据的采样频率
        :param power_frequency: 数据的电源频率
        :return: 第0至31次谐波的频率，对应的谐波有效值和谐波相角
        """
        data = np.array(data)
        index = np.arange(32) * int(
            np.size(data, 0) / sampling_frequency * power_frequency)
        x = np.fft.fft(data, np.size(data, 0), axis=0) / np.size(data, 0) * 2
        x = x[index]
        x[0] /= np.sqrt(2)
        freq = np.fft.fftfreq(np.size(data, 0), 1 / sampling_frequency)
        freq = freq[index]
        hp = np.angle(x) / np.pi * 180
        hm = np.abs(x) / np.sqrt(2)
        if hp[0] > 90:
            hp[0] -= 180
            hm[0] = -hm[0]
        return freq, hm, hp

    @staticmethod
    def get_wt_data(data, wt_name, wave_level):
        """离散小波变换计算

        :param data: 要进行小波变换的数据
        :param wt_name: 所使用的小波基名称
        :param wave_level: 要进行的小波分析层数
        :return: 小波系数和各层小波能量等
        """
        coeffs = pywt.wavedec(data, wt_name, level=wave_level)  # 小波系数获取
        cd_list = []
        cd_e_list = []
        for i in range(len(coeffs) - 1):
            cd_list.append(coeffs[len(coeffs) - i - 1])
            cd_e_list.append(np.mean(np.square(coeffs[len(coeffs) - i - 1])))
        wt_result = {
            "Coeffs": coeffs,  # 小波系数，coeffs[0]是近似系数，coeffs[1]是最底层细节系数
            "CA": coeffs[0],  # 近似小波系数
            "CD": cd_list,  # 细节小波系数
            "CD_E_list": cd_e_list,  # 各层细节系数能量值
            "CA_E": np.mean(np.square(coeffs[0])),  # 近似系数能量
            "CD_E": np.sum(cd_e_list)  # 细节系数能量
        }
        return wt_result

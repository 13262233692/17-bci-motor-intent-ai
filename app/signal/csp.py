import numpy as np
from scipy import linalg
from typing import Tuple, Optional


class CSP:
    """
    共空间模式 (Common Spatial Patterns, CSP) 特征提取器
    
    CSP是脑机接口中经典的空间滤波算法，通过最大化两类信号的方差比
    来提取具有判别性的空间特征。特别适用于运动想象EEG信号的特征增强。
    """

    def __init__(self, n_components: int = 6, reg: float = 1e-6):
        self.n_components = n_components
        self.reg = reg
        self.filters_: Optional[np.ndarray] = None
        self.patterns_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSP":
        """
        训练CSP空间滤波器
        
        Args:
            X: EEG数据，形状为 (n_trials, n_channels, n_samples)
            y: 标签数组，形状为 (n_trials,)，只支持二分类
            
        Returns:
            self: 训练好的CSP对象
        """
        if X.ndim != 3:
            raise ValueError(f"X必须是3维数组 (n_trials, n_channels, n_samples)，实际是 {X.ndim} 维")

        n_trials, n_channels, n_samples = X.shape

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP只支持二分类，实际有 {len(classes)} 类")

        class_0, class_1 = classes[0], classes[1]

        cov_0 = self._compute_covariance(X[y == class_0])
        cov_1 = self._compute_covariance(X[y == class_1])

        cov_total = cov_0 + cov_1

        eigvals, eigvecs = linalg.eigh(cov_total)
        eigvals = np.maximum(eigvals, self.reg)

        whitening_matrix = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        s_0 = whitening_matrix.T @ cov_0 @ whitening_matrix
        s_1 = whitening_matrix.T @ cov_1 @ whitening_matrix

        eigvals_0, eigvecs_0 = linalg.eigh(s_0)
        eigvals_1, eigvecs_1 = linalg.eigh(s_1)

        idx = np.argsort(eigvals_0)[::-1]
        eigvals_0 = eigvals_0[idx]
        eigvecs_0 = eigvecs_0[:, idx]

        spatial_filters = whitening_matrix @ eigvecs_0

        n_select = min(self.n_components, n_channels // 2)
        select_idx = np.concatenate([
            np.arange(n_select),
            np.arange(n_channels - n_select, n_channels)
        ])

        self.filters_ = spatial_filters[:, select_idx].T
        self.patterns_ = linalg.pinv(spatial_filters[:, select_idx])

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        使用训练好的CSP滤波器提取特征
        
        Args:
            X: EEG数据，形状为 (n_trials, n_channels, n_samples) 
               或 (n_channels, n_samples) 单个样本
            
        Returns:
            特征向量，形状为 (n_trials, n_features) 或 (n_features,)
        """
        if self.filters_ is None:
            raise ValueError("CSP模型尚未训练，请先调用fit()")

        single_sample = X.ndim == 2
        if single_sample:
            X = X[np.newaxis, ...]

        if X.ndim != 3:
            raise ValueError(f"X必须是2维或3维数组，实际是 {X.ndim} 维")

        n_trials, n_channels, n_samples = X.shape

        if n_channels != self.filters_.shape[1]:
            raise ValueError(
                f"通道数不匹配: 期望 {self.filters_.shape[1]}, 实际 {n_channels}"
            )

        features = []
        for i in range(n_trials):
            filtered = self.filters_ @ X[i]
            variances = np.var(filtered, axis=1)
            log_var = np.log(variances + self.reg)
            features.append(log_var)

        features = np.array(features)

        if single_sample:
            return features[0]

        return features

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        训练CSP并提取特征
        
        Args:
            X: EEG数据，形状为 (n_trials, n_channels, n_samples)
            y: 标签数组
            
        Returns:
            特征向量
        """
        self.fit(X, y)
        return self.transform(X)

    def apply_filter(self, X: np.ndarray) -> np.ndarray:
        """
        应用CSP空间滤波，返回滤波后的信号
        
        Args:
            X: EEG数据，形状为 (n_channels, n_samples) 
               或 (n_trials, n_channels, n_samples)
            
        Returns:
            滤波后的信号，形状为 (n_components*2, n_samples)
            或 (n_trials, n_components*2, n_samples)
        """
        if self.filters_ is None:
            raise ValueError("CSP模型尚未训练，请先调用fit()")

        if X.ndim == 2:
            return self.filters_ @ X
        elif X.ndim == 3:
            result = []
            for trial in X:
                result.append(self.filters_ @ trial)
            return np.array(result)
        else:
            raise ValueError(f"X必须是2维或3维数组，实际是 {X.ndim} 维")

    def _compute_covariance(self, X: np.ndarray) -> np.ndarray:
        """
        计算平均协方差矩阵
        
        Args:
            X: EEG数据，形状为 (n_trials, n_channels, n_samples)
            
        Returns:
            平均协方差矩阵，形状为 (n_channels, n_channels)
        """
        n_trials = X.shape[0]
        cov_sum = 0

        for i in range(n_trials):
            trial_data = X[i]
            cov = trial_data @ trial_data.T / trial_data.shape[1]
            cov_sum += cov

        cov_avg = cov_sum / n_trials

        trace = np.trace(cov_avg)
        if trace > 0:
            cov_avg /= trace

        return cov_avg

    def save(self, filepath: str) -> None:
        """保存CSP模型"""
        if self.filters_ is None:
            raise ValueError("没有可保存的模型，请先训练")

        np.savez(
            filepath,
            filters=self.filters_,
            patterns=self.patterns_,
            n_components=np.array([self.n_components]),
            reg=np.array([self.reg]),
        )

    @classmethod
    def load(cls, filepath: str) -> "CSP":
        """加载CSP模型"""
        data = np.load(filepath)
        csp = cls(n_components=int(data["n_components"][0]), reg=float(data["reg"][0]))
        csp.filters_ = data["filters"]
        csp.patterns_ = data["patterns"]
        return csp


class HighPassFilter:
    """
    高通滤波器 - 用于消除基线漂移
    
    移除低于截止频率的信号成分，有效消除：
    - 电极极化导致的直流偏移
    - 患者出汗引起的缓慢基线漂移
    - 运动伪迹产生的超低频干扰
    
    通常使用0.5-1Hz的截止频率
    """

    def __init__(
        self,
        cutoff_freq: float = 1.0,
        sampling_rate: int = 250,
        order: int = 4,
    ):
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        self.order = order

        nyquist = sampling_rate / 2.0
        cutoff = cutoff_freq / nyquist

        from scipy.signal import butter, filtfilt
        self._butter = butter
        self._filtfilt = filtfilt

        self.b, self.a = butter(order, cutoff, btype="high")

    def apply(self, X: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        应用高通滤波
        
        Args:
            X: EEG数据
            axis: 滤波的轴，默认为最后一个轴（时间轴）
            
        Returns:
            滤波后的数据，已消除基线漂移
        """
        return self._filtfilt(self.b, self.a, X, axis=axis)


class BandPassFilter:
    """
    带通滤波器 - 使用FIR滤波器实现
    
    用于EEG信号的频带滤波，常见频段：
    - Mu节律: 8-13 Hz
    - Beta节律: 14-30 Hz
    """

    def __init__(
        self,
        low_freq: float = 8.0,
        high_freq: float = 30.0,
        sampling_rate: int = 250,
        order: int = 4,
    ):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.sampling_rate = sampling_rate
        self.order = order

        nyquist = sampling_rate / 2.0
        low = low_freq / nyquist
        high = high_freq / nyquist

        from scipy.signal import butter, filtfilt
        self._butter = butter
        self._filtfilt = filtfilt

        self.b, self.a = butter(order, [low, high], btype="band")

    def apply(self, X: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        应用带通滤波
        
        Args:
            X: EEG数据
            axis: 滤波的轴，默认为最后一个轴（时间轴）
            
        Returns:
            滤波后的数据
        """
        return self._filtfilt(self.b, self.a, X, axis=axis)


def normalize_signal(signal: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    对EEG信号进行归一化处理
    
    Args:
        signal: 输入信号
        method: 归一化方法: 'zscore' | 'minmax'
        
    Returns:
        归一化后的信号
    """
    if method == "zscore":
        mean = np.mean(signal, axis=-1, keepdims=True)
        std = np.std(signal, axis=-1, keepdims=True) + 1e-8
        return (signal - mean) / std
    elif method == "minmax":
        min_val = np.min(signal, axis=-1, keepdims=True)
        max_val = np.max(signal, axis=-1, keepdims=True) + 1e-8
        return (signal - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"未知的归一化方法: {method}")


def check_numerical_stability(signal: np.ndarray) -> dict:
    """
    检查信号的数值稳定性
    
    Args:
        signal: 输入信号
        
    Returns:
        包含稳定性指标的字典
    """
    result = {
        "has_nan": bool(np.any(np.isnan(signal))),
        "has_inf": bool(np.any(np.isinf(signal))),
        "max_value": float(np.max(np.abs(signal))) if signal.size > 0 else 0.0,
        "min_value": float(np.min(signal)) if signal.size > 0 else 0.0,
        "mean_value": float(np.mean(signal)) if signal.size > 0 else 0.0,
        "std_value": float(np.std(signal)) if signal.size > 0 else 0.0,
        "is_stable": True,
        "issues": [],
    }

    if result["has_nan"]:
        result["is_stable"] = False
        result["issues"].append("NaN values detected")

    if result["has_inf"]:
        result["is_stable"] = False
        result["issues"].append("Inf values detected")

    if result["max_value"] > 1e6:
        result["is_stable"] = False
        result["issues"].append(f"Extreme values detected: max={result['max_value']:.2e}")

    if result["std_value"] < 1e-10:
        result["is_stable"] = False
        result["issues"].append("Near-constant signal detected")

    return result


def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
    """
    移除信号的直流偏移（简单的去均值）
    
    Args:
        signal: 输入信号，形状为 (n_channels, n_samples)
        
    Returns:
        去除直流偏移后的信号
    """
    mean = np.mean(signal, axis=-1, keepdims=True)
    return signal - mean


def clip_extreme_values(
    signal: np.ndarray,
    std_multiplier: float = 10.0,
) -> np.ndarray:
    """
    截断极端值，防止算术溢出
    
    Args:
        signal: 输入信号
        std_multiplier: 标准差倍数，超过此阈值的值将被截断
        
    Returns:
        截断后的信号
    """
    mean = np.mean(signal, axis=-1, keepdims=True)
    std = np.std(signal, axis=-1, keepdims=True) + 1e-8

    upper_bound = mean + std_multiplier * std
    lower_bound = mean - std_multiplier * std

    return np.clip(signal, lower_bound, upper_bound)


def detect_baseline_drift(
    signal: np.ndarray,
    window_size: int = 100,
    threshold: float = 50.0,
) -> dict:
    """
    检测基线漂移
    
    Args:
        signal: 输入信号，形状为 (n_channels, n_samples)
        window_size: 滑动窗口大小（样本数）
        threshold: 漂移阈值（微伏）
        
    Returns:
        漂移检测结果
    """
    n_channels, n_samples = signal.shape

    if n_samples < window_size:
        return {
            "has_drift": False,
            "drift_magnitude": 0.0,
            "max_drift_channel": -1,
            "details": "Insufficient samples",
        }

    drift_magnitudes = np.zeros(n_channels)

    for ch in range(n_channels):
        moving_avg = np.convolve(
            signal[ch],
            np.ones(window_size) / window_size,
            mode="valid",
        )
        drift_range = np.max(moving_avg) - np.min(moving_avg)
        drift_magnitudes[ch] = drift_range

    max_drift = np.max(drift_magnitudes)
    max_drift_ch = np.argmax(drift_magnitudes)

    return {
        "has_drift": bool(max_drift > threshold),
        "drift_magnitude": float(max_drift),
        "max_drift_channel": int(max_drift_ch),
        "channel_drifts": drift_magnitudes.tolist(),
        "threshold": threshold,
    }


def robust_preprocessing_pipeline(
    signal: np.ndarray,
    highpass_filter: Optional[HighPassFilter] = None,
    bandpass_filter: Optional[BandPassFilter] = None,
    sampling_rate: int = 250,
) -> tuple[np.ndarray, dict]:
    """
    鲁棒的预处理流水线 - 包含基线漂移检测与消除
    
    处理步骤：
    1. 直流偏移移除
    2. 极端值截断
    3. 基线漂移检测
    4. 高通滤波（消除基线漂移）
    5. 带通滤波（Mu/Beta频段）
    6. 数值稳定性检查
    
    Args:
        signal: 原始EEG信号，形状为 (n_channels, n_samples)，单位为伏
        highpass_filter: 可选的高通滤波器实例
        bandpass_filter: 可选的带通滤波器实例
        sampling_rate: 采样率
        
    Returns:
        (处理后的信号, 处理状态字典)
    """
    status = {
        "steps_completed": [],
        "drift_detected": False,
        "drift_magnitude": 0.0,
        "numerical_stability": {},
        "input_shape": signal.shape,
        "output_shape": None,
    }

    signal = signal.astype(np.float64)
    status["steps_completed"].append("type_conversion")

    stability = check_numerical_stability(signal)
    status["numerical_stability"]["input"] = stability

    if not stability["is_stable"]:
        if stability["has_nan"] or stability["has_inf"]:
            signal = np.nan_to_num(signal, nan=0.0, posinf=1e6, neginf=-1e6)
            status["steps_completed"].append("nan_inf_fix")

        signal = remove_dc_offset(signal)
        status["steps_completed"].append("dc_offset_removal")

        signal = clip_extreme_values(signal, std_multiplier=10.0)
        status["steps_completed"].append("extreme_value_clipping")

    drift_info = detect_baseline_drift(signal * 1e6, threshold=50.0)
    status["drift_detected"] = drift_info["has_drift"]
    status["drift_magnitude"] = drift_info["drift_magnitude"]
    status["drift_info"] = drift_info

    if drift_info["has_drift"] or highpass_filter is not None:
        if highpass_filter is None:
            highpass_filter = HighPassFilter(
                cutoff_freq=1.0,
                sampling_rate=sampling_rate,
                order=4,
            )
        signal = highpass_filter.apply(signal, axis=-1)
        status["steps_completed"].append("highpass_filtering")

    if bandpass_filter is not None:
        signal = bandpass_filter.apply(signal, axis=-1)
        status["steps_completed"].append("bandpass_filtering")

    signal = normalize_signal(signal, method="zscore")
    status["steps_completed"].append("normalization")

    signal = clip_extreme_values(signal, std_multiplier=5.0)
    status["steps_completed"].append("final_clipping")

    stability_out = check_numerical_stability(signal)
    status["numerical_stability"]["output"] = stability_out
    status["output_shape"] = signal.shape

    return signal, status

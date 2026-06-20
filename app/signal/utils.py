import numpy as np
from typing import Tuple, Optional


def generate_mock_eeg(
    n_channels: int = 64,
    n_samples: int = 500,
    sampling_rate: int = 250,
    class_label: Optional[int] = None,
    noise_level: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成模拟的运动想象EEG数据
    
    用于测试和开发目的，模拟Mu/Beta频段的事件相关去同步化(ERD)现象。
    
    Args:
        n_channels: 通道数
        n_samples: 采样点数
        sampling_rate: 采样率
        class_label: 0=左手, 1=右手, None=随机
        noise_level: 噪声水平
        
    Returns:
        (eeg_data, label) - EEG数据形状为 (n_channels, n_samples)，单位为微伏
    """
    if class_label is None:
        class_label = np.random.randint(0, 2)

    t = np.arange(n_samples) / sampling_rate

    eeg_data = np.random.randn(n_channels, n_samples) * noise_level * 5

    mu_freq = 10.0
    beta_freq = 20.0

    if class_label == 0:
        erd_channels_left = slice(0, n_channels // 4)
        erd_channels_right = slice(n_channels // 2, 3 * n_channels // 4)
        erd_strength_left = 0.3
        erd_strength_right = 0.7
    else:
        erd_channels_left = slice(0, n_channels // 4)
        erd_channels_right = slice(n_channels // 2, 3 * n_channels // 4)
        erd_strength_left = 0.7
        erd_strength_right = 0.3

    for ch in range(n_channels):
        mu_osc = np.sin(2 * np.pi * mu_freq * t + np.random.rand() * 2 * np.pi) * 3
        beta_osc = np.sin(2 * np.pi * beta_freq * t + np.random.rand() * 2 * np.pi) * 1.5

        if ch < n_channels // 4:
            mu_osc *= (1 - erd_strength_left)
            beta_osc *= (1 - erd_strength_left)
        elif n_channels // 2 <= ch < 3 * n_channels // 4:
            mu_osc *= (1 - erd_strength_right)
            beta_osc *= (1 - erd_strength_right)

        eeg_data[ch] += mu_osc + beta_osc

    drift = np.linspace(0, np.random.randn() * 2, n_samples)
    eeg_data += drift

    eeg_data *= 10

    return eeg_data, np.array([class_label])


def generate_eeg_batch(
    n_trials: int = 10,
    n_channels: int = 64,
    n_samples: int = 500,
    sampling_rate: int = 250,
    balanced: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成一批模拟EEG数据
    
    Args:
        n_trials: 试次数量
        n_channels: 通道数
        n_samples: 采样点数
        sampling_rate: 采样率
        balanced: 是否平衡类别
        
    Returns:
        (X, y) - X形状为 (n_trials, n_channels, n_samples)，y形状为 (n_trials,)
    """
    X = np.zeros((n_trials, n_channels, n_samples))
    y = np.zeros(n_trials, dtype=int)

    for i in range(n_trials):
        if balanced:
            label = i % 2
        else:
            label = np.random.randint(0, 2)

        eeg_data, _ = generate_mock_eeg(
            n_channels=n_channels,
            n_samples=n_samples,
            sampling_rate=sampling_rate,
            class_label=label,
            noise_level=1.0,
        )
        X[i] = eeg_data
        y[i] = label

    return X, y


def check_signal_quality(eeg_data: np.ndarray) -> dict:
    """
    检查EEG信号质量
    
    Args:
        eeg_data: EEG数据，形状为 (n_channels, n_samples)
        
    Returns:
        信号质量指标字典
    """
    n_channels, n_samples = eeg_data.shape

    means = np.mean(eeg_data, axis=1)
    stds = np.std(eeg_data, axis=1)
    max_vals = np.max(eeg_data, axis=1)
    min_vals = np.min(eeg_data, axis=1)
    ranges = max_vals - min_vals

    variance_stability = np.std(stds) / (np.mean(stds) + 1e-8)

    bad_channels = np.sum(np.abs(means) > 100) + np.sum(ranges > 500)

    quality_score = max(0, 100 - bad_channels * 10 - variance_stability * 20)

    return {
        "n_channels": n_channels,
        "n_samples": n_samples,
        "mean_mean": float(np.mean(means)),
        "mean_std": float(np.mean(stds)),
        "mean_range": float(np.mean(ranges)),
        "variance_stability": float(variance_stability),
        "bad_channels_estimate": int(bad_channels),
        "quality_score": float(quality_score),
        "is_good_quality": quality_score > 60,
    }


def segment_signal(
    signal: np.ndarray,
    window_size: int,
    overlap: int = 0,
) -> np.ndarray:
    """
    将信号分割成重叠窗口
    
    Args:
        signal: 输入信号，形状为 (n_channels, n_samples)
        window_size: 窗口大小（采样点数）
        overlap: 重叠大小（采样点数）
        
    Returns:
        分割后的信号，形状为 (n_windows, n_channels, window_size)
    """
    n_channels, n_samples = signal.shape
    step = window_size - overlap

    n_windows = (n_samples - window_size) // step + 1

    windows = np.zeros((n_windows, n_channels, window_size))

    for i in range(n_windows):
        start = i * step
        end = start + window_size
        windows[i] = signal[:, start:end]

    return windows

"""
BCI 推理引擎配置
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SignalConfig:
    """信号采集配置"""
    num_channels: int = 64
    sampling_rate: int = 250
    window_size_seconds: float = 2.0
    max_buffer_seconds: float = 10.0
    eeg_scale_factor: float = 1e-6  # 微伏到伏的转换系数


@dataclass
class CSPConfig:
    """CSP特征提取配置"""
    n_components: int = 6
    reg: float = 1e-6
    low_freq: float = 8.0
    high_freq: float = 30.0
    filter_order: int = 4


@dataclass
class ModelConfig:
    """深度学习模型配置"""
    model_type: str = "standard"  # 'standard' 或 'lightweight'
    n_classes: int = 2
    dropout_rate: float = 0.3
    device: str = "cpu"  # 'cpu' 或 'cuda'
    model_path: str = "saved_models/eeg_cnn.pth"
    csp_model_path: str = "saved_models/csp.npz"


@dataclass
class InferenceConfig:
    """推理配置"""
    confidence_threshold: float = 0.6
    decision_smoothing_window: int = 5
    min_inference_interval_ms: float = 100.0
    class_names: List[str] = field(default_factory=lambda: ["left_hand", "right_hand"])


@dataclass
class ExoskeletonCommandConfig:
    """外骨骼控制指令配置"""
    command_type: str = "motor_control"
    left_motor_id: int = 0
    right_motor_id: int = 1
    default_speed: float = 50.0
    command_timeout_ms: float = 500.0


@dataclass
class APIConfig:
    """API配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    workers: int = 1


@dataclass
class BCIConfig:
    """完整的BCI系统配置"""
    signal: SignalConfig = field(default_factory=SignalConfig)
    csp: CSPConfig = field(default_factory=CSPConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    exoskeleton: ExoskeletonCommandConfig = field(default_factory=ExoskeletonCommandConfig)
    api: APIConfig = field(default_factory=APIConfig)

    @property
    def window_size_samples(self) -> int:
        return int(self.signal.window_size_seconds * self.signal.sampling_rate)

    @property
    def n_csp_channels(self) -> int:
        return self.csp.n_components * 2

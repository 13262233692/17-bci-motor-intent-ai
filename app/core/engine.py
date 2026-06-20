import time
import threading
import numpy as np
import torch
from typing import Optional, Dict, Any, Tuple, List
from collections import deque

from app.signal.buffer import LatchedSignalBuffer
from app.signal.csp import CSP, BandPassFilter, normalize_signal
from app.models.eeg_cnn import build_model
from app.core.config import BCIConfig


class MotorImageryInferenceEngine:
    """
    运动想象推理引擎
    
    整合信号缓冲、CSP特征提取和1D-CNN分类器，
    提供端到端的EEG信号到运动意图的推理。
    
    核心特性：
    1. 严格的锁存缓冲机制 - 防止数据污染
    2. 实时流式推理 - 低延迟处理
    3. 决策平滑 - 提高稳定性
    4. 置信度阈值 - 避免误触发
    """

    def __init__(self, config: Optional[BCIConfig] = None):
        self.config = config or BCIConfig()

        self._buffer = LatchedSignalBuffer(
            num_channels=self.config.signal.num_channels,
            sampling_rate=self.config.signal.sampling_rate,
            window_size_seconds=self.config.signal.window_size_seconds,
            max_buffer_seconds=self.config.signal.max_buffer_seconds,
        )

        self._bandpass_filter: Optional[BandPassFilter] = None
        self._csp: Optional[CSP] = None
        self._model: Optional[torch.nn.Module] = None

        self._decision_history: deque = deque(
            maxlen=self.config.inference.decision_smoothing_window
        )
        self._last_inference_time: float = 0.0
        self._last_command: Optional[Dict[str, Any]] = None

        self._inference_lock = threading.Lock()
        self._initialized = False

    def initialize(self) -> None:
        """初始化推理引擎的所有组件"""
        self._bandpass_filter = BandPassFilter(
            low_freq=self.config.csp.low_freq,
            high_freq=self.config.csp.high_freq,
            sampling_rate=self.config.signal.sampling_rate,
            order=self.config.csp.filter_order,
        )

        try:
            self._csp = CSP.load(self.config.model.csp_model_path)
        except (FileNotFoundError, ValueError):
            self._csp = CSP(
                n_components=self.config.csp.n_components,
                reg=self.config.csp.reg,
            )

        self._model = build_model(
            model_type=self.config.model.model_type,
            n_channels=self.config.n_csp_channels,
            n_timepoints=self.config.window_size_samples,
            n_classes=self.config.model.n_classes,
            dropout_rate=self.config.model.dropout_rate,
        )

        try:
            state_dict = torch.load(
                self.config.model.model_path,
                map_location=self.config.model.device,
                weights_only=True,
            )
            self._model.load_state_dict(state_dict)
        except (FileNotFoundError, RuntimeError):
            pass

        self._model.eval()
        self._model.to(self.config.model.device)

        self._initialized = True

    def feed_signal(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None) -> None:
        """
        接收EEG信号数据并写入缓冲区
        
        Args:
            data: EEG数据，形状为 (num_channels, n_samples)，单位为微伏
            timestamps: 可选，时间戳数组
        """
        if not self._initialized:
            raise RuntimeError("引擎尚未初始化，请先调用 initialize()")

        data_scaled = data.astype(np.float64) * self.config.signal.eeg_scale_factor

        self._buffer.push(data_scaled, timestamps)

    def is_ready(self) -> bool:
        """检查是否有足够的数据进行推理"""
        return self._buffer.is_ready()

    def _preprocess(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        信号预处理流水线
        
        步骤：
        1. 带通滤波 (Mu/Beta频段)
        2. 归一化
        3. CSP空间滤波
        
        Args:
            eeg_data: 原始EEG数据，形状 (n_channels, n_samples)
            
        Returns:
            预处理后的数据，形状 (n_csp_channels, n_samples)
        """
        filtered = self._bandpass_filter.apply(eeg_data, axis=-1)

        normalized = normalize_signal(filtered, method="zscore")

        if self._csp.filters_ is not None:
            csp_filtered = self._csp.apply_filter(normalized)
        else:
            csp_filtered = normalized[: self.config.n_csp_channels, :]

        return csp_filtered

    def _inference(self, processed_data: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        执行模型推理
        
        Args:
            processed_data: 预处理后的数据，形状 (n_csp_channels, n_samples)
            
        Returns:
            (predicted_class, confidence, probabilities)
        """
        input_tensor = torch.from_numpy(processed_data).float().unsqueeze(0)
        input_tensor = input_tensor.to(self.config.model.device)

        with torch.no_grad():
            logits = self._model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probabilities, dim=1)

        pred_class = self.config.inference.class_names[pred_idx.item()]
        conf = confidence.item()
        probs = probabilities.squeeze(0).cpu().numpy()

        return pred_class, conf, probs

    def _smooth_decision(self, pred_class: str, confidence: float) -> Tuple[str, float]:
        """
        决策平滑 - 使用滑动窗口投票
        
        Args:
            pred_class: 当前预测类别
            confidence: 当前置信度
            
        Returns:
            (smoothed_class, smoothed_confidence)
        """
        self._decision_history.append((pred_class, confidence))

        if len(self._decision_history) < self._decision_history.maxlen // 2:
            return pred_class, confidence

        class_votes: Dict[str, List[float]] = {}
        for cls, conf in self._decision_history:
            if cls not in class_votes:
                class_votes[cls] = []
            class_votes[cls].append(conf)

        best_class = None
        best_score = -1

        for cls, confs in class_votes.items():
            avg_conf = np.mean(confs)
            vote_count = len(confs)
            score = avg_conf * vote_count

            if score > best_score:
                best_score = score
                best_class = cls

        smoothed_conf = np.mean(class_votes.get(best_class, [confidence]))

        return best_class, smoothed_conf

    def _generate_command(
        self,
        predicted_class: str,
        confidence: float,
        probabilities: np.ndarray,
    ) -> Dict[str, Any]:
        """
        生成外骨骼控制指令
        
        Args:
            predicted_class: 预测的运动想象类别
            confidence: 置信度
            probabilities: 各类别概率
            
        Returns:
            控制指令字典
        """
        if confidence < self.config.inference.confidence_threshold:
            command = {
                "type": self.config.exoskeleton.command_type,
                "action": "hold",
                "confidence": float(confidence),
                "probabilities": {
                    cls: float(prob)
                    for cls, prob in zip(
                        self.config.inference.class_names, probabilities
                    )
                },
                "timestamp": time.time(),
                "valid": False,
                "reason": "low_confidence",
            }
        else:
            if predicted_class == "left_hand":
                action = "move_left"
                motor_id = self.config.exoskeleton.left_motor_id
            elif predicted_class == "right_hand":
                action = "move_right"
                motor_id = self.config.exoskeleton.right_motor_id
            else:
                action = "hold"
                motor_id = -1

            command = {
                "type": self.config.exoskeleton.command_type,
                "action": action,
                "motor_id": motor_id,
                "speed": self.config.exoskeleton.default_speed,
                "confidence": float(confidence),
                "probabilities": {
                    cls: float(prob)
                    for cls, prob in zip(
                        self.config.inference.class_names, probabilities
                    )
                },
                "timestamp": time.time(),
                "valid": True,
                "timeout_ms": self.config.exoskeleton.command_timeout_ms,
            }

        return command

    def infer(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """
        执行一次推理
        
        Args:
            force: 是否强制推理（忽略最小间隔限制）
            
        Returns:
            控制指令字典，如果数据不足则返回 None
        """
        if not self._initialized:
            raise RuntimeError("引擎尚未初始化，请先调用 initialize()")

        with self._inference_lock:
            current_time = time.time()
            min_interval = self.config.inference.min_inference_interval_ms / 1000.0

            if not force and (current_time - self._last_inference_time) < min_interval:
                return self._last_command

            window_data = self._buffer.await_window()
            if window_data is None:
                return None

            processed = self._preprocess(window_data)

            pred_class, confidence, probs = self._inference(processed)

            smoothed_class, smoothed_conf = self._smooth_decision(
                pred_class, confidence
            )

            command = self._generate_command(smoothed_class, smoothed_conf, probs)

            self._last_inference_time = current_time
            self._last_command = command

            return command

    def infer_latched(self) -> Optional[Dict[str, Any]]:
        """
        使用锁存窗口进行推理
        
        Returns:
            控制指令字典
        """
        if not self._initialized:
            raise RuntimeError("引擎尚未初始化，请先调用 initialize()")

        with self._inference_lock:
            if not self._buffer.latch_window():
                return None

            window_data = self._buffer.get_window_and_clear()
            if window_data is None:
                return None

            processed = self._preprocess(window_data)
            pred_class, confidence, probs = self._inference(processed)
            smoothed_class, smoothed_conf = self._smooth_decision(pred_class, confidence)
            command = self._generate_command(smoothed_class, smoothed_conf, probs)

            self._last_inference_time = time.time()
            self._last_command = command

            return command

    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态信息"""
        return {
            "initialized": self._initialized,
            "buffer": self._buffer.get_stats(),
            "last_inference_time": self._last_inference_time,
            "decision_history_length": len(self._decision_history),
            "last_command": self._last_command,
        }

    def reset(self) -> None:
        """重置引擎状态"""
        self._buffer.clear()
        self._decision_history.clear()
        self._last_inference_time = 0.0
        self._last_command = None

    @property
    def buffer(self) -> LatchedSignalBuffer:
        """获取信号缓冲区"""
        return self._buffer

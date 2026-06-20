import time
import threading
import numpy as np
import torch
from typing import Optional, Dict, Any, Tuple, List
from collections import deque

from app.signal.buffer import LatchedSignalBuffer
from app.signal.csp import (
    CSP,
    BandPassFilter,
    HighPassFilter,
    normalize_signal,
    robust_preprocessing_pipeline,
    check_numerical_stability,
    detect_baseline_drift,
)
from app.models.eeg_cnn import build_model
from app.core.config import BCIConfig
from app.xai.explainer import BCIExplainer, XAIOutputFormatter


class MotorImageryInferenceEngine:
    """
    运动想象推理引擎 - 抗基线漂移增强版
    
    整合信号缓冲、鲁棒预处理、CSP特征提取和1D-CNN分类器，
    提供端到端的EEG信号到运动意图的推理。
    
    核心改进（针对基线漂移问题）：
    1. 前置高通滤波 - 1Hz截止频率消除基线漂移
    2. InstanceNorm替代BatchNorm - 对分布偏移更鲁棒
    3. 鲁棒预处理流水线 - 漂移检测、NaN修复、极值截断
    4. 数值稳定性保护 - 多层级防崩溃机制
    
    原有特性：
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

        self._highpass_filter: Optional[HighPassFilter] = None
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

        self._drift_count: int = 0
        self._total_inferences: int = 0
        self._nan_occurrences: int = 0
        self._preprocessing_stats: Dict[str, Any] = {}

        self._explainer: Optional[BCIExplainer] = None
        self._last_xai_result: Optional[Dict[str, Any]] = None
        self._xai_enabled: bool = True

    def initialize(self) -> None:
        """初始化推理引擎的所有组件"""
        if self.config.highpass.enabled:
            self._highpass_filter = HighPassFilter(
                cutoff_freq=self.config.highpass.cutoff_freq,
                sampling_rate=self.config.signal.sampling_rate,
                order=self.config.highpass.order,
            )

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

        if self._xai_enabled:
            self._explainer = BCIExplainer(
                model=self._model,
                csp_filters=self._csp.filters_,
                csp_patterns=self._csp.patterns_ if hasattr(self._csp, 'patterns_') else None,
                device=self.config.model.device,
                num_channels=self.config.signal.num_channels,
            )

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

        if self.config.numerical_stability.nan_detection:
            data_scaled = np.nan_to_num(
                data_scaled,
                nan=0.0,
                posinf=1e-3,
                neginf=-1e-3,
            )

        self._buffer.push(data_scaled, timestamps)

    def is_ready(self) -> bool:
        """检查是否有足够的数据进行推理"""
        return self._buffer.is_ready()

    def _preprocess(self, eeg_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        鲁棒的信号预处理流水线
        
        处理步骤（增强版）：
        1. 数值稳定性检查
        2. 基线漂移检测
        3. 高通滤波（消除基线漂移）
        4. 带通滤波 (Mu/Beta频段)
        5. 归一化
        6. CSP空间滤波
        
        Args:
            eeg_data: 原始EEG数据，形状 (n_channels, n_samples)，单位为伏
            
        Returns:
            (预处理后的数据, 处理状态字典)
        """
        preprocess_status = {}

        if self.config.numerical_stability.enable_robust_preprocessing:
            processed, status = robust_preprocessing_pipeline(
                eeg_data,
                highpass_filter=self._highpass_filter,
                bandpass_filter=self._bandpass_filter,
                sampling_rate=self.config.signal.sampling_rate,
            )
            preprocess_status = status

            if status["drift_detected"]:
                self._drift_count += 1
        else:
            filtered = self._bandpass_filter.apply(eeg_data, axis=-1)
            processed = normalize_signal(filtered, method="zscore")
            preprocess_status = {
                "steps_completed": ["bandpass_filtering", "normalization"],
                "drift_detected": False,
                "drift_magnitude": 0.0,
            }

        if self._csp.filters_ is not None:
            csp_filtered = self._csp.apply_filter(processed)
        else:
            csp_filtered = processed[: self.config.n_csp_channels, :]

        preprocess_status["csp_applied"] = self._csp.filters_ is not None
        preprocess_status["output_shape"] = csp_filtered.shape

        output_stability = check_numerical_stability(csp_filtered)
        preprocess_status["output_stability"] = output_stability

        if not output_stability["is_stable"]:
            self._nan_occurrences += 1
            csp_filtered = np.nan_to_num(
                csp_filtered,
                nan=0.0,
                posinf=5.0,
                neginf=-5.0,
            )
            csp_filtered = np.clip(csp_filtered, -5.0, 5.0)

        return csp_filtered, preprocess_status

    def _inference(
        self,
        processed_data: np.ndarray,
    ) -> Tuple[str, float, np.ndarray, Dict[str, Any]]:
        """
        执行模型推理（增强版，带数值稳定性保护）
        
        Args:
            processed_data: 预处理后的数据，形状 (n_csp_channels, n_samples)
            
        Returns:
            (predicted_class, confidence, probabilities, inference_stats)
        """
        inference_stats = {
            "has_nan_input": bool(np.any(np.isnan(processed_data))),
            "has_inf_input": bool(np.any(np.isinf(processed_data))),
            "input_max": float(np.max(np.abs(processed_data))),
            "input_mean": float(np.mean(processed_data)),
        }

        if inference_stats["has_nan_input"] or inference_stats["has_inf_input"]:
            processed_data = np.nan_to_num(
                processed_data,
                nan=0.0,
                posinf=5.0,
                neginf=-5.0,
            )
            processed_data = np.clip(processed_data, -5.0, 5.0)
            self._nan_occurrences += 1
            inference_stats["nan_fixed"] = True

        input_tensor = torch.from_numpy(processed_data).float().unsqueeze(0)
        input_tensor = input_tensor.to(self.config.model.device)

        with torch.no_grad():
            probabilities = self._model.predict_proba(input_tensor)
            confidence, pred_idx = torch.max(probabilities, dim=1)

        probs_np = probabilities.squeeze(0).cpu().numpy()
        conf_np = confidence.item()
        pred_idx_np = pred_idx.item()

        inference_stats["output_max_prob"] = float(np.max(probs_np))
        inference_stats["output_min_prob"] = float(np.min(probs_np))
        inference_stats["has_nan_output"] = bool(np.any(np.isnan(probs_np)))

        if inference_stats["has_nan_output"]:
            self._nan_occurrences += 1
            probs_np = np.array([0.5, 0.5])
            conf_np = 0.5
            pred_idx_np = 0 if np.random.rand() > 0.5 else 1
            inference_stats["output_nan_fixed"] = True

        pred_class = self.config.inference.class_names[pred_idx_np]

        return pred_class, conf_np, probs_np, inference_stats

    def _smooth_decision(
        self,
        pred_class: str,
        confidence: float,
    ) -> Tuple[str, float]:
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
        preprocess_status: Optional[Dict[str, Any]] = None,
        inference_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        生成外骨骼控制指令
        
        Args:
            predicted_class: 预测的运动想象类别
            confidence: 置信度
            probabilities: 各类别概率
            preprocess_status: 预处理状态
            inference_stats: 推理统计
            
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

        if preprocess_status is not None:
            command["preprocessing"] = {
                "drift_detected": preprocess_status.get("drift_detected", False),
                "drift_magnitude": preprocess_status.get("drift_magnitude", 0.0),
                "steps_completed": preprocess_status.get("steps_completed", []),
            }

        if inference_stats is not None:
            command["inference_stats"] = {
                "has_nan": inference_stats.get("has_nan_input", False)
                           or inference_stats.get("has_nan_output", False),
                "input_max": inference_stats.get("input_max", 0.0),
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

            self._total_inferences += 1

            processed, preprocess_status = self._preprocess(window_data)
            self._preprocessing_stats = preprocess_status

            pred_class, confidence, probs, inference_stats = self._inference(processed)

            smoothed_class, smoothed_conf = self._smooth_decision(
                pred_class, confidence
            )

            command = self._generate_command(
                smoothed_class,
                smoothed_conf,
                probs,
                preprocess_status,
                inference_stats,
            )

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

            self._total_inferences += 1

            processed, preprocess_status = self._preprocess(window_data)
            self._preprocessing_stats = preprocess_status

            pred_class, confidence, probs, inference_stats = self._inference(processed)
            smoothed_class, smoothed_conf = self._smooth_decision(pred_class, confidence)
            command = self._generate_command(
                smoothed_class,
                smoothed_conf,
                probs,
                preprocess_status,
                inference_stats,
            )

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
            "drift_detected_count": self._drift_count,
            "total_inferences": self._total_inferences,
            "nan_occurrences": self._nan_occurrences,
            "highpass_enabled": self.config.highpass.enabled,
            "robust_preprocessing": self.config.numerical_stability.enable_robust_preprocessing,
            "last_preprocessing": self._preprocessing_stats,
            "xai_enabled": self.xai_enabled,
            "has_last_xai": self._last_xai_result is not None,
        }

    def reset(self) -> None:
        """重置引擎状态"""
        self._buffer.clear()
        self._decision_history.clear()
        self._last_inference_time = 0.0
        self._last_command = None
        self._drift_count = 0
        self._total_inferences = 0
        self._nan_occurrences = 0
        self._preprocessing_stats = {}
        self._last_xai_result = None

    def explain(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """
        执行推理并返回完整的可解释性分析结果
        
        Args:
            force: 是否强制执行，忽略最小间隔限制
            
        Returns:
            包含控制指令和XAI分析结果的字典，如果数据不足返回None
        """
        if not self._initialized:
            raise RuntimeError("引擎尚未初始化，请先调用 initialize()")

        with self._inference_lock:
            current_time = time.time()
            min_interval = self.config.inference.min_inference_interval_ms / 1000.0

            if not force and (current_time - self._last_inference_time) < min_interval:
                if self._last_xai_result is not None:
                    return self._last_xai_result
                return None

            window_data = self._buffer.await_window()
            if window_data is None:
                return None

            self._total_inferences += 1

            processed, preprocess_status = self._preprocess(window_data)
            self._preprocessing_stats = preprocess_status

            pred_class, confidence, probs, inference_stats = self._inference(processed)

            smoothed_class, smoothed_conf = self._smooth_decision(
                pred_class, confidence
            )

            command = self._generate_command(
                smoothed_class,
                smoothed_conf,
                probs,
                preprocess_status,
                inference_stats,
            )

            xai_result = None
            if self._explainer is not None:
                try:
                    xai_start = time.time()
                    xai_analysis = self._explainer.explain(
                        csp_processed_input=processed,
                        target_class=None,
                        class_names=self.config.inference.class_names,
                    )
                    xai_time_ms = (time.time() - xai_start) * 1000

                    xai_result = XAIOutputFormatter.to_api_response(xai_analysis)
                    xai_result["xai_computation_time_ms"] = xai_time_ms
                except Exception as e:
                    xai_result = {
                        "error": str(e),
                        "has_spatial_heatmap": False,
                        "prediction": command,
                    }

            result = {
                "command": command,
                "xai": xai_result,
                "timestamp": time.time(),
            }

            self._last_inference_time = current_time
            self._last_command = command
            self._last_xai_result = result

            return result

    def explain_latched(self) -> Optional[Dict[str, Any]]:
        """
        使用锁存窗口执行推理并返回XAI分析结果
        
        Returns:
            包含控制指令和XAI分析结果的字典
        """
        if not self._initialized:
            raise RuntimeError("引擎尚未初始化，请先调用 initialize()")

        with self._inference_lock:
            if not self._buffer.latch_window():
                return None

            window_data = self._buffer.get_window_and_clear()
            if window_data is None:
                return None

            self._total_inferences += 1

            processed, preprocess_status = self._preprocess(window_data)
            self._preprocessing_stats = preprocess_status

            pred_class, confidence, probs, inference_stats = self._inference(processed)
            smoothed_class, smoothed_conf = self._smooth_decision(pred_class, confidence)
            command = self._generate_command(
                smoothed_class,
                smoothed_conf,
                probs,
                preprocess_status,
                inference_stats,
            )

            xai_result = None
            if self._explainer is not None:
                try:
                    xai_start = time.time()
                    xai_analysis = self._explainer.explain(
                        csp_processed_input=processed,
                        target_class=None,
                        class_names=self.config.inference.class_names,
                    )
                    xai_time_ms = (time.time() - xai_start) * 1000

                    xai_result = XAIOutputFormatter.to_api_response(xai_analysis)
                    xai_result["xai_computation_time_ms"] = xai_time_ms
                except Exception as e:
                    xai_result = {
                        "error": str(e),
                        "has_spatial_heatmap": False,
                        "prediction": command,
                    }

            result = {
                "command": command,
                "xai": xai_result,
                "timestamp": time.time(),
            }

            self._last_inference_time = time.time()
            self._last_command = command
            self._last_xai_result = result

            return result

    def get_last_xai(self) -> Optional[Dict[str, Any]]:
        """获取最近一次XAI分析结果"""
        return self._last_xai_result

    @property
    def xai_enabled(self) -> bool:
        """XAI功能是否启用"""
        return self._xai_enabled and self._explainer is not None

    @property
    def buffer(self) -> LatchedSignalBuffer:
        """获取信号缓冲区"""
        return self._buffer

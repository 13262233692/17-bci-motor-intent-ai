"""
BCI 可解释性成像 (XAI) 主模块

整合Grad-CAM类激活映射、CSP反向投影、10-20电极坐标系统，
提供端到端的运动想象脑电可解释性分析。

输出前端可直接渲染的空间热力图数据。
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple

from app.xai.gradcam import EEGXAnalyzer
from app.xai.electrode_montage import (
    get_standard_channel_order,
    generate_heatmap_data,
    compute_region_contributions,
    get_motor_cortex_activity,
)


class BCIExplainer:
    """
    BCI运动想象可解释性分析器
    
    完整的XAI流水线：
    1. Grad-CAM时间热力图
    2. 梯度×输入的CSP通道权重
    3. CSP→物理通道反向投影
    4. 10-20电极坐标映射
    5. 脑区贡献分析
    6. 运动皮层偏侧化分析
    """

    def __init__(
        self,
        model: nn.Module,
        csp_filters: Optional[np.ndarray] = None,
        csp_patterns: Optional[np.ndarray] = None,
        target_layer: Optional[nn.Module] = None,
        device: str = "cpu",
        channel_names: Optional[List[str]] = None,
        num_channels: int = 64,
    ):
        self.device = device
        self.num_channels = num_channels

        if channel_names is None:
            self.channel_names = get_standard_channel_order(num_channels)
        else:
            self.channel_names = channel_names

        self._analyzer = EEGXAnalyzer(
            model=model,
            csp_filters=csp_filters,
            csp_patterns=csp_patterns,
            target_layer=target_layer,
        )

        self._enabled = csp_filters is not None

    def explain(
        self,
        csp_processed_input: np.ndarray,
        target_class: Optional[int] = None,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        执行完整的XAI分析
        
        Args:
            csp_processed_input: CSP处理后的EEG数据，形状 (n_csp_channels, n_timepoints)
            target_class: 目标类别索引，None表示使用预测类别
            class_names: 类别名称列表，如 ["left_hand", "right_hand"]
            
        Returns:
            完整的可解释性分析结果，包含：
            - prediction: 预测结果
            - grad_cam: Grad-CAM时间热力图
            - channel_importance: 物理通道重要性
            - heatmap: 前端可渲染的热力图数据
            - brain_regions: 各脑区贡献
            - motor_cortex: 运动皮层偏侧化分析
        """
        analysis = self._analyzer.analyze(
            processed_input=csp_processed_input,
            target_class=target_class,
            device=self.device,
        )

        result = {
            "prediction": {
                "predicted_class_idx": analysis["predicted_class"],
                "predicted_class_name": (
                    class_names[analysis["predicted_class"]]
                    if class_names is not None
                    else str(analysis["predicted_class"])
                ),
                "class_probabilities": analysis["class_probabilities"],
                "confidence": max(analysis["class_probabilities"]),
            },
            "grad_cam": {
                "time_heatmap": analysis["grad_cam_heatmap"],
                "csp_channel_importance": analysis.get("csp_channel_importance", []),
                "csp_channel_weights": analysis.get("csp_channel_weights", []),
            },
        }

        if self._enabled and "physical_channel_importance" in analysis:
            channel_imp = np.array(analysis["physical_channel_importance"])

            result["channel_importance"] = {
                "values": channel_imp.tolist(),
                "channel_names": self.channel_names[: len(channel_imp)],
                "top_channels": analysis.get("top_channels", []),
                "top_channel_names": [
                    self.channel_names[i]
                    for i in analysis.get("top_channels", [])
                    if i < len(self.channel_names)
                ],
                "top_importance_values": analysis.get("top_channel_importance", []),
            }

            heatmap = generate_heatmap_data(
                channel_importance=channel_imp,
                channel_names=self.channel_names,
                num_channels=self.num_channels,
            )
            result["heatmap"] = heatmap

            result["brain_regions"] = compute_region_contributions(
                channel_importance=channel_imp,
                channel_names=self.channel_names,
            )

            result["motor_cortex"] = get_motor_cortex_activity(
                channel_importance=channel_imp,
                channel_names=self.channel_names,
            )

        else:
            result["channel_importance"] = None
            result["heatmap"] = None
            result["brain_regions"] = None
            result["motor_cortex"] = None
            result["note"] = "CSP滤波器未加载，仅提供Grad-CAM时间分析"

        return result

    def explain_from_raw(
        self,
        raw_eeg: np.ndarray,
        csp_filter_func,
        preprocess_func,
        target_class: Optional[int] = None,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        从原始EEG数据开始解释
        
        Args:
            raw_eeg: 原始EEG数据，形状 (n_channels, n_timepoints)
            csp_filter_func: CSP滤波函数
            preprocess_func: 预处理函数
            target_class: 目标类别
            class_names: 类别名称
            
        Returns:
            XAI分析结果
        """
        preprocessed = preprocess_func(raw_eeg)
        csp_processed = csp_filter_func(preprocessed)
        return self.explain(csp_processed, target_class, class_names)


class XAIOutputFormatter:
    """
    XAI结果格式化器
    
    将XAI分析结果格式化为适合API输出或前端渲染的格式。
    """

    @staticmethod
    def to_api_response(xai_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换为API响应格式
        
        Args:
            xai_result: explain()返回的结果
            
        Returns:
            API响应字典
        """
        response = {
            "prediction": xai_result["prediction"],
            "grad_cam": xai_result["grad_cam"],
            "has_spatial_heatmap": xai_result["heatmap"] is not None,
        }

        if xai_result["heatmap"] is not None:
            response["heatmap"] = {
                "electrodes": [
                    {
                        "channel": e["channel"],
                        "x": e["x"],
                        "y": e["y"],
                        "importance": e["importance"],
                        "normalized_importance": e["normalized_importance"],
                        "region": e["region"],
                        "hemisphere": e["hemisphere"],
                    }
                    for e in xai_result["heatmap"]["electrodes"]
                ],
                "top_electrodes": xai_result["heatmap"]["top_electrodes"],
                "region_importance": xai_result["heatmap"]["normalized_region_importance"],
                "motor_candidates": xai_result["heatmap"]["motor_candidate_channels"],
            }

            response["brain_region_summary"] = {
                region: {
                    "importance": v.get("normalized_importance", v.get("mean_importance", 0)),
                    "percentage": v.get("percentage", 0),
                }
                for region, v in xai_result["brain_regions"].items()
            }

            motor = xai_result["motor_cortex"]
            response["motor_cortex_analysis"] = {
                "left_hand_area": motor.get("left_hand", {}),
                "right_hand_area": motor.get("right_hand", {}),
                "laterality_index": motor.get("laterality_index", 0),
                "interpretation": XAIOutputFormatter._interpret_laterality(
                    motor.get("laterality_index", 0),
                    xai_result["prediction"]["predicted_class_name"],
                ),
            }

        return response

    @staticmethod
    def _interpret_laterality(laterality_index: float, predicted_class: str) -> str:
        """
        解释偏侧化指数
        
        Args:
            laterality_index: 偏侧化指数
            predicted_class: 预测类别
            
        Returns:
            文字解释
        """
        if abs(laterality_index) < 0.1:
            return "双侧运动皮层激活均衡"

        is_right_lateralized = laterality_index > 0.2
        is_left_lateralized = laterality_index < -0.2

        expected_left_activation = "right_hand" in predicted_class.lower()
        expected_right_activation = "left_hand" in predicted_class.lower()

        if expected_left_activation and is_left_lateralized:
            return "激活模式与预测一致：右手运动想象伴随左侧运动皮层(C3)激活"
        elif expected_right_activation and is_right_lateralized:
            return "激活模式与预测一致：左手运动想象伴随右侧运动皮层(C4)激活"
        elif expected_left_activation and is_right_lateralized:
            return "注意：激活模式偏侧化方向与预期相反，可能存在双侧代偿或伪迹干扰"
        elif expected_right_activation and is_left_lateralized:
            return "注意：激活模式偏侧化方向与预期相反，可能存在双侧代偿或伪迹干扰"
        else:
            if is_left_lateralized:
                return "左侧运动皮层(C3/C1/CP3)主导激活"
            else:
                return "右侧运动皮层(C4/C2/CP4)主导激活"

"""
Grad-CAM (Gradient-weighted Class Activation Mapping) 实现

适用于1D-CNN EEG运动想象分类模型的可解释性分析。
通过反向传播计算梯度，确定哪些输入通道对分类决策贡献最大。
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict, Any


class GradCAM1D:
    """
    适用于1D卷积神经网络的Grad-CAM实现
    
    原理：
    1. 前向传播获取目标卷积层的激活图 A
    2. 反向传播获取目标类别对激活图的梯度 dY/dA
    3. 对梯度进行全局平均池化得到通道权重 α_k = 1/Z * Σ_i dY/dA_k,i
    4. Grad-CAM = ReLU(Σ_k α_k * A_k)
    
    对于EEG数据，Grad-CAM可以揭示：
    - 哪些CSP滤波通道对决策最重要
    - 哪些时间窗口包含最具判别性的信息
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        初始化Grad-CAM
        
        Args:
            model: 训练好的1D-CNN模型
            target_layer: 目标卷积层（通常是最后一个卷积层）
        """
        self.model = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        """注册前向和反向传播钩子"""

        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        handle_fwd = self.target_layer.register_forward_hook(forward_hook)
        handle_bwd = self.target_layer.register_full_backward_hook(backward_hook)

        self._handles = [handle_fwd, handle_bwd]

    def remove_hooks(self) -> None:
        """移除所有钩子"""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _compute_cam(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
    ) -> np.ndarray:
        """
        计算Grad-CAM热力图
        
        Args:
            activations: 卷积层激活图，形状 (batch, channels, length)
            gradients: 梯度张量，形状 (batch, channels, length)
            
        Returns:
            Grad-CAM热力图，形状 (length,)
        """
        grad_mean = torch.mean(gradients, dim=(0, 2))

        cam = torch.zeros(activations.shape[2], dtype=activations.dtype)
        for i, w in enumerate(grad_mean):
            cam += w * activations[0, i, :]

        cam = torch.relu(cam)

        cam_np = cam.detach().cpu().numpy()

        cam_min = cam_np.min()
        cam_max = cam_np.max()
        if cam_max - cam_min > 1e-10:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)

        return cam_np

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        生成Grad-CAM热力图
        
        Args:
            input_tensor: 输入张量，形状 (1, n_channels, n_timepoints)
            target_class: 目标类别索引，如果为None则使用预测类别
            
        Returns:
            (cam_heatmap, predicted_class, class_probabilities)
            - cam_heatmap: 时间维度的Grad-CAM热力图，形状 (n_timepoints,)
            - predicted_class: 预测的类别索引
            - class_probabilities: 各类别概率
        """
        self.model.eval()

        self.model.zero_grad()

        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)

        if target_class is None:
            target_class = torch.argmax(probs, dim=1).item()

        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)

        if self._activations is None or self._gradients is None:
            raise RuntimeError("无法获取激活或梯度，请检查目标层是否正确")

        cam = self._compute_cam(self._activations, self._gradients)

        probs_np = probs.detach().cpu().numpy().squeeze()

        return cam, target_class, probs_np

    def __del__(self):
        self.remove_hooks()


class GradCAMChannelWeighter:
    """
    将Grad-CAM时间热力图转换为通道权重
    
    对于EEG 1D-CNN，Grad-CAM给出的是时间维度的重要性。
    为了获得每个输入通道的重要性，我们使用梯度×输入的方法。
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def compute_channel_weights(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        """
        计算每个输入通道的重要性权重
        
        使用梯度×输入（Gradient × Input）的方法：
        对于每个通道，计算 dY/dX_c * X_c，然后在时间维度上求和，
        得到该通道对分类决策的贡献。
        
        Args:
            input_tensor: 输入张量，形状 (1, n_channels, n_timepoints)
            target_class: 目标类别索引
            
        Returns:
            (channel_weights, channel_importance, predicted_class, probabilities)
            - channel_weights: 每个通道的权重，形状 (n_channels,)
            - channel_importance: 归一化的重要性分数，形状 (n_channels,)
            - predicted_class: 预测类别
            - probabilities: 类别概率
        """
        self.model.eval()

        input_tensor.requires_grad_(True)
        self.model.zero_grad()

        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)

        if target_class is None:
            target_class = torch.argmax(probs, dim=1).item()

        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)

        gradients = input_tensor.grad
        input_values = input_tensor.detach()

        grad_x_input = gradients * input_values

        channel_weights = torch.sum(torch.abs(grad_x_input), dim=(0, 2)).detach().cpu().numpy()

        max_weight = channel_weights.max()
        if max_weight > 1e-10:
            channel_importance = channel_weights / max_weight
        else:
            channel_importance = np.zeros_like(channel_weights)

        probs_np = probs.detach().cpu().numpy().squeeze()

        return channel_weights, channel_importance, target_class, probs_np


class CSPProjector:
    """
    将CSP空间权重映射回原始EEG物理通道
    
    CSP将64通道EEG投影到12个空间滤波通道。
    为了将CSP通道的权重解释回原始物理通道，
    我们使用CSP滤波器（空间模式）进行反向投影。
    """

    def __init__(self, csp_filters: np.ndarray, csp_patterns: Optional[np.ndarray] = None):
        """
        初始化CSP投影器
        
        Args:
            csp_filters: CSP空间滤波器，形状 (n_csp_channels, n_original_channels)
            csp_patterns: CSP空间模式，形状 (n_original_channels, n_csp_channels)
                          或 (n_csp_channels, n_original_channels)
        """
        self.csp_filters = csp_filters
        self.n_csp_channels, self.n_original_channels = csp_filters.shape

        if csp_patterns is not None:
            if csp_patterns.shape == (self.n_csp_channels, self.n_original_channels):
                self.csp_patterns = csp_patterns.T
            else:
                self.csp_patterns = csp_patterns
        else:
            self.csp_patterns = np.linalg.pinv(csp_filters)

    def project_csp_weights_to_channels(
        self,
        csp_channel_weights: np.ndarray,
    ) -> np.ndarray:
        """
        将CSP通道的权重投影回原始EEG物理通道
        
        使用CSP空间模式（patterns）进行反向投影：
        channel_importance = Σ_k |pattern_c,k| * csp_weight_k
        
        Args:
            csp_channel_weights: CSP通道权重，形状 (n_csp_channels,)
            
        Returns:
            原始物理通道的重要性，形状 (n_original_channels,)
        """
        weighted_patterns = np.abs(self.csp_patterns) * csp_channel_weights[np.newaxis, :]
        channel_importance = np.sum(weighted_patterns, axis=1)

        max_val = channel_importance.max()
        if max_val > 1e-10:
            channel_importance = channel_importance / max_val

        return channel_importance

    def project_activation_to_channels(
        self,
        csp_activation: np.ndarray,
    ) -> np.ndarray:
        """
        将CSP空间滤波后的激活图投影回原始EEG物理通道
        
        Args:
            csp_activation: CSP激活图，形状 (n_csp_channels, n_timepoints)
            
        Returns:
            原始通道激活图，形状 (n_original_channels, n_timepoints)
        """
        return self.csp_patterns @ csp_activation


class EEGXAnalyzer:
    """
    EEG可解释性分析器 - 整合Grad-CAM和通道权重计算
    
    提供端到端的XAI分析：
    1. CSP→原始通道的权重投影
    2. Grad-CAM时间热力图
    3. 物理通道重要性排序
    """

    def __init__(
        self,
        model: nn.Module,
        csp_filters: Optional[np.ndarray] = None,
        csp_patterns: Optional[np.ndarray] = None,
        target_layer: Optional[nn.Module] = None,
    ):
        self.model = model

        if target_layer is None:
            target_layer = self._find_last_conv_layer(model)

        self.gradcam = GradCAM1D(model, target_layer)
        self.channel_weighter = GradCAMChannelWeighter(model)

        self.csp_projector = None
        if csp_filters is not None:
            self.csp_projector = CSPProjector(csp_filters, csp_patterns)

    def _find_last_conv_layer(self, model: nn.Module) -> nn.Module:
        """查找模型中最后一个1D卷积层"""
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv1d):
                last_conv = module
        if last_conv is None:
            raise ValueError("模型中未找到Conv1d层")
        return last_conv

    def analyze(
        self,
        processed_input: np.ndarray,
        target_class: Optional[int] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        执行完整的XAI分析
        
        Args:
            processed_input: CSP处理后的输入，形状 (n_csp_channels, n_timepoints)
            target_class: 目标类别，None表示使用预测类别
            device: 计算设备
            
        Returns:
            包含完整XAI分析结果的字典
        """
        input_tensor = torch.from_numpy(processed_input).float().unsqueeze(0).to(device)

        cam_heatmap, pred_class, probs = self.gradcam.generate(input_tensor, target_class)

        input_tensor_for_weight = torch.from_numpy(processed_input).float().unsqueeze(0).to(device)
        csp_weights, csp_importance, _, _ = self.channel_weighter.compute_channel_weights(
            input_tensor_for_weight, target_class
        )

        result = {
            "predicted_class": pred_class,
            "class_probabilities": probs.tolist(),
            "grad_cam_heatmap": cam_heatmap.tolist(),
            "csp_channel_weights": csp_weights.tolist(),
            "csp_channel_importance": csp_importance.tolist(),
        }

        if self.csp_projector is not None:
            channel_importance = self.csp_projector.project_csp_weights_to_channels(
                csp_importance
            )
            result["physical_channel_importance"] = channel_importance.tolist()

            top_channels = np.argsort(channel_importance)[::-1][:10]
            result["top_channels"] = top_channels.tolist()
            result["top_channel_importance"] = channel_importance[top_channels].tolist()

        return result

    def __del__(self):
        if hasattr(self, 'gradcam'):
            self.gradcam.remove_hooks()

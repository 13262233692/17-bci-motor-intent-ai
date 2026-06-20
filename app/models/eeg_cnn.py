import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class InstanceNormBlock1D(nn.Module):
    """
    1D实例归一化块 - 对分布偏移更鲁棒
    
    与BatchNorm不同，InstanceNorm独立地对每个样本的每个通道进行归一化，
    不依赖于批次统计量。这使得它在推理时对输入分布的变化（如基线漂移）
    具有更好的鲁棒性。
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        self.instance_norm = nn.InstanceNorm1d(
            num_features=num_features,
            eps=eps,
            affine=affine,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.instance_norm(x)


class ConvBlock1D(nn.Module):
    """
    1D卷积块 - 包含卷积、实例归一化、激活和池化
    
    使用InstanceNorm替代BatchNorm，提高对分布偏移的鲁棒性
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        pool_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.inst_norm = InstanceNormBlock1D(
            num_features=out_channels,
            affine=True,
        )
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.inst_norm(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class LayerNormBlock(nn.Module):
    """
    层归一化块 - 用于全连接层
    
    LayerNorm对每个样本独立进行归一化，同样不依赖批次统计，
    适合全连接层的分布归一化。
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x)


class EEG1DCNN(nn.Module):
    """
    用于运动想象EEG分类的一维卷积神经网络
    
    关键改进：
    - 使用InstanceNorm替代BatchNorm，提高对基线漂移等分布偏移的鲁棒性
    - 使用LayerNorm对全连接层进行归一化
    - 添加数值稳定性保护，防止NaN和Inf
    - logit截断机制，避免softmax饱和
    
    输入形状: (batch_size, n_channels, n_timepoints)
    输出: 二分类概率 (batch_size, 2)
    """

    def __init__(
        self,
        n_channels: int = 12,
        n_timepoints: int = 500,
        n_classes: int = 2,
        dropout_rate: float = 0.3,
        logit_clip: float = 20.0,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.n_classes = n_classes
        self.logit_clip = logit_clip

        self.conv1 = ConvBlock1D(
            in_channels=n_channels,
            out_channels=32,
            kernel_size=11,
            stride=1,
            padding=5,
            pool_size=2,
            dropout=dropout_rate,
        )

        self.conv2 = ConvBlock1D(
            in_channels=32,
            out_channels=64,
            kernel_size=7,
            stride=1,
            padding=3,
            pool_size=2,
            dropout=dropout_rate,
        )

        self.conv3 = ConvBlock1D(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            stride=1,
            padding=2,
            pool_size=2,
            dropout=dropout_rate,
        )

        self.conv4 = ConvBlock1D(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            pool_size=2,
            dropout=dropout_rate,
        )

        dummy_input = torch.zeros(1, n_channels, n_timepoints)
        with torch.no_grad():
            dummy_output = self._forward_conv(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc_ln1 = LayerNormBlock(256)
        self.fc_dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(256, 64)
        self.fc_ln2 = LayerNormBlock(64)
        self.fc_dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(64, n_classes)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    def _ensure_stable(self, x: torch.Tensor) -> torch.Tensor:
        """
        确保张量数值稳定，替换NaN和Inf
        
        Args:
            x: 输入张量
            
        Returns:
            数值稳定的张量
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=self.logit_clip, neginf=-self.logit_clip)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_stable(x)

        x = self._forward_conv(x)
        x = self._ensure_stable(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc_ln1(x)
        x = F.elu(x)
        x = self.fc_dropout1(x)
        x = self._ensure_stable(x)

        x = self.fc2(x)
        x = self.fc_ln2(x)
        x = F.elu(x)
        x = self.fc_dropout2(x)
        x = self._ensure_stable(x)

        x = self.fc3(x)

        x = torch.clamp(x, min=-self.logit_clip, max=self.logit_clip)
        x = self._ensure_stable(x)

        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测类别概率
        
        Args:
            x: 输入张量
            
        Returns:
            概率分布
        """
        logits = self.forward(x)
        logits = self._ensure_stable(logits)
        logits = torch.clamp(logits, min=-self.logit_clip, max=self.logit_clip)
        
        logits_shifted = logits - torch.max(logits, dim=1, keepdim=True)[0]
        exp_logits = torch.exp(logits_shifted)
        proba = exp_logits / (torch.sum(exp_logits, dim=1, keepdim=True) + 1e-10)
        
        proba = torch.clamp(proba, min=1e-7, max=1.0 - 1e-7)
        return self._ensure_stable(proba)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测类别
        
        Args:
            x: 输入张量
            
        Returns:
            类别索引
        """
        proba = self.predict_proba(x)
        return torch.argmax(proba, dim=1)


class DepthwiseSeparableConv1D(nn.Module):
    """
    深度可分离1D卷积 - 减少参数数量
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()

        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EEGLightweightCNN(nn.Module):
    """
    轻量级EEG分类网络 - 使用深度可分离卷积和InstanceNorm
    
    适用于资源受限的边缘设备部署，同时保持对分布偏移的鲁棒性
    """

    def __init__(
        self,
        n_channels: int = 12,
        n_timepoints: int = 500,
        n_classes: int = 2,
        dropout_rate: float = 0.25,
        logit_clip: float = 20.0,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.n_classes = n_classes
        self.logit_clip = logit_clip

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=25, stride=1, padding=12),
            InstanceNormBlock1D(16, affine=True),
            nn.ELU(),
            nn.MaxPool1d(3),
            nn.Dropout(dropout_rate),
        )

        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv1D(16, 32, kernel_size=15, padding=7),
            InstanceNormBlock1D(32, affine=True),
            nn.ELU(),
            nn.MaxPool1d(3),
            nn.Dropout(dropout_rate),
        )

        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv1D(32, 64, kernel_size=9, padding=4),
            InstanceNormBlock1D(64, affine=True),
            nn.ELU(),
            nn.MaxPool1d(3),
            nn.Dropout(dropout_rate),
        )

        dummy_input = torch.zeros(1, n_channels, n_timepoints)
        with torch.no_grad():
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            flattened_size = x.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 128),
            LayerNormBlock(128),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, n_classes),
        )

    def _ensure_stable(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=self.logit_clip, neginf=-self.logit_clip)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_stable(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self._ensure_stable(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        x = torch.clamp(x, min=-self.logit_clip, max=self.logit_clip)
        x = self._ensure_stable(x)

        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        logits = self._ensure_stable(logits)
        logits = torch.clamp(logits, min=-self.logit_clip, max=self.logit_clip)
        
        logits_shifted = logits - torch.max(logits, dim=1, keepdim=True)[0]
        exp_logits = torch.exp(logits_shifted)
        proba = exp_logits / (torch.sum(exp_logits, dim=1, keepdim=True) + 1e-10)
        
        proba = torch.clamp(proba, min=1e-7, max=1.0 - 1e-7)
        return self._ensure_stable(proba)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        proba = self.predict_proba(x)
        return torch.argmax(proba, dim=1)


def count_parameters(model: nn.Module) -> int:
    """计算模型的可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    model_type: str = "standard",
    n_channels: int = 12,
    n_timepoints: int = 500,
    n_classes: int = 2,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """
    构建EEG分类模型
    
    Args:
        model_type: 模型类型 - 'standard' | 'lightweight'
        n_channels: 输入通道数
        n_timepoints: 时间点数
        n_classes: 类别数
        dropout_rate: dropout比率
        
    Returns:
        PyTorch模型
    """
    if model_type == "standard":
        return EEG1DCNN(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes,
            dropout_rate=dropout_rate,
        )
    elif model_type == "lightweight":
        return EEGLightweightCNN(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes,
            dropout_rate=dropout_rate,
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class EEGDataRequest(BaseModel):
    """EEG数据请求模型"""
    data: List[List[float]] = Field(
        ...,
        description="EEG数据矩阵，形状为 [num_channels, n_samples]，单位为微伏"
    )
    timestamps: Optional[List[float]] = Field(
        None,
        description="可选的时间戳数组，长度为 n_samples"
    )
    sample_id: Optional[str] = Field(None, description="样本ID，用于追踪")

    class Config:
        json_schema_extra = {
            "example": {
                "data": [[0.1, 0.2, 0.3] * 64,
                "timestamps": [0.0, 0.004, 0.008],
                "sample_id": "sample_001"
            }
        }


class InferenceResponse(BaseModel):
    """推理响应模型"""
    type: str = Field(..., description="指令类型")
    action: str = Field(..., description="动作指令")
    motor_id: Optional[int] = Field(None, description="目标电机ID")
    speed: Optional[float] = Field(None, description="电机速度")
    confidence: float = Field(..., description="预测置信度")
    probabilities: Dict[str, float] = Field(..., description="各类别概率")
    timestamp: float = Field(..., description="时间戳")
    valid: bool = Field(..., description="指令是否有效")
    reason: Optional[str] = Field(None, description="无效原因")
    timeout_ms: Optional[float] = Field(None, description="指令超时时间")
    inference_time_ms: Optional[float] = Field(None, description="推理耗时(毫秒)")


class StatusResponse(BaseModel):
    """状态响应模型"""
    status: str = Field(..., description="系统状态")
    initialized: bool = Field(..., description="是否已初始化")
    buffer: Dict[str, Any] = Field(..., description="缓冲区状态")
    model_loaded: bool = Field(..., description="模型是否已加载")
    uptime_seconds: float = Field(..., description="运行时间(秒)")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchEEGRequest(BaseModel):
    """批量EEG数据请求"""
    samples: List[EEGDataRequest] = Field(..., description="EEG数据样本列表")


class ModelInfoResponse(BaseModel):
    """模型信息响应"""
    model_type: str
    n_channels: int
    n_classes: int
    class_names: List[str]
    sampling_rate: int
    window_size_seconds: float


class ElectrodeHeatmap(BaseModel):
    """单电极热力图数据"""
    channel: str = Field(..., description="电极名称，如C3, C4")
    x: float = Field(..., description="10-20系统x坐标，范围[-1, 1]")
    y: float = Field(..., description="10-20系统y坐标，范围[-1, 1]")
    importance: float = Field(..., description="原始重要性权重")
    normalized_importance: float = Field(..., description="归一化重要性[0, 1]")
    region: Optional[str] = Field(None, description="所属脑区")
    hemisphere: Optional[str] = Field(None, description="大脑半球: left/right/midline")


class GradCAMResponse(BaseModel):
    """Grad-CAM时间热力图响应"""
    time_heatmap: List[float] = Field(..., description="时间维度Grad-CAM热力图")
    csp_channel_importance: List[float] = Field(..., description="CSP通道重要性")
    csp_channel_weights: List[float] = Field(..., description="CSP通道原始权重")


class PredictionInfo(BaseModel):
    """预测信息"""
    predicted_class_idx: int = Field(..., description="预测类别索引")
    predicted_class_name: str = Field(..., description="预测类别名称，如left_hand/right_hand")
    class_probabilities: List[float] = Field(..., description="各类别概率")
    confidence: float = Field(..., description="最大置信度")


class MotorCortexAnalysis(BaseModel):
    """运动皮层分析结果"""
    left_hand_area: Dict[str, Any] = Field(..., description="左手运动想象相关区域激活")
    right_hand_area: Dict[str, Any] = Field(..., description="右手运动想象相关区域激活")
    laterality_index: float = Field(..., description="偏侧化指数，正值表示右侧激活更强，负值表示左侧")
    interpretation: str = Field(..., description="偏侧化模式的文字解释")


class XAIHeatmapResponse(BaseModel):
    """完整的XAI可解释性热力图响应"""
    prediction: PredictionInfo = Field(..., description="预测结果信息")
    grad_cam: GradCAMResponse = Field(..., description="Grad-CAM时间热力图")
    has_spatial_heatmap: bool = Field(..., description="是否包含空间热力图")
    heatmap: Optional[Dict[str, Any]] = Field(
        None,
        description="10-20系统空间热力图数据，包含电极坐标和权重"
    )
    brain_region_summary: Optional[Dict[str, Any]] = Field(
        None,
        description="各脑区贡献汇总"
    )
    motor_cortex_analysis: Optional[MotorCortexAnalysis] = Field(
        None,
        description="运动皮层偏侧化分析"
    )
    xai_computation_time_ms: Optional[float] = Field(
        None,
        description="XAI计算耗时(毫秒)"
    )


class InferenceWithXAIResponse(BaseModel):
    """带XAI分析的完整推理响应"""
    command: InferenceResponse = Field(..., description="外骨骼控制指令")
    xai: Optional[XAIHeatmapResponse] = Field(
        None,
        description="可解释性分析结果，包含空间热力图"
    )
    timestamp: float = Field(..., description="时间戳")

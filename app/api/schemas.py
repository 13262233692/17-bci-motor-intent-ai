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

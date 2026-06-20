import time
import numpy as np
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any

from app.api.schemas import (
    EEGDataRequest,
    InferenceResponse,
    StatusResponse,
    HealthResponse,
    BatchEEGRequest,
    ModelInfoResponse,
    InferenceWithXAIResponse,
    XAIHeatmapResponse,
)
from app.core.engine import MotorImageryInferenceEngine
from app.core.config import BCIConfig

router = APIRouter(prefix="/api/v1", tags=["bci"])

_engine: MotorImageryInferenceEngine = None
_start_time: float = None
_config: BCIConfig = None


def get_engine() -> MotorImageryInferenceEngine:
    """获取推理引擎单例"""
    global _engine
    if _engine is None:
        raise RuntimeError("推理引擎未初始化")
    return _engine


def initialize_engine(config: BCIConfig = None) -> None:
    """初始化推理引擎"""
    global _engine, _start_time, _config
    _config = config or BCIConfig()
    _engine = MotorImageryInferenceEngine(_config)
    _engine.initialize()
    _start_time = time.time()


@router.post("/eeg/stream", response_model=InferenceResponse)
async def receive_eeg_stream(request: EEGDataRequest):
    """
    接收EEG数据流并执行推理
    
    接收来自64通道脑电帽的高频微伏级EEG信号，经过CSP特征增强和
    1D-CNN分类后，返回外骨骼控制指令。
    """
    try:
        engine = get_engine()

        data = np.array(request.data, dtype=np.float64)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[0] != engine.config.signal.num_channels:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"通道数不匹配: 期望 {engine.config.signal.num_channels}, 实际 {data.shape[0]}"
            )

        timestamps = np.array(request.timestamps) if request.timestamps else None

        engine.feed_signal(data, timestamps)

        start_time = time.time()
        command = engine.infer()
        inference_time = (time.time() - start_time) * 1000

        if command is None:
            return InferenceResponse(
                type="motor_control",
                action="hold",
                confidence=0.0,
                probabilities={
                    cls: 0.0 for cls in engine.config.inference.class_names
                },
                timestamp=time.time(),
                valid=False,
                reason="insufficient_data",
                inference_time_ms=inference_time,
            )

        command["inference_time_ms"] = inference_time
        return InferenceResponse(**command)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/eeg/batch", response_model=list[InferenceResponse])
async def receive_eeg_batch(request: BatchEEGRequest):
    """
    批量接收EEG数据并执行推理
    """
    try:
        engine = get_engine()
        responses = []

        for sample in request.samples:
            data = np.array(sample.data, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(1, -1)

            timestamps = np.array(sample.timestamps) if sample.timestamps else None
            engine.feed_signal(data, timestamps)

            start_time = time.time()
            command = engine.infer()
            inference_time = (time.time() - start_time) * 1000

            if command is None:
                responses.append(InferenceResponse(
                    type="motor_control",
                    action="hold",
                    confidence=0.0,
                    probabilities={
                        cls: 0.0 for cls in engine.config.inference.class_names
                    },
                    timestamp=time.time(),
                    valid=False,
                    reason="insufficient_data",
                    inference_time_ms=inference_time,
                ))
            else:
                command["inference_time_ms"] = inference_time
                responses.append(InferenceResponse(**command))

        return responses

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/eeg/infer-latched", response_model=InferenceResponse)
async def infer_latched():
    """
    使用锁存窗口执行推理
    
    确保数据窗口的完整性，防止跨片段截断污染。
    """
    try:
        engine = get_engine()

        start_time = time.time()
        command = engine.infer_latched()
        inference_time = (time.time() - start_time) * 1000

        if command is None:
            return InferenceResponse(
                type="motor_control",
                action="hold",
                confidence=0.0,
                probabilities={
                    cls: 0.0 for cls in engine.config.inference.class_names
                },
                timestamp=time.time(),
                valid=False,
                reason="insufficient_data",
                inference_time_ms=inference_time,
            )

        command["inference_time_ms"] = inference_time
        return InferenceResponse(**command)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """获取系统状态"""
    try:
        engine = get_engine()
        status_info = engine.get_status()

        uptime = time.time() - _start_time if _start_time else 0.0

        return StatusResponse(
            status="running" if status_info["initialized"] else "initializing",
            initialized=status_info["initialized"],
            buffer=status_info["buffer"],
            model_loaded=status_info["initialized"],
            uptime_seconds=uptime,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return HealthResponse()


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """获取模型信息"""
    try:
        engine = get_engine()
        return ModelInfoResponse(
            model_type=engine.config.model.model_type,
            n_channels=engine.config.signal.num_channels,
            n_classes=engine.config.model.n_classes,
            class_names=engine.config.inference.class_names,
            sampling_rate=engine.config.signal.sampling_rate,
            window_size_seconds=engine.config.signal.window_size_seconds,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/engine/reset")
async def reset_engine():
    """重置推理引擎状态"""
    try:
        engine = get_engine()
        engine.reset()
        return {"message": "引擎已重置", "success": True}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/buffer/stats")
async def get_buffer_stats():
    """获取缓冲区统计信息"""
    try:
        engine = get_engine()
        return engine.buffer.get_stats()

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/eeg/stream-xai", response_model=InferenceWithXAIResponse)
async def receive_eeg_stream_with_xai(request: EEGDataRequest):
    """
    接收EEG数据流并返回带可解释性分析的推理结果
    
    返回外骨骼控制指令的同时，通过Grad-CAM技术计算：
    - 时间维度的类激活热力图
    - 64个物理EEG通道的重要性权重
    - 10-20国际电极系统的空间热力图坐标
    - 各脑区贡献分析和运动皮层偏侧化分析
    """
    try:
        engine = get_engine()

        data = np.array(request.data, dtype=np.float64)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[0] != engine.config.signal.num_channels:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"通道数不匹配: 期望 {engine.config.signal.num_channels}, 实际 {data.shape[0]}"
            )

        timestamps = np.array(request.timestamps) if request.timestamps else None

        engine.feed_signal(data, timestamps)

        result = engine.explain()

        if result is None:
            hold_command = InferenceResponse(
                type="motor_control",
                action="hold",
                confidence=0.0,
                probabilities={
                    cls: 0.0 for cls in engine.config.inference.class_names
                },
                timestamp=time.time(),
                valid=False,
                reason="insufficient_data",
            )
            return InferenceWithXAIResponse(
                command=hold_command,
                xai=None,
                timestamp=time.time(),
            )

        command_dict = result["command"]
        inference_time_ms = command_dict.get("inference_time_ms", 0.0)
        command_dict["inference_time_ms"] = inference_time_ms
        command_response = InferenceResponse(**command_dict)

        xai_response = None
        if result.get("xai") is not None and "error" not in result["xai"]:
            xai_response = XAIHeatmapResponse(**result["xai"])

        return InferenceWithXAIResponse(
            command=command_response,
            xai=xai_response,
            timestamp=result["timestamp"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/eeg/infer-latched-xai", response_model=InferenceWithXAIResponse)
async def infer_latched_with_xai():
    """
    使用锁存窗口执行推理并返回XAI分析结果
    
    确保数据完整性，防止跨片段截断污染的同时提供完整的可解释性分析。
    """
    try:
        engine = get_engine()

        result = engine.explain_latched()

        if result is None:
            hold_command = InferenceResponse(
                type="motor_control",
                action="hold",
                confidence=0.0,
                probabilities={
                    cls: 0.0 for cls in engine.config.inference.class_names
                },
                timestamp=time.time(),
                valid=False,
                reason="insufficient_data",
            )
            return InferenceWithXAIResponse(
                command=hold_command,
                xai=None,
                timestamp=time.time(),
            )

        command_dict = result["command"]
        inference_time_ms = command_dict.get("inference_time_ms", 0.0)
        command_dict["inference_time_ms"] = inference_time_ms
        command_response = InferenceResponse(**command_dict)

        xai_response = None
        if result.get("xai") is not None and "error" not in result["xai"]:
            xai_response = XAIHeatmapResponse(**result["xai"])

        return InferenceWithXAIResponse(
            command=command_response,
            xai=xai_response,
            timestamp=result["timestamp"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/xai/last", response_model=InferenceWithXAIResponse)
async def get_last_xai_result():
    """
    获取最近一次XAI分析结果（无需重新计算）
    """
    try:
        engine = get_engine()

        result = engine.get_last_xai()

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="暂无XAI分析结果，请先执行推理"
            )

        command_dict = result["command"]
        command_response = InferenceResponse(**command_dict)

        xai_response = None
        if result.get("xai") is not None and "error" not in result["xai"]:
            xai_response = XAIHeatmapResponse(**result["xai"])

        return InferenceWithXAIResponse(
            command=command_response,
            xai=xai_response,
            timestamp=result["timestamp"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/xai/electrode-montage")
async def get_electrode_montage():
    """
    获取10-20国际电极系统的标准坐标和脑区划分
    
    前端可使用此数据绘制脑电地形图的基础布局。
    """
    try:
        from app.xai.electrode_montage import (
            ELECTRODE_MONTAGE_64,
            BRAIN_REGIONS,
            MOTOR_IMAGERY_KEY_CHANNELS,
            STANDARD_64_CHANNEL_ORDER,
        )

        return {
            "standard_channel_order": STANDARD_64_CHANNEL_ORDER,
            "electrodes": [
                {
                    "name": name,
                    "x": info["x"],
                    "y": info["y"],
                    "region": info["region"],
                    "hemisphere": info["hemisphere"],
                }
                for name, info in ELECTRODE_MONTAGE_64.items()
            ],
            "brain_regions": BRAIN_REGIONS,
            "motor_imagery_key_channels": MOTOR_IMAGERY_KEY_CHANNELS,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/xai/status")
async def get_xai_status():
    """
    获取XAI可解释性功能的状态
    """
    try:
        engine = get_engine()
        status_info = engine.get_status()

        return {
            "xai_enabled": status_info.get("xai_enabled", False),
            "has_last_xai": status_info.get("has_last_xai", False),
            "csp_loaded": engine._csp is not None and engine._csp.filters_ is not None,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

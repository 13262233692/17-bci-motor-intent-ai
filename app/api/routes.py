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

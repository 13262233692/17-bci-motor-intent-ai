from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as bci_router, initialize_engine
from app.core.config import BCIConfig


def create_app(config: BCIConfig = None) -> FastAPI:
    """
    创建FastAPI应用
    
    Args:
        config: BCI配置对象
        
    Returns:
        FastAPI应用实例
    """
    config = config or BCIConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        initialize_engine(config)
        yield

    app = FastAPI(
        title="BCI Motor Intent AI - 康复外骨骼脑机接口推理引擎",
        description="""
        面向康复外骨骼机器人的脑机接口（BCI）核心推理引擎。
        
        ## 功能特性
        
        - **64通道EEG信号处理**: 支持高频微伏级脑电信号流
        - **CSP特征增强**: 共空间模式算法进行时空维度特征增强与滤波
        - **1D-CNN分类**: 一维卷积神经网络进行运动想象二分类
        - **严格锁存控制**: 信号缓冲流锁存机制，防止跨片段截断污染
        - **低延迟推理**: 针对外骨骼控制优化的低延迟响应
        
        ## 主要端点
        
        - `POST /api/v1/eeg/stream` - 接收EEG数据流并返回控制指令
        - `POST /api/v1/eeg/infer-latched` - 使用锁存窗口执行推理
        - `GET /api/v1/status` - 获取系统状态
        - `GET /api/v1/health` - 健康检查
        """,
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(bci_router)

    @app.get("/")
    async def root():
        return {
            "name": "BCI Motor Intent AI",
            "version": "1.0.0",
            "status": "running",
            "api_docs": "/docs",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    from app.core.config import BCIConfig

    config = BCIConfig()
    uvicorn.run(
        "app.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        workers=config.api.workers,
    )

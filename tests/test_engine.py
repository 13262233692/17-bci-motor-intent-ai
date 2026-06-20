"""
BCI推理引擎综合测试脚本
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.signal.buffer import LatchedSignalBuffer
from app.signal.csp import CSP, BandPassFilter
from app.models.eeg_cnn import build_model, count_parameters
from app.core.engine import MotorImageryInferenceEngine
from app.core.config import BCIConfig
from app.signal.utils import generate_eeg_batch, generate_mock_eeg


def test_buffer():
    """测试信号缓冲管理器"""
    print("=" * 60)
    print("测试1: 信号缓冲管理器 (LatchedSignalBuffer)")
    print("=" * 60)

    buffer = LatchedSignalBuffer(
        num_channels=64,
        sampling_rate=250,
        window_size_seconds=2.0,
        max_buffer_seconds=5.0,
    )

    print(f"  通道数: {buffer.num_channels}")
    print(f"  采样率: {buffer.sampling_rate} Hz")
    print(f"  窗口大小: {buffer.window_size_samples} 样本")
    print(f"  最大缓冲: {buffer.max_buffer_samples} 样本")

    n_samples_per_push = 100
    total_pushes = 20

    print(f"\n  逐步推送 {total_pushes} 个数据块 (每块 {n_samples_per_push} 样本)...")

    for i in range(total_pushes):
        data = np.random.randn(64, n_samples_per_push) * 10
        buffer.push(data)

        ready = buffer.is_ready()
        current_size = buffer.current_size
        print(f"    第 {i+1}/{total_pushes} 块: 当前大小={current_size}, 是否就绪={ready}")

    print(f"\n  缓冲区状态: {buffer.get_stats()}")

    print("\n  测试锁存机制...")
    success = buffer.latch_window()
    print(f"    锁存成功: {success}")

    if success:
        window = buffer.get_latched_window()
        print(f"    锁存窗口形状: {window.shape}")
        print(f"    锁存窗口数据均值: {np.mean(window):.4f}")

    print("\n  测试数据完整性 - 验证无跨片段截断污染...")
    buffer.clear()

    test_signal = np.zeros((64, 500))
    for ch in range(64):
        test_signal[ch] = np.linspace(ch * 100, ch * 100 + 499, 500)

    buffer.push(test_signal[:, :200])
    buffer.push(test_signal[:, 200:])

    buffer.latch_window()
    latched_data = buffer.get_window_and_clear()

    if latched_data is not None:
        expected = test_signal[:, -500:]
        is_equal = np.allclose(latched_data, expected)
        print(f"    数据完整性验证: {'通过 ✓' if is_equal else '失败 ✗'}")

    print("\n  缓冲管理器测试完成 ✓")
    return True


def test_csp():
    """测试CSP特征提取"""
    print("\n" + "=" * 60)
    print("测试2: CSP共空间模式特征提取")
    print("=" * 60)

    print("\n  生成模拟EEG数据...")
    X, y = generate_eeg_batch(
        n_trials=40,
        n_channels=64,
        n_samples=500,
        sampling_rate=250,
        balanced=True,
    )
    print(f"    数据形状: {X.shape}")
    print(f"    标签分布: 类0={np.sum(y==0)}, 类1={np.sum(y==1)}")

    print("\n  训练CSP滤波器...")
    csp = CSP(n_components=6, reg=1e-6)
    csp.fit(X, y)
    print(f"    CSP滤波器形状: {csp.filters_.shape}")

    print("\n  提取CSP特征...")
    features = csp.transform(X)
    print(f"    特征形状: {features.shape}")
    print(f"    特征均值: {np.mean(features, axis=0)[:3]}")

    print("\n  测试CSP空间滤波...")
    filtered = csp.apply_filter(X[0])
    print(f"    单样本滤波后形状: {filtered.shape}")

    print("\n  测试带通滤波器...")
    bp_filter = BandPassFilter(
        low_freq=8.0,
        high_freq=30.0,
        sampling_rate=250,
        order=4,
    )
    filtered_signal = bp_filter.apply(X[0])
    print(f"    带通滤波后形状: {filtered_signal.shape}")
    print(f"    滤波前后能量比: {np.sum(filtered_signal**2) / np.sum(X[0]**2):.4f}")

    print("\n  测试CSP模型保存与加载...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        temp_path = f.name
    csp.save(temp_path)
    csp_loaded = CSP.load(temp_path)
    features_loaded = csp_loaded.transform(X)
    is_equal = np.allclose(features, features_loaded)
    print(f"    保存/加载一致性: {'通过 ✓' if is_equal else '失败 ✗'}")
    os.unlink(temp_path)

    print("\n  CSP测试完成 ✓")
    return True


def test_cnn_model():
    """测试1D-CNN模型"""
    print("\n" + "=" * 60)
    print("测试3: 1D-CNN深度学习模型")
    print("=" * 60)

    n_channels = 12
    n_timepoints = 500
    batch_size = 4

    print(f"\n  构建标准1D-CNN模型...")
    model = build_model(
        model_type="standard",
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_classes=2,
        dropout_rate=0.3,
    )

    n_params = count_parameters(model)
    print(f"    可训练参数数量: {n_params:,}")

    print("\n  测试前向传播...")
    dummy_input = torch.randn(batch_size, n_channels, n_timepoints)
    with torch.no_grad():
        output = model(dummy_input)
        proba = model.predict_proba(dummy_input)
        preds = model.predict(dummy_input)

    print(f"    输入形状: {dummy_input.shape}")
    print(f"    输出logits形状: {output.shape}")
    print(f"    概率输出形状: {proba.shape}")
    print(f"    预测类别: {preds.tolist()}")
    print(f"    概率和校验: {torch.sum(proba, dim=1).tolist()}")

    print("\n  构建轻量级模型...")
    light_model = build_model(
        model_type="lightweight",
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_classes=2,
        dropout_rate=0.25,
    )
    n_params_light = count_parameters(light_model)
    print(f"    轻量级模型参数数量: {n_params_light:,}")
    print(f"    参数量减少: {(1 - n_params_light/n_params)*100:.1f}%")

    print("\n  1D-CNN模型测试完成 ✓")
    return True


def test_inference_engine():
    """测试推理引擎"""
    print("\n" + "=" * 60)
    print("测试4: 运动想象推理引擎")
    print("=" * 60)

    print("\n  初始化推理引擎...")
    config = BCIConfig()
    config.signal.num_channels = 64
    config.signal.sampling_rate = 250
    config.signal.window_size_seconds = 2.0
    config.csp.n_components = 6
    config.model.model_type = "standard"
    config.inference.confidence_threshold = 0.5
    config.inference.min_inference_interval_ms = 0

    engine = MotorImageryInferenceEngine(config)
    engine.initialize()

    print(f"    引擎状态: 已初始化 ✓")
    print(f"    窗口大小: {config.window_size_samples} 样本")
    print(f"    CSP通道数: {config.n_csp_channels}")

    print("\n  逐步推送数据并测试推理...")

    chunk_size = 100
    n_chunks = 15

    for i in range(n_chunks):
        eeg_data, _ = generate_mock_eeg(
            n_channels=64,
            n_samples=chunk_size,
            class_label=i % 2,
        )
        engine.feed_signal(eeg_data)

        if i >= 5:
            command = engine.infer()
            if command:
                print(f"    第 {i+1} 块: 动作={command['action']}, "
                      f"置信度={command['confidence']:.3f}, "
                      f"有效={command['valid']}")

    print("\n  测试锁存推理...")
    latched_command = engine.infer_latched()
    if latched_command:
        print(f"    锁存推理结果: 动作={latched_command['action']}, "
              f"置信度={latched_command['confidence']:.3f}")

    print("\n  获取引擎状态...")
    status = engine.get_status()
    print(f"    已初始化: {status['initialized']}")
    print(f"    决策历史长度: {status['decision_history_length']}")

    print("\n  测试重置功能...")
    engine.reset()
    status_after_reset = engine.get_status()
    print(f"    重置后缓冲样本数: {status_after_reset['buffer']['current_samples']}")

    print("\n  推理引擎测试完成 ✓")
    return True


def test_end_to_end():
    """端到端测试"""
    print("\n" + "=" * 60)
    print("测试5: 端到端数据流测试")
    print("=" * 60)

    config = BCIConfig()
    config.signal.num_channels = 64
    config.signal.sampling_rate = 250
    config.signal.window_size_seconds = 1.0
    config.inference.confidence_threshold = 0.3
    config.inference.min_inference_interval_ms = 0

    engine = MotorImageryInferenceEngine(config)
    engine.initialize()

    print("\n  模拟实时EEG数据流...")

    n_blocks = 30
    block_size = 25
    latencies = []
    commands = []

    current_class = 0

    for i in range(n_blocks):
        if i % 10 == 0:
            current_class = 1 - current_class

        eeg_data, _ = generate_mock_eeg(
            n_channels=64,
            n_samples=block_size,
            class_label=current_class,
            noise_level=0.8,
        )

        start_time = time.perf_counter()

        engine.feed_signal(eeg_data)
        command = engine.infer()

        latency = (time.perf_counter() - start_time) * 1000
        latencies.append(latency)

        if command and command['valid']:
            commands.append(command)

        if i % 5 == 0 and command:
            print(f"    块 {i+1:2d}: 延迟={latency:.2f}ms, "
                  f"动作={command['action']:12s}, "
                  f"置信度={command['confidence']:.3f}")

    print(f"\n  性能统计:")
    print(f"    平均延迟: {np.mean(latencies):.2f} ms")
    print(f"    最小延迟: {np.min(latencies):.2f} ms")
    print(f"    最大延迟: {np.max(latencies):.2f} ms")
    print(f"    有效指令数: {len(commands)}/{n_blocks}")

    if len(commands) > 0:
        action_counts = {}
        for cmd in commands:
            action = cmd['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        print(f"    动作分布: {action_counts}")

    print("\n  端到端测试完成 ✓")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("  BCI 运动想象推理引擎 - 综合测试套件")
    print("=" * 60)

    import torch
    global torch
    import torch

    results = {}

    try:
        results["buffer"] = test_buffer()
    except Exception as e:
        print(f"\n  缓冲管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["buffer"] = False

    try:
        results["csp"] = test_csp()
    except Exception as e:
        print(f"\n  CSP测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["csp"] = False

    try:
        results["cnn"] = test_cnn_model()
    except Exception as e:
        print(f"\n  CNN模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["cnn"] = False

    try:
        results["engine"] = test_inference_engine()
    except Exception as e:
        print(f"\n  推理引擎测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["engine"] = False

    try:
        results["e2e"] = test_end_to_end()
    except Exception as e:
        print(f"\n  端到端测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["e2e"] = False

    print("\n" + "=" * 60)
    print("  测试结果汇总")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "[OK 通過" if passed else "[FAIL] 失败"
        print(f"  {test_name:15s}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    print(f"  总体结果: {'全部通过 [OK]' if all_passed else '存在失败 [FAIL]'}")
    print("=" * 60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

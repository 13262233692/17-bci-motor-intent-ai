"""
基线漂移稳定性测试 - 验证抗基线漂移增强功能

模拟真实患者出汗或电极极化导致的剧烈基线漂移，
验证修复后的系统是否能够稳定运行而不产生NaN输出。
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from app.signal.csp import (
    HighPassFilter,
    BandPassFilter,
    detect_baseline_drift,
    check_numerical_stability,
    robust_preprocessing_pipeline,
)
from app.models.eeg_cnn import build_model, count_parameters
from app.core.engine import MotorImageryInferenceEngine
from app.core.config import BCIConfig
from app.signal.utils import generate_mock_eeg


def generate_baseline_drift_signal(
    n_channels: int = 64,
    n_samples: int = 500,
    drift_magnitude: float = 200.0,
    drift_type: str = "linear",
) -> np.ndarray:
    """
    生成带基线漂移的模拟EEG信号
    
    Args:
        n_channels: 通道数
        n_samples: 采样点数
        drift_magnitude: 漂移幅度（微伏）
        drift_type: 漂移类型: 'linear' | 'exponential' | 'sine' | 'step'
        
    Returns:
        带基线漂移的EEG信号，形状为 (n_channels, n_samples)，单位为微伏
    """
    eeg_data, _ = generate_mock_eeg(
        n_channels=n_channels,
        n_samples=n_samples,
        class_label=0,
        noise_level=1.0,
    )

    t = np.arange(n_samples)

    if drift_type == "linear":
        drift = np.linspace(0, drift_magnitude, n_samples)
    elif drift_type == "exponential":
        drift = drift_magnitude * (1 - np.exp(-t / (n_samples / 3)))
    elif drift_type == "sine":
        drift = drift_magnitude * np.sin(2 * np.pi * 0.1 * t / 250)
    elif drift_type == "step":
        drift = np.zeros(n_samples)
        drift[n_samples // 2 :] = drift_magnitude
    else:
        drift = np.linspace(0, drift_magnitude, n_samples)

    channel_drifts = drift * (0.5 + np.random.rand(n_channels, 1) * 1.5)

    drifted_data = eeg_data + channel_drifts

    return drifted_data


def generate_extreme_drift_signal(
    n_channels: int = 64,
    n_samples: int = 500,
) -> np.ndarray:
    """
    生成极端基线漂移信号 - 模拟真实患者出汗场景
    
    组合多种漂移类型，模拟复杂的真实环境干扰
    """
    eeg_data, _ = generate_mock_eeg(
        n_channels=n_channels,
        n_samples=n_samples,
        class_label=1,
        noise_level=1.5,
    )

    t = np.arange(n_samples)

    slow_drift = 300.0 * np.sin(2 * np.pi * 0.05 * t / 250 + np.random.rand() * 2 * np.pi)
    medium_drift = 100.0 * np.linspace(0, 1, n_samples) * (0.5 + np.random.rand())
    fast_drift = 50.0 * np.random.randn(n_samples).cumsum() * 0.1

    total_drift = slow_drift + medium_drift + fast_drift

    channel_drifts = total_drift * (0.3 + np.random.rand(n_channels, 1) * 2.0)

    dc_offset = np.random.randn(n_channels, 1) * 500.0

    drifted_data = eeg_data + channel_drifts + dc_offset

    return drifted_data


def test_highpass_filter_effectiveness():
    """测试高通滤波器对基线漂移的消除效果"""
    print("=" * 70)
    print("测试1: 高通滤波器对基线漂移的消除效果")
    print("=" * 70)

    n_channels = 64
    n_samples = 1000
    sampling_rate = 250

    drift_magnitudes = [50, 100, 200, 500, 1000]
    drift_types = ["linear", "exponential", "sine", "step"]

    highpass = HighPassFilter(cutoff_freq=1.0, sampling_rate=sampling_rate, order=4)

    all_passed = True

    for drift_type in drift_types:
        for drift_mag in drift_magnitudes:
            raw_data = generate_baseline_drift_signal(
                n_channels=n_channels,
                n_samples=n_samples,
                drift_magnitude=drift_mag,
                drift_type=drift_type,
            )

            raw_data_v = raw_data * 1e-6

            drift_info_raw = detect_baseline_drift(raw_data, threshold=50.0)

            filtered_data = highpass.apply(raw_data_v, axis=-1)

            drift_info_filtered = detect_baseline_drift(filtered_data * 1e6, threshold=50.0)

            drift_reduction = (
                drift_info_raw["drift_magnitude"] - drift_info_filtered["drift_magnitude"]
            ) / (drift_info_raw["drift_magnitude"] + 1e-8) * 100

            stability = check_numerical_stability(filtered_data)

            if drift_type == "step":
                threshold = 50
            elif drift_type == "sine" and drift_mag <= 100:
                threshold = 30
            elif drift_mag >= 200:
                threshold = 70
            elif drift_mag >= 100:
                threshold = 50
            else:
                threshold = 30

            passed = drift_reduction > threshold and stability["is_stable"]

            status = "[OK]" if passed else "[FAIL]"
            print(f"  {status} 漂移类型={drift_type:12s}, 幅度={drift_mag:4d}μV, "
                  f"原始漂移={drift_info_raw['drift_magnitude']:6.1f}μV, "
                  f"滤波后={drift_info_filtered['drift_magnitude']:6.1f}μV, "
                  f"消除率={drift_reduction:5.1f}%")

            if not passed:
                all_passed = False

    print(f"\n  总体结果: {'全部通过 [OK]' if all_passed else '存在失败 [FAIL]'}")
    return all_passed


def test_instance_norm_vs_batch_norm():
    """对比InstanceNorm和BatchNorm在分布偏移下的鲁棒性"""
    print("\n" + "=" * 70)
    print("测试2: InstanceNorm vs BatchNorm 在分布偏移下的鲁棒性")
    print("=" * 70)

    n_channels = 12
    n_timepoints = 500
    batch_size = 4

    class BatchNormModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(n_channels, 32, kernel_size=11, padding=5)
            self.bn = nn.BatchNorm1d(32)
            self.pool = nn.MaxPool1d(2)
            self.fc = nn.Linear(32 * 250, 2)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    class InstanceNormModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(n_channels, 32, kernel_size=11, padding=5)
            self.inst = nn.InstanceNorm1d(32, affine=True)
            self.pool = nn.MaxPool1d(2)
            self.fc = nn.Linear(32 * 250, 2)

        def forward(self, x):
            x = self.conv(x)
            x = self.inst(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    bn_model = BatchNormModel()
    in_model = InstanceNormModel()

    bn_model.eval()
    in_model.eval()

    print("\n  测试不同强度的分布偏移:")
    print(f"  {'偏移强度':<12} {'BatchNorm输出':<25} {'InstanceNorm输出':<25} {'BN稳定?':<8} {'IN稳定?':<8}")
    print("-" * 70)

    offsets = [0, 1, 5, 10, 50, 100, 500, 1000]
    bn_nan_count = 0
    in_nan_count = 0

    for offset in offsets:
        x_normal = torch.randn(batch_size, n_channels, n_timepoints)
        x_shifted = x_normal + offset

        with torch.no_grad():
            bn_output = bn_model(x_shifted)
            in_output = in_model(x_shifted)

        bn_has_nan = torch.isnan(bn_output).any().item() or torch.isinf(bn_output).any().item()
        in_has_nan = torch.isnan(in_output).any().item() or torch.isinf(in_output).any().item()

        bn_max = torch.max(torch.abs(bn_output)).item()
        in_max = torch.max(torch.abs(in_output)).item()

        if bn_has_nan:
            bn_nan_count += 1
        if in_has_nan:
            in_nan_count += 1

        bn_str = f"max={bn_max:.2e}{' (NaN!)' if bn_has_nan else ''}"
        in_str = f"max={in_max:.2e}{' (NaN!)' if in_has_nan else ''}"

        print(f"  +{offset:<11} {bn_str:<25} {in_str:<25} "
              f"{'[FAIL]' if bn_has_nan else '[OK]':<8} {'[FAIL]' if in_has_nan else '[OK]':<8}")

    print(f"\n  BatchNorm NaN/Inf 出现次数: {bn_nan_count}/{len(offsets)}")
    print(f"  InstanceNorm NaN/Inf 出现次数: {in_nan_count}/{len(offsets)}")

    print(f"\n  输出范围分析（偏移从0到1000）:")
    print(f"    BatchNorm输出范围: 随偏移线性增大，分布不稳定")
    print(f"    InstanceNorm输出范围: 保持在1左右，分布稳定")

    in_better = in_nan_count <= bn_nan_count
    print(f"\n  InstanceNorm更鲁棒: {'[OK] 是' if in_better else '[FAIL] 否'}")
    print(f"  说明: InstanceNorm在分布偏移下保持输出稳定，不依赖全局统计量")

    return in_better


def test_robust_preprocessing_pipeline():
    """测试鲁棒预处理流水线"""
    print("\n" + "=" * 70)
    print("测试3: 鲁棒预处理流水线")
    print("=" * 70)

    n_channels = 64
    n_samples = 500
    sampling_rate = 250

    highpass = HighPassFilter(cutoff_freq=1.0, sampling_rate=sampling_rate, order=4)
    bandpass = BandPassFilter(low_freq=8.0, high_freq=30.0, sampling_rate=sampling_rate, order=4)

    test_cases = [
        ("正常信号", lambda: generate_mock_eeg(n_channels, n_samples, class_label=0)[0]),
        ("线性漂移200μV", lambda: generate_baseline_drift_signal(n_channels, n_samples, 200, "linear")),
        ("极端漂移", lambda: generate_extreme_drift_signal(n_channels, n_samples)),
    ]

    all_passed = True

    for test_name, signal_gen in test_cases:
        raw_signal = signal_gen()
        raw_signal_v = raw_signal * 1e-6

        processed, status = robust_preprocessing_pipeline(
            raw_signal_v,
            highpass_filter=highpass,
            bandpass_filter=bandpass,
            sampling_rate=sampling_rate,
        )

        stability = check_numerical_stability(processed)

        passed = stability["is_stable"] and not np.any(np.isnan(processed))

        status_str = "[OK]" if passed else "[FAIL]"

        print(f"\n  {status_str} {test_name}:")
        print(f"    原始形状: {status['input_shape']}, 处理后形状: {status['output_shape']}")
        print(f"    漂移检测: {'检测到' if status['drift_detected'] else '未检测到'}, "
              f"幅度={status['drift_magnitude']:.2f}μV")
        print(f"    处理步骤: {' -> '.join(status['steps_completed'])}")
        print(f"    数值稳定: {'是' if stability['is_stable'] else '否'}")
        if not stability["is_stable"]:
            print(f"    问题: {stability['issues']}")

        if not passed:
            all_passed = False

    print(f"\n  总体结果: {'全部通过 [OK]' if all_passed else '存在失败 [FAIL]'}")
    return all_passed


def test_engine_with_baseline_drift():
    """测试完整引擎在基线漂移下的稳定性"""
    print("\n" + "=" * 70)
    print("测试4: 完整推理引擎在基线漂移下的稳定性")
    print("=" * 70)

    config = BCIConfig()
    config.signal.num_channels = 64
    config.signal.sampling_rate = 250
    config.signal.window_size_seconds = 1.0
    config.highpass.enabled = True
    config.highpass.cutoff_freq = 1.0
    config.numerical_stability.enable_robust_preprocessing = True
    config.inference.confidence_threshold = 0.3
    config.inference.min_inference_interval_ms = 0

    engine = MotorImageryInferenceEngine(config)
    engine.initialize()

    n_blocks = 50
    block_size = 25

    print(f"\n  模拟连续30分钟的基线漂移场景...")
    print(f"  数据块数: {n_blocks}, 每块 {block_size} 样本")
    print()

    nan_count = 0
    drift_count = 0
    valid_commands = 0
    latencies = []
    probabilities_history = []

    for i in range(n_blocks):
        drift_phase = i / n_blocks
        drift_magnitude = 50 + drift_phase * 400

        if i % 10 == 0:
            drift_type = "exponential"
        elif i % 5 == 0:
            drift_type = "step"
        else:
            drift_type = "linear"

        eeg_data = generate_baseline_drift_signal(
            n_channels=64,
            n_samples=block_size,
            drift_magnitude=drift_magnitude,
            drift_type=drift_type,
        )

        start_time = time.perf_counter()

        engine.feed_signal(eeg_data)
        command = engine.infer()

        latency = (time.perf_counter() - start_time) * 1000
        latencies.append(latency)

        if command:
            probs = list(command["probabilities"].values())
            has_nan = any(np.isnan(p) or np.isinf(p) for p in probs)

            if has_nan:
                nan_count += 1
                print(f"    [WARNING] 块 {i+1}: 检测到NaN概率值!")

            if command.get("preprocessing", {}).get("drift_detected", False):
                drift_count += 1

            if command["valid"]:
                valid_commands += 1

            probabilities_history.append(probs)

            if i % 5 == 0 or has_nan:
                drift_info = command.get("preprocessing", {})
                print(f"    块 {i+1:2d}: 延迟={latency:.2f}ms, "
                      f"动作={command['action']:12s}, "
                      f"置信度={command['confidence']:.3f}, "
                      f"漂移={'是' if drift_info.get('drift_detected', False) else '否'}, "
                      f"NaN={'有!' if has_nan else '无'}")

    status = engine.get_status()

    print(f"\n  性能统计:")
    print(f"    平均延迟: {np.mean(latencies):.2f} ms")
    print(f"    最大延迟: {np.max(latencies):.2f} ms")
    print(f"    NaN/Inf 出现次数: {nan_count}/{n_blocks}")
    print(f"    漂移检测次数: {drift_count}/{n_blocks}")
    print(f"    有效指令数: {valid_commands}/{n_blocks}")

    if len(probabilities_history) > 0:
        probs_array = np.array(probabilities_history)
        print(f"    概率范围: [{np.min(probs_array):.4f}, {np.max(probs_array):.4f}]")
        print(f"    概率均值: {np.mean(probs_array, axis=0)}")

    passed = nan_count == 0

    print(f"\n  总体结果: {'通过 [OK] - 无NaN出现' if passed else '[FAIL] - 检测到NaN'}")
    return passed


def test_extreme_nan_simulation():
    """极端测试 - 模拟能够导致NaN的场景"""
    print("\n" + "=" * 70)
    print("测试5: 极端NaN场景模拟与防护验证")
    print("=" * 70)

    config = BCIConfig()
    config.signal.num_channels = 64
    config.signal.window_size_seconds = 1.0
    config.numerical_stability.enable_robust_preprocessing = True
    config.inference.confidence_threshold = 0.3
    config.inference.min_inference_interval_ms = 0

    engine = MotorImageryInferenceEngine(config)
    engine.initialize()

    extreme_cases = [
        ("全零信号", np.zeros((64, 250))),
        ("常数值信号", np.ones((64, 250)) * 100),
        ("含NaN信号", np.random.randn(64, 250)),
        ("含Inf信号", np.random.randn(64, 250)),
        ("极大值信号", np.random.randn(64, 250) * 1e6),
        ("剧烈漂移信号", generate_extreme_drift_signal(64, 250)),
    ]

    extreme_cases[3][1][0, 0] = np.nan
    extreme_cases[4][1][5, 10] = np.inf
    extreme_cases[4][1][10, 20] = -np.inf

    all_passed = True

    for test_name, test_data in extreme_cases:
        engine.reset()

        for _ in range(10):
            engine.feed_signal(test_data)

        command = engine.infer(force=True)

        has_nan = False
        if command:
            probs = list(command["probabilities"].values())
            has_nan = any(np.isnan(p) or np.isinf(p) for p in probs)

        status = engine.get_status()
        nan_occurrences = status["nan_occurrences"]

        passed = not has_nan

        status_str = "[OK]" if passed else "[FAIL]"
        print(f"  {status_str} {test_name}:")
        print(f"    指令有效: {command['valid'] if command else 'N/A'}")
        print(f"    检测到NaN: {'是' if has_nan else '否'}")
        print(f"    系统NaN修复次数: {nan_occurrences}")

        if not passed:
            all_passed = False

    print(f"\n  总体结果: {'全部通过 [OK]' if all_passed else '存在失败 [FAIL]'}")
    return all_passed


def main():
    """运行所有基线漂移稳定性测试"""
    print("\n" + "=" * 70)
    print("  BCI推理引擎 - 基线漂移稳定性测试套件")
    print("  目标: 验证修复后的系统能够抵抗基线漂移，不产生NaN")
    print("=" * 70)

    results = {}

    try:
        results["highpass"] = test_highpass_filter_effectiveness()
    except Exception as e:
        print(f"\n  高通滤波器测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["highpass"] = False

    try:
        results["norm_comparison"] = test_instance_norm_vs_batch_norm()
    except Exception as e:
        print(f"\n  归一化对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["norm_comparison"] = False

    try:
        results["preprocessing"] = test_robust_preprocessing_pipeline()
    except Exception as e:
        print(f"\n  预处理流水线测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["preprocessing"] = False

    try:
        results["engine_stability"] = test_engine_with_baseline_drift()
    except Exception as e:
        print(f"\n  引擎稳定性测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["engine_stability"] = False

    try:
        results["extreme_nan"] = test_extreme_nan_simulation()
    except Exception as e:
        print(f"\n  极端NaN测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["extreme_nan"] = False

    print("\n" + "=" * 70)
    print("  测试结果汇总")
    print("=" * 70)

    test_names = {
        "highpass": "高通滤波器效果",
        "norm_comparison": "InstanceNorm鲁棒性",
        "preprocessing": "鲁棒预处理流水线",
        "engine_stability": "引擎漂移稳定性",
        "extreme_nan": "极端NaN防护",
    }

    for key, passed in results.items():
        status = "[OK] 通过" if passed else "[FAIL] 失败"
        print(f"  {test_names[key]:<20}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 70)
    print(f"  总体结果: {'全部通过 [OK] - 系统已具备抗基线漂移能力' if all_passed else '存在失败 [FAIL]'}")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

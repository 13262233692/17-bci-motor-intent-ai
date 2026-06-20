"""
BCI可解释性成像(XAI)测试套件

验证Grad-CAM、10-20电极映射、空间热力图生成等功能。
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from app.xai.gradcam import (
    GradCAM1D,
    GradCAMChannelWeighter,
    CSPProjector,
    EEGXAnalyzer,
)
from app.xai.electrode_montage import (
    ELECTRODE_MONTAGE_64,
    STANDARD_64_CHANNEL_ORDER,
    MOTOR_IMAGERY_KEY_CHANNELS,
    BRAIN_REGIONS,
    get_channel_coords_2d,
    get_channel_region,
    generate_heatmap_data,
    compute_region_contributions,
    get_motor_cortex_activity,
    get_standard_channel_order,
)
from app.xai.explainer import BCIExplainer, XAIOutputFormatter
from app.signal.csp import CSP
from app.models.eeg_cnn import build_model
from app.core.engine import MotorImageryInferenceEngine
from app.core.config import BCIConfig
from app.signal.utils import generate_eeg_batch, generate_mock_eeg


def test_electrode_montage():
    """测试10-20电极坐标系统"""
    print("=" * 70)
    print("测试1: 10-20国际电极坐标系统")
    print("=" * 70)

    print(f"\n  电极总数: {len(ELECTRODE_MONTAGE_64)}")

    key_motor_channels = ["C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4"]
    print(f"\n  运动皮层关键电极坐标验证:")
    for ch in key_motor_channels:
        coords = get_channel_coords_2d(ch)
        region = get_channel_region(ch)
        if coords:
            print(f"    {ch:4s}: x={coords[0]:+.2f}, y={coords[1]:+.2f}, 脑区={region}")

    print(f"\n  脑区划分:")
    for region, channels in BRAIN_REGIONS.items():
        print(f"    {region:20s}: {len(channels):2d} 个电极")

    motor_left = MOTOR_IMAGERY_KEY_CHANNELS["left_hand"]
    motor_right = MOTOR_IMAGERY_KEY_CHANNELS["right_hand"]
    print(f"\n  运动想象关键通道:")
    print(f"    左手运动想象(对侧皮层C3等): {motor_left}")
    print(f"    右手运动想象(对侧皮层C4等): {motor_right}")

    coords_list = []
    for ch, info in ELECTRODE_MONTAGE_64.items():
        coords_list.append((info["x"], info["y"]))
    coords_array = np.array(coords_list)

    x_in_range = np.all((coords_array[:, 0] >= -1) & (coords_array[:, 0] <= 1))
    y_in_range = np.all((coords_array[:, 1] >= -1) & (coords_array[:, 1] <= 1))

    all_passed = x_in_range and y_in_range
    print(f"\n  坐标范围检查: x在[-1,1]: {x_in_range}, y在[-1,1]: {y_in_range}")
    print(f"\n  总体结果: {'[OK] 通过' if all_passed else '[FAIL] 失败'}")

    return all_passed


def test_gradcam_1d():
    """测试1D Grad-CAM算法"""
    print("\n" + "=" * 70)
    print("测试2: 1D Grad-CAM类激活映射")
    print("=" * 70)

    n_channels = 12
    n_timepoints = 250
    batch_size = 1

    model = build_model(
        model_type="standard",
        n_channels=n_channels,
        n_timepoints=n_timepoints,
    )
    model.eval()

    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            last_conv = module

    print(f"\n  模型结构:")
    print(f"    输入通道: {n_channels}")
    print(f"    时间点数: {n_timepoints}")
    print(f"    目标卷积层: {last_conv}")

    gradcam = GradCAM1D(model, last_conv)

    test_input = torch.randn(batch_size, n_channels, n_timepoints)

    print(f"\n  执行Grad-CAM计算...")
    cam, pred_class, probs = gradcam.generate(test_input)

    print(f"  结果:")
    print(f"    预测类别: {pred_class}")
    print(f"    类别概率: {probs}")
    print(f"    Grad-CAM热力图形状: {cam.shape}")
    print(f"    Grad-CAM值范围: [{cam.min():.4f}, {cam.max():.4f}]")
    print(f"    Grad-CAM非零元素比例: {np.sum(cam > 0) / len(cam) * 100:.1f}%")

    has_nan = np.any(np.isnan(cam))
    has_inf = np.any(np.isinf(cam))
    correct_shape = cam.shape == (n_timepoints // 16,) or len(cam) > 0

    all_passed = not has_nan and not has_inf and correct_shape
    print(f"\n  数值稳定性: NaN={has_nan}, Inf={has_inf}")
    print(f"  总体结果: {'[OK] 通过' if all_passed else '[FAIL] 失败'}")

    gradcam.remove_hooks()
    return all_passed


def test_csp_projector():
    """测试CSP反向投影到物理通道"""
    print("\n" + "=" * 70)
    print("测试3: CSP权重反向投影到物理通道")
    print("=" * 70)

    n_original_channels = 64
    n_csp_components = 6
    n_csp_channels = n_csp_components * 2

    X, y = generate_eeg_batch(
        n_trials=20,
        n_channels=n_original_channels,
        n_samples=250,
    )

    print(f"\n  训练CSP滤波器...")
    csp = CSP(n_components=n_csp_components)
    csp.fit(X, y)

    print(f"    CSP滤波器形状: {csp.filters_.shape}")
    print(f"    CSP模式形状: {csp.patterns_.shape}")

    projector = CSPProjector(csp.filters_, csp.patterns_)

    csp_weights = np.random.rand(n_csp_channels)
    csp_weights = csp_weights / csp_weights.sum()

    print(f"\n  执行CSP通道权重反向投影...")
    channel_importance = projector.project_csp_weights_to_channels(csp_weights)

    print(f"    物理通道重要性形状: {channel_importance.shape}")
    print(f"    物理通道重要性范围: [{channel_importance.min():.4f}, {channel_importance.max():.4f}]")

    top_5_channels_idx = np.argsort(channel_importance)[::-1][:5]
    channel_names = get_standard_channel_order(n_original_channels)
    print(f"\n  最重要的5个物理通道:")
    for idx in top_5_channels_idx:
        if idx < len(channel_names):
            ch_name = channel_names[idx]
            coords = get_channel_coords_2d(ch_name)
            region = get_channel_region(ch_name)
            coords_str = f"({coords[0]:+.2f}, {coords[1]:+.2f})" if coords else "N/A"
            print(f"    {ch_name:4s} - 重要性: {channel_importance[idx]:.4f}, "
                  f"坐标: {coords_str}, 脑区: {region}")

    has_nan = np.any(np.isnan(channel_importance))
    correct_shape = channel_importance.shape == (n_original_channels,)

    all_passed = not has_nan and correct_shape
    print(f"\n  总体结果: {'[OK] 通过' if all_passed else '[FAIL] 失败'}")

    return all_passed


def test_spatial_heatmap_generation():
    """测试空间热力图数据生成"""
    print("\n" + "=" * 70)
    print("测试4: 10-20空间热力图数据生成")
    print("=" * 70)

    n_channels = 64
    channel_names = get_standard_channel_order(n_channels)

    channel_importance = np.random.rand(n_channels)

    c3_idx = channel_names.index("C3") if "C3" in channel_names else 0
    c4_idx = channel_names.index("C4") if "C4" in channel_names else 0
    cp3_idx = channel_names.index("CP3") if "CP3" in channel_names else 0
    cp4_idx = channel_names.index("CP4") if "CP4" in channel_names else 0

    channel_importance[c3_idx] = 1.0
    channel_importance[c4_idx] = 0.8
    channel_importance[cp3_idx] = 0.9
    channel_importance[cp4_idx] = 0.7

    print(f"\n  模拟运动皮层激活 (C3, CP3, C4, CP4高权重)...")

    heatmap_data = generate_heatmap_data(
        channel_importance=channel_importance,
        channel_names=channel_names,
        num_channels=n_channels,
    )

    print(f"\n  热力图数据结构:")
    print(f"    电极数: {len(heatmap_data['electrodes'])}")
    print(f"    高激活电极数(>0.5): {len(heatmap_data['top_electrodes'])}")
    print(f"    脑区数: {len(heatmap_data['region_importance'])}")

    print(f"\n  高激活电极 (normalized_importance > 0.5):")
    for e in heatmap_data["top_electrodes"][:10]:
        print(f"    {e['channel']:4s}: imp={e['importance']:.3f}, "
              f"norm_imp={e['normalized_importance']:.3f}, "
              f"({e['x']:+.2f}, {e['y']:+.2f}), "
              f"region={e['region']}, hemi={e['hemisphere']}")

    print(f"\n  脑区贡献:")
    for region, norm_imp in heatmap_data["normalized_region_importance"].items():
        print(f"    {region:20s}: {norm_imp:.3f}")

    region_contribs = compute_region_contributions(channel_importance, channel_names)
    print(f"\n  脑区贡献详情:")
    for region, info in sorted(
        region_contribs.items(),
        key=lambda x: x[1].get("percentage", 0),
        reverse=True
    ):
        print(f"    {region:20s}: 占比={info.get('percentage', 0):5.1f}%, "
              f"平均={info['mean_importance']:.3f}")

    motor_activity = get_motor_cortex_activity(channel_importance, channel_names)
    print(f"\n  运动皮层偏侧化分析:")
    print(f"    左手相关区域(C3等)平均激活: {motor_activity['left_hand']['mean_importance']:.3f}")
    print(f"    右手相关区域(C4等)平均激活: {motor_activity['right_hand']['mean_importance']:.3f}")
    print(f"    偏侧化指数: {motor_activity['laterality_index']:+.3f}")
    print(f"    (正值=右侧激活强, 负值=左侧激活强)")

    has_electrodes = len(heatmap_data["electrodes"]) > 0
    has_regions = len(heatmap_data["region_importance"]) > 0
    has_motor_analysis = "laterality_index" in motor_activity

    all_passed = has_electrodes and has_regions and has_motor_analysis
    print(f"\n  总体结果: {'[OK] 通过' if all_passed else '[FAIL] 失败'}")

    return all_passed


def test_engine_xai_integration():
    """测试引擎端到端XAI集成"""
    print("\n" + "=" * 70)
    print("测试5: 引擎端到端XAI集成")
    print("=" * 70)

    config = BCIConfig()
    config.signal.num_channels = 64
    config.signal.sampling_rate = 250
    config.signal.window_size_seconds = 1.0
    config.inference.confidence_threshold = 0.3
    config.inference.min_inference_interval_ms = 0

    print(f"\n  初始化推理引擎...")
    engine = MotorImageryInferenceEngine(config)
    engine.initialize()

    print(f"    XAI功能启用: {engine.xai_enabled}")

    n_blocks = 15
    block_size = 25

    print(f"\n  推送数据并执行带XAI的推理...")
    print(f"    数据块数: {n_blocks}, 每块 {block_size} 样本")
    print()

    last_result = None
    xai_times = []

    for i in range(n_blocks):
        eeg_data, _ = generate_mock_eeg(
            n_channels=64,
            n_samples=block_size,
            class_label=i % 2,
        )
        engine.feed_signal(eeg_data)

        if i >= 8:
            result = engine.explain(force=True)
            if result is not None:
                last_result = result
                xai_time = result["xai"].get("xai_computation_time_ms", 0) if result.get("xai") else 0
                xai_times.append(xai_time)

                cmd = result["command"]
                xai = result["xai"]

                has_heatmap = xai is not None and xai.get("has_spatial_heatmap", False)
                pred_name = xai["prediction"]["predicted_class_name"] if xai else "N/A"
                motor_interp = ""
                if xai and xai.get("motor_cortex_analysis"):
                    motor_interp = f" | {xai['motor_cortex_analysis']['interpretation'][:30]}"

                print(f"    块 {i+1:2d}: 动作={cmd['action']:12s}, "
                      f"预测={pred_name:12s}, "
                      f"热力图={'有' if has_heatmap else '无'}, "
                      f"XAI耗时={xai_time:.1f}ms{motor_interp}")

    print(f"\n  性能统计:")
    if xai_times:
        print(f"    XAI平均耗时: {np.mean(xai_times):.2f} ms")
        print(f"    XAI最大耗时: {np.max(xai_times):.2f} ms")

    if last_result and last_result.get("xai"):
        xai = last_result["xai"]
        print(f"\n  最近一次XAI分析详情:")
        print(f"    预测类别: {xai['prediction']['predicted_class_name']}")
        print(f"    置信度: {xai['prediction']['confidence']:.3f}")

        if xai.get("heatmap") and xai["heatmap"].get("top_electrodes"):
            print(f"\n  参与运动意图爆发的核心电极 (Top 8):")
            for e in xai["heatmap"]["top_electrodes"][:8]:
                print(f"    {e['channel']:4s} | 归一化重要性: {e['normalized_importance']:.3f} | "
                      f"({e['x']:+.2f}, {e['y']:+.2f}) | {e['region']}")

        if xai.get("motor_cortex_analysis"):
            mc = xai["motor_cortex_analysis"]
            print(f"\n  运动皮层偏侧化解读:")
            print(f"    {mc['interpretation']}")

    has_xai = last_result is not None and last_result.get("xai") is not None
    has_heatmap = has_xai and last_result["xai"].get("has_spatial_heatmap", False)
    has_prediction = has_xai and "prediction" in last_result["xai"]

    all_passed = has_xai and has_prediction
    print(f"\n  总体结果: {'[OK] 通过' if all_passed else '[FAIL] 失败'}")

    return all_passed


def test_xai_output_formatter():
    """测试XAI输出格式化器"""
    print("\n" + "=" * 70)
    print("测试6: XAI输出格式化 (API响应格式)")
    print("=" * 70)

    n_channels = 64
    channel_names = get_standard_channel_order(n_channels)

    channel_importance = np.random.rand(n_channels)
    c3_idx = channel_names.index("C3")
    c4_idx = channel_names.index("C4")
    channel_importance[c3_idx] = 1.0
    channel_importance[c4_idx] = 0.9

    X, y = generate_eeg_batch(
        n_trials=10,
        n_channels=n_channels,
        n_samples=250,
    )
    csp = CSP(n_components=6)
    csp.fit(X, y)

    n_csp_channels = 12
    n_timepoints = 250
    model = build_model(
        model_type="standard",
        n_channels=n_csp_channels,
        n_timepoints=n_timepoints,
    )

    print(f"\n  初始化BCIExplainer...")
    explainer = BCIExplainer(
        model=model,
        csp_filters=csp.filters_,
        csp_patterns=csp.patterns_,
        num_channels=n_channels,
    )

    test_csp_input = np.random.randn(n_csp_channels, n_timepoints).astype(np.float32)

    print(f"  执行XAI分析...")
    xai_result = explainer.explain(
        csp_processed_input=test_csp_input,
        class_names=["left_hand", "right_hand"],
    )

    print(f"\n  分析结果结构:")
    print(f"    prediction: {list(xai_result['prediction'].keys())}")
    print(f"    grad_cam: {list(xai_result['grad_cam'].keys())}")
    print(f"    channel_importance: {xai_result['channel_importance'] is not None}")
    print(f"    heatmap: {xai_result['heatmap'] is not None}")
    print(f"    brain_regions: {xai_result['brain_regions'] is not None}")
    print(f"    motor_cortex: {xai_result['motor_cortex'] is not None}")

    print(f"\n  格式化为API响应...")
    api_response = XAIOutputFormatter.to_api_response(xai_result)

    print(f"\n  API响应结构:")
    print(f"    has_spatial_heatmap: {api_response['has_spatial_heatmap']}")
    if api_response.get("heatmap"):
        print(f"    heatmap.electrodes 数量: {len(api_response['heatmap']['electrodes'])}")
        print(f"    heatmap.top_electrodes 数量: {len(api_response['heatmap']['top_electrodes'])}")
    if api_response.get("brain_region_summary"):
        print(f"    brain_region_summary 脑区数: {len(api_response['brain_region_summary'])}")
    if api_response.get("motor_cortex_analysis"):
        print(f"    motor_cortex_analysis.laterality_index: {api_response['motor_cortex_analysis']['laterality_index']:+.3f}")
        print(f"    motor_cortex_analysis.interpretation: {api_response['motor_cortex_analysis']['interpretation'][:50]}...")

    has_prediction = "prediction" in api_response
    has_gradcam = "grad_cam" in api_response
    has_heatmap_flag = "has_spatial_heatmap" in api_response

    all_passed = has_prediction and has_gradcam and has_heatmap_flag
    print(f"\n  总体结果: {'[OK] 通过' if all_passed else '[FAIL] 失败'}")

    return all_passed


def main():
    """运行所有XAI测试"""
    print("\n" + "=" * 70)
    print("  BCI可解释性成像 (XAI) 测试套件")
    print("  验证 Grad-CAM / 10-20电极映射 / 空间热力图生成")
    print("=" * 70)

    results = {}

    try:
        results["montage"] = test_electrode_montage()
    except Exception as e:
        print(f"\n  电极系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["montage"] = False

    try:
        results["gradcam"] = test_gradcam_1d()
    except Exception as e:
        print(f"\n  Grad-CAM测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["gradcam"] = False

    try:
        results["csp_projector"] = test_csp_projector()
    except Exception as e:
        print(f"\n  CSP投影测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["csp_projector"] = False

    try:
        results["heatmap"] = test_spatial_heatmap_generation()
    except Exception as e:
        print(f"\n  热力图生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["heatmap"] = False

    try:
        results["engine_xai"] = test_engine_xai_integration()
    except Exception as e:
        print(f"\n  引擎XAI集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["engine_xai"] = False

    try:
        results["formatter"] = test_xai_output_formatter()
    except Exception as e:
        print(f"\n  XAI格式化测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["formatter"] = False

    print("\n" + "=" * 70)
    print("  测试结果汇总")
    print("=" * 70)

    test_names = {
        "montage": "10-20电极坐标系统",
        "gradcam": "1D Grad-CAM算法",
        "csp_projector": "CSP反向投影",
        "heatmap": "空间热力图生成",
        "engine_xai": "引擎XAI集成",
        "formatter": "API输出格式化",
    }

    for key, passed in results.items():
        status = "[OK] 通过" if passed else "[FAIL] 失败"
        print(f"  {test_names[key]:<20}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 70)
    print(f"  总体结果: {'全部通过 [OK] - XAI功能完整可用' if all_passed else '存在失败 [FAIL]'}")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

"""
10-20 国际脑电波电极分布系统

提供标准64通道EEG电极的2D/3D坐标，
以及通道名称与物理坐标的映射关系。

关键运动皮层区域（运动想象相关）：
- C3: 左侧运动皮层 (左手运动想象对应右侧皮层)
- C4: 右侧运动皮层 (右手运动想象对应左侧皮层)
- Cz: 中央运动皮层
- FC3, FC4: 额中央区
- CP3, CP4: 中央顶区
"""
from typing import Dict, Tuple, List, Optional
import numpy as np


ELECTRODE_MONTAGE_64 = {
    "Fp1": {"x": -0.30, "y": 0.90, "region": "prefrontal", "hemisphere": "left"},
    "Fp2": {"x": 0.30, "y": 0.90, "region": "prefrontal", "hemisphere": "right"},
    "Fpz": {"x": 0.00, "y": 0.95, "region": "prefrontal", "hemisphere": "midline"},
    "AF7": {"x": -0.45, "y": 0.75, "region": "prefrontal", "hemisphere": "left"},
    "AF3": {"x": -0.25, "y": 0.80, "region": "prefrontal", "hemisphere": "left"},
    "AFz": {"x": 0.00, "y": 0.80, "region": "prefrontal", "hemisphere": "midline"},
    "AF4": {"x": 0.25, "y": 0.80, "region": "prefrontal", "hemisphere": "right"},
    "AF8": {"x": 0.45, "y": 0.75, "region": "prefrontal", "hemisphere": "right"},
    "F9":  {"x": -0.65, "y": 0.55, "region": "frontal", "hemisphere": "left"},
    "F7":  {"x": -0.55, "y": 0.55, "region": "frontal", "hemisphere": "left"},
    "F5":  {"x": -0.40, "y": 0.55, "region": "frontal", "hemisphere": "left"},
    "F3":  {"x": -0.25, "y": 0.55, "region": "frontal", "hemisphere": "left"},
    "F1":  {"x": -0.10, "y": 0.55, "region": "frontal", "hemisphere": "left"},
    "Fz":  {"x": 0.00, "y": 0.55, "region": "frontal", "hemisphere": "midline"},
    "F2":  {"x": 0.10, "y": 0.55, "region": "frontal", "hemisphere": "right"},
    "F4":  {"x": 0.25, "y": 0.55, "region": "frontal", "hemisphere": "right"},
    "F6":  {"x": 0.40, "y": 0.55, "region": "frontal", "hemisphere": "right"},
    "F8":  {"x": 0.55, "y": 0.55, "region": "frontal", "hemisphere": "right"},
    "F10": {"x": 0.65, "y": 0.55, "region": "frontal", "hemisphere": "right"},
    "FT9": {"x": -0.60, "y": 0.35, "region": "frontotemporal", "hemisphere": "left"},
    "FT7": {"x": -0.50, "y": 0.35, "region": "frontotemporal", "hemisphere": "left"},
    "FC5": {"x": -0.35, "y": 0.35, "region": "fronto-central", "hemisphere": "left"},
    "FC3": {"x": -0.20, "y": 0.35, "region": "fronto-central", "hemisphere": "left"},
    "FC1": {"x": -0.05, "y": 0.35, "region": "fronto-central", "hemisphere": "left"},
    "FCz": {"x": 0.00, "y": 0.35, "region": "fronto-central", "hemisphere": "midline"},
    "FC2": {"x": 0.05, "y": 0.35, "region": "fronto-central", "hemisphere": "right"},
    "FC4": {"x": 0.20, "y": 0.35, "region": "fronto-central", "hemisphere": "right"},
    "FC6": {"x": 0.35, "y": 0.35, "region": "fronto-central", "hemisphere": "right"},
    "FC8": {"x": 0.50, "y": 0.35, "region": "frontotemporal", "hemisphere": "right"},
    "FT10": {"x": 0.60, "y": 0.35, "region": "frontotemporal", "hemisphere": "right"},
    "T9":  {"x": -0.70, "y": 0.00, "region": "temporal", "hemisphere": "left"},
    "T7":  {"x": -0.60, "y": 0.00, "region": "temporal", "hemisphere": "left"},
    "C5":  {"x": -0.35, "y": 0.00, "region": "central_motor", "hemisphere": "left"},
    "C3":  {"x": -0.20, "y": 0.00, "region": "central_motor", "hemisphere": "left"},
    "C1":  {"x": -0.05, "y": 0.00, "region": "central_motor", "hemisphere": "left"},
    "Cz":  {"x": 0.00, "y": 0.00, "region": "central_motor", "hemisphere": "midline"},
    "C2":  {"x": 0.05, "y": 0.00, "region": "central_motor", "hemisphere": "right"},
    "C4":  {"x": 0.20, "y": 0.00, "region": "central_motor", "hemisphere": "right"},
    "C6":  {"x": 0.35, "y": 0.00, "region": "central_motor", "hemisphere": "right"},
    "T8":  {"x": 0.60, "y": 0.00, "region": "temporal", "hemisphere": "right"},
    "T10": {"x": 0.70, "y": 0.00, "region": "temporal", "hemisphere": "right"},
    "TP9": {"x": -0.60, "y": -0.35, "region": "temporoparietal", "hemisphere": "left"},
    "TP7": {"x": -0.50, "y": -0.35, "region": "temporoparietal", "hemisphere": "left"},
    "CP5": {"x": -0.35, "y": -0.35, "region": "centro-parietal", "hemisphere": "left"},
    "CP3": {"x": -0.20, "y": -0.35, "region": "centro-parietal", "hemisphere": "left"},
    "CP1": {"x": -0.05, "y": -0.35, "region": "centro-parietal", "hemisphere": "left"},
    "CPz": {"x": 0.00, "y": -0.35, "region": "centro-parietal", "hemisphere": "midline"},
    "CP2": {"x": 0.05, "y": -0.35, "region": "centro-parietal", "hemisphere": "right"},
    "CP4": {"x": 0.20, "y": -0.35, "region": "centro-parietal", "hemisphere": "right"},
    "CP6": {"x": 0.35, "y": -0.35, "region": "centro-parietal", "hemisphere": "right"},
    "TP8": {"x": 0.50, "y": -0.35, "region": "temporoparietal", "hemisphere": "right"},
    "TP10": {"x": 0.60, "y": -0.35, "region": "temporoparietal", "hemisphere": "right"},
    "P9":  {"x": -0.55, "y": -0.55, "region": "parietal", "hemisphere": "left"},
    "P7":  {"x": -0.45, "y": -0.55, "region": "parietal", "hemisphere": "left"},
    "P5":  {"x": -0.30, "y": -0.55, "region": "parietal", "hemisphere": "left"},
    "P3":  {"x": -0.20, "y": -0.55, "region": "parietal", "hemisphere": "left"},
    "P1":  {"x": -0.05, "y": -0.55, "region": "parietal", "hemisphere": "left"},
    "Pz":  {"x": 0.00, "y": -0.55, "region": "parietal", "hemisphere": "midline"},
    "P2":  {"x": 0.05, "y": -0.55, "region": "parietal", "hemisphere": "right"},
    "P4":  {"x": 0.20, "y": -0.55, "region": "parietal", "hemisphere": "right"},
    "P6":  {"x": 0.30, "y": -0.55, "region": "parietal", "hemisphere": "right"},
    "P8":  {"x": 0.45, "y": -0.55, "region": "parietal", "hemisphere": "right"},
    "P10": {"x": 0.55, "y": -0.55, "region": "parietal", "hemisphere": "right"},
    "PO7": {"x": -0.35, "y": -0.75, "region": "parieto-occipital", "hemisphere": "left"},
    "PO3": {"x": -0.15, "y": -0.75, "region": "parieto-occipital", "hemisphere": "left"},
    "POz": {"x": 0.00, "y": -0.75, "region": "parieto-occipital", "hemisphere": "midline"},
    "PO4": {"x": 0.15, "y": -0.75, "region": "parieto-occipital", "hemisphere": "right"},
    "PO8": {"x": 0.35, "y": -0.75, "region": "parieto-occipital", "hemisphere": "right"},
    "O1":  {"x": -0.20, "y": -0.90, "region": "occipital", "hemisphere": "left"},
    "Oz":  {"x": 0.00, "y": -0.95, "region": "occipital", "hemisphere": "midline"},
    "O2":  {"x": 0.20, "y": -0.90, "region": "occipital", "hemisphere": "right"},
    "O9":  {"x": -0.60, "y": -0.70, "region": "occipital", "hemisphere": "left"},
    "O10": {"x": 0.60, "y": -0.70, "region": "occipital", "hemisphere": "right"},
    "Iz":  {"x": 0.00, "y": -1.00, "region": "occipital", "hemisphere": "midline"},
}


STANDARD_64_CHANNEL_ORDER = [
    "Fp1", "Fp2", "Fpz", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F9", "F7", "F5", "F3", "F1", "Fz", "F2", "F4",
    "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FC8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2", "O9", "O10", "Iz",
]


MOTOR_IMAGERY_KEY_CHANNELS = {
    "left_hand": ["C3", "C1", "CP3", "CP1", "FC3", "FC1", "C5", "CP5", "FC5"],
    "right_hand": ["C4", "C2", "CP4", "CP2", "FC4", "FC2", "C6", "CP6", "FC6"],
    "bilateral": ["Cz", "C1", "C2", "C3", "C4"],
}


BRAIN_REGIONS = {
    "prefrontal": ["Fp1", "Fp2", "Fpz", "AF7", "AF3", "AFz", "AF4", "AF8", "F9", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "F10"],
    "frontal": ["FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FC8", "FT10"],
    "central_motor": ["T9", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "T10"],
    "temporal": ["TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10"],
    "parietal": ["P9", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10"],
    "parieto-occipital": ["PO7", "PO3", "POz", "PO4", "PO8"],
    "occipital": ["O1", "Oz", "O2", "O9", "O10", "Iz"],
}


def get_channel_coords_2d(channel_name: str) -> Optional[Tuple[float, float]]:
    """
    获取电极的2D坐标
    
    Args:
        channel_name: 电极名称 (如 "C3", "C4", "Fp1"
        
    Returns:
        (x, y) 坐标，范围在 [-1, 1]，如果电极不存在则返回None
    """
    if channel_name not in ELECTRODE_MONTAGE_64:
        return None
    info = ELECTRODE_MONTAGE_64[channel_name]
    return (info["x"], info["y"])


def get_channel_region(channel_name: str) -> Optional[str]:
    """
    获取电极所属脑区
    
    Args:
        channel_name: 电极名称
        
    Returns:
        脑区名称
    """
    if channel_name not in ELECTRODE_MONTAGE_64:
        return None
    return ELECTRODE_MONTAGE_64[channel_name]["region"]


def get_channel_hemisphere(channel_name: str) -> Optional[str]:
    """
    获取电极所在大脑半球
    
    Args:
        channel_name: 电极名称
        
    Returns:
        半球: 'left' | 'right' | 'midline'
    """
    if channel_name not in ELECTRODE_MONTAGE_64:
        return None
    return ELECTRODE_MONTAGE_64[channel_name]["hemisphere"]


def get_standard_channel_order(num_channels: int = 64) -> List[str]:
    """
    获取标准通道顺序
    
    Args:
        num_channels: 通道数
        
    Returns:
        通道名称列表
    """
    if num_channels <= len(STANDARD_64_CHANNEL_ORDER):
        return STANDARD_64_CHANNEL_ORDER[:num_channels]
    return STANDARD_64_CHANNEL_ORDER


def generate_heatmap_data(
    channel_importance: np.ndarray,
    channel_names: Optional[List[str]] = None,
    num_channels: int = 64,
) -> dict:
    """
    生成前端可渲染的脑电空间热力图数据
    
    Args:
        channel_importance: 通道重要性权重数组
        channel_names: 通道名称列表，如为None则使用标准顺序
        num_channels: 通道数
        
    Returns:
        包含电极坐标、权重、脑区信息的字典
    """
    if channel_names is None:
        channel_names = get_standard_channel_order(num_channels)

    n = min(len(channel_importance), len(channel_names))

    electrodes = []
    for i in range(n):
        ch_name = channel_names[i]
        if ch_name in ELECTRODE_MONTAGE_64:
            coords = get_channel_coords_2d(ch_name)
            if coords is not None:
                electrodes.append({
                    "channel": ch_name,
                    "x": float(coords[0]),
                    "y": float(coords[1]),
                    "importance": float(channel_importance[i]),
                    "region": get_channel_region(ch_name),
                    "hemisphere": get_channel_hemisphere(ch_name),
                })

    max_imp = max(e["importance"] for e in electrodes) if electrodes else 1.0
    for e in electrodes:
        e["normalized_importance"] = float(e["importance"] / max_imp) if max_imp > 1e-10 else 0.0

    electrodes.sort(key=lambda x: x["normalized_importance"], reverse=True)
    top_electrodes = [e for e in electrodes if e["normalized_importance"] > 0.5][:15]

    region_importance = {}
    for region, channels in BRAIN_REGIONS.items():
        region_imps = []
        for e in electrodes:
            if e["channel"] in channels:
                region_imps.append(e["importance"])
        if region_imps:
            region_importance[region] = float(np.mean(region_imps))

    region_max = max(region_importance.values()) if region_importance else 1.0
    normalized_regions = {
        region: float(imp / region_max) if region_max > 1e-10 else 0.0
        for region, imp in region_importance.items()
    }

    return {
        "electrodes": electrodes,
        "top_electrodes": top_electrodes,
        "region_importance": region_importance,
        "normalized_region_importance": normalized_regions,
        "motor_candidate_channels": MOTOR_IMAGERY_KEY_CHANNELS,
    }


def compute_region_contributions(
    channel_importance: np.ndarray,
    channel_names: Optional[List[str]] = None,
) -> dict:
    """
    计算各脑区对分类决策的贡献
    
    Args:
        channel_importance: 通道重要性
        channel_names: 通道名称
        
    Returns:
        各脑区贡献字典
    """
    if channel_names is None:
        channel_names = get_standard_channel_order(len(channel_importance))

    contributions = {}
    for region, region_channels in BRAIN_REGIONS.items():
        region_weights = []
        for i, ch in enumerate(channel_names):
            if ch in region_channels and i < len(channel_importance):
                region_weights.append(channel_importance[i])
        if region_weights:
            contributions[region] = {
                "mean_importance": float(np.mean(region_weights)),
                "max_importance": float(np.max(region_weights)),
                "sum_importance": float(np.sum(region_weights)),
                "n_channels": len(region_weights),
            }

    total = sum(v["sum_importance"] for v in contributions.values())
    if total > 1e-10:
        for region in contributions:
            contributions[region]["percentage"] = float(contributions[region]["sum_importance"] / total * 100)
    else:
        for region in contributions:
            contributions[region]["percentage"] = 0.0

    return contributions


def get_motor_cortex_activity(
    channel_importance: np.ndarray,
    channel_names: Optional[List[str]] = None,
) -> dict:
    """
    获取运动皮层区域的激活分析
    
    Args:
        channel_importance: 通道重要性
        channel_names: 通道名称
        
    Returns:
        运动皮层激活分析
    """
    if channel_names is None:
        channel_names = get_standard_channel_order(len(channel_importance))

    result = {}

    for side, key_channels in MOTOR_IMAGERY_KEY_CHANNELS.items():
        imps = []
        active_chs = []
        for i, ch in enumerate(channel_names):
            if ch in key_channels and i < len(channel_importance):
                imp = channel_importance[i]
                imps.append(imp)
                if imp > 0.5:
                    active_chs.append(ch)
        result[side] = {
            "mean_importance": float(np.mean(imps)) if imps else 0.0,
            "max_importance": float(np.max(imps)) if imps else 0.0,
            "active_channels": active_chs,
        }

    left_imp = result.get("left_hand", {}).get("mean_importance", 0)
    right_imp = result.get("right_hand", {}).get("mean_importance", 0)
    total = left_imp + right_imp
    if total > 1e-10:
        result["laterality_index"] = float((right_imp - left_imp) / total)
    else:
        result["laterality_index"] = 0.0

    return result

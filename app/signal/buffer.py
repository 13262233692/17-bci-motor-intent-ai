import threading
import numpy as np
from collections import deque
from typing import Optional, Tuple


class LatchedSignalBuffer:
    """
    带锁存控制的信号缓冲管理器
    
    核心设计原则：
    1. 严格的锁存机制 - 确保读取时数据完整性，防止跨片段截断污染
    2. 环形缓冲结构 - 高效处理高频数据流
    3. 双缓冲设计 - 写入和读取互不阻塞
    4. 时间戳对齐 - 确保数据片段的时间连续性
    """

    def __init__(
        self,
        num_channels: int = 64,
        sampling_rate: int = 250,
        window_size_seconds: float = 2.0,
        max_buffer_seconds: float = 10.0,
    ):
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.window_size_samples = int(window_size_seconds * sampling_rate)
        self.max_buffer_samples = int(max_buffer_seconds * sampling_rate)

        self._write_lock = threading.Lock()
        self._read_lock = threading.Lock()
        self._latched = False
        self._latch_condition = threading.Condition(self._read_lock)

        self._buffer = np.zeros((num_channels, self.max_buffer_samples), dtype=np.float64)
        self._write_index = 0
        self._total_written = 0
        self._timestamps = deque(maxlen=self.max_buffer_samples)

        self._latched_window: Optional[np.ndarray] = None
        self._latch_start_idx: int = 0
        self._latch_end_idx: int = 0

    def push(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None) -> None:
        """
        向缓冲区写入新的EEG数据块
        
        Args:
            data: 形状为 (num_channels, n_samples) 的EEG数据
            timestamps: 可选，对应采样点的时间戳
        """
        if data.shape[0] != self.num_channels:
            raise ValueError(
                f"通道数不匹配: 期望 {self.num_channels}, 实际 {data.shape[0]}"
            )

        n_samples = data.shape[1]

        with self._write_lock:
            if n_samples >= self.max_buffer_samples:
                start_idx = n_samples - self.max_buffer_samples
                data = data[:, start_idx:]
                n_samples = data.shape[1]
                self._write_index = 0
                self._total_written += n_samples
                self._buffer[:, :n_samples] = data
                if timestamps is not None:
                    self._timestamps.extend(timestamps[start_idx:].tolist())
                return

            remaining = self.max_buffer_samples - self._write_index
            if n_samples <= remaining:
                self._buffer[:, self._write_index:self._write_index + n_samples] = data
                self._write_index += n_samples
            else:
                first_part = remaining
                second_part = n_samples - remaining
                self._buffer[:, self._write_index:] = data[:, :first_part]
                self._buffer[:, :second_part] = data[:, first_part:]
                self._write_index = second_part

            self._total_written += n_samples

            if timestamps is not None:
                self._timestamps.extend(timestamps.tolist())

    def is_ready(self) -> bool:
        """检查是否有足够的数据用于一个完整的窗口"""
        return self._total_written >= self.window_size_samples

    def latch_window(self) -> bool:
        """
        锁存一个完整的数据窗口
        
        只有当缓冲区中有足够的完整数据时才会锁存成功。
        锁存后的数据不会被新写入的数据覆盖，确保读取时的完整性。
        
        Returns:
            bool: 是否成功锁存
        """
        with self._write_lock:
            if not self.is_ready():
                return False

            end_idx = self._write_index
            start_idx = (end_idx - self.window_size_samples) % self.max_buffer_samples

            if start_idx < end_idx:
                window_data = self._buffer[:, start_idx:end_idx].copy()
            else:
                first_part = self._buffer[:, start_idx:].copy()
                second_part = self._buffer[:, :end_idx].copy()
                window_data = np.concatenate([first_part, second_part], axis=1)

        with self._latch_condition:
            self._latched_window = window_data
            self._latched = True
            self._latch_start_idx = start_idx
            self._latch_end_idx = end_idx
            self._latch_condition.notify_all()

        return True

    def get_latched_window(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        获取已锁存的数据窗口
        
        Args:
            timeout: 等待锁存的超时时间（秒），None表示不等待
            
        Returns:
            锁存的数据窗口，形状为 (num_channels, window_size_samples)，
            如果没有可用数据则返回 None
        """
        with self._latch_condition:
            if not self._latched:
                if timeout is None:
                    return None
                self._latch_condition.wait(timeout=timeout)
                if not self._latched:
                    return None

            window_data = self._latched_window.copy()
            return window_data

    def get_window_and_clear(self) -> Optional[np.ndarray]:
        """
        获取锁存窗口并清除锁存状态
        
        Returns:
            锁存的数据窗口，如果没有则返回 None
        """
        with self._latch_condition:
            if not self._latched or self._latched_window is None:
                return None
            window_data = self._latched_window.copy()
            self._latched_window = None
            self._latched = False
            return window_data

    def await_window(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        等待并获取一个新的数据窗口
        
        此方法会清除当前锁存并等待下一个窗口。
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            新的数据窗口
        """
        with self._latch_condition:
            self._latched = False
            self._latched_window = None

        with self._write_lock:
            if not self.is_ready():
                return None

            end_idx = self._write_index
            start_idx = (end_idx - self.window_size_samples) % self.max_buffer_samples

            if start_idx < end_idx:
                window_data = self._buffer[:, start_idx:end_idx].copy()
            else:
                first_part = self._buffer[:, start_idx:].copy()
                second_part = self._buffer[:, :end_idx].copy()
                window_data = np.concatenate([first_part, second_part], axis=1)

        return window_data

    def clear(self) -> None:
        """清空缓冲区"""
        with self._write_lock:
            self._buffer.fill(0)
            self._write_index = 0
            self._total_written = 0
            self._timestamps.clear()

        with self._latch_condition:
            self._latched_window = None
            self._latched = False

    @property
    def current_size(self) -> int:
        """当前缓冲区中的样本数"""
        return min(self._total_written, self.max_buffer_samples)

    @property
    def is_latched(self) -> bool:
        """是否有已锁存的数据"""
        return self._latched

    def get_stats(self) -> dict:
        """获取缓冲区统计信息"""
        return {
            "num_channels": self.num_channels,
            "sampling_rate": self.sampling_rate,
            "window_size_samples": self.window_size_samples,
            "max_buffer_samples": self.max_buffer_samples,
            "current_samples": self.current_size,
            "total_written": self._total_written,
            "is_ready": self.is_ready(),
            "is_latched": self._latched,
        }

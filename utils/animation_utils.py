import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy import signal as scipy_signal
import uuid
import os
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger("animation_utils")
plt.switch_backend('Agg')

class AnimationGenerator:
    def __init__(self, artifact_dir: str = "artifacts", fps: int = 10):
        self.artifact_dir = artifact_dir
        self.fps = fps
        os.makedirs(self.artifact_dir, exist_ok=True)
    
    def create_fourier_animation(self, signal: np.ndarray, sr: float, data_type: str) -> str:
        try:
            if len(signal) < 100:
                logger.warning("Signal too short for Fourier animation")
                return ""
                
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            time_axis = np.arange(len(signal)) / sr
            line1, = ax1.plot([], [], 'b-', linewidth=2, alpha=0.8)
            ax1.set_xlim(0, time_axis[-1])
            ax1.set_ylim(np.min(signal), np.max(signal))
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.set_title('Time Domain Signal')
            ax1.grid(True, alpha=0.3)
            
            full_fft = fft(signal)
            full_freqs = fftfreq(len(signal), 1/sr)
            pos_mask = full_freqs > 0
            full_mags = np.abs(full_fft[pos_mask]) / len(signal)
            full_freqs_pos = full_freqs[pos_mask]
            
            line2, = ax2.plot([], [], 'r-', linewidth=2, alpha=0.8)
            ax2.set_xlim(0, max(full_freqs_pos) if len(full_freqs_pos) > 0 else 1000)
            ax2.set_ylim(0, max(full_mags) * 1.1 if len(full_mags) > 0 else 1)
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Magnitude')
            ax2.set_title('Fourier Transform Progress')
            ax2.grid(True, alpha=0.3)
            
            progress_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, fontsize=12,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            def animate(frame):
                progress = min((frame + 1) / len(signal), 1.0)
                current_samples = int(progress * len(signal))
                line1.set_data(time_axis[:current_samples], signal[:current_samples])
                
                if current_samples > 10:
                    partial_fft = fft(signal[:current_samples])
                    partial_freqs = fftfreq(current_samples, 1/sr)
                    pos_mask = partial_freqs > 0
                    partial_mags = np.abs(partial_fft[pos_mask]) / current_samples
                    partial_freqs_pos = partial_freqs[pos_mask]
                    
                    if len(partial_freqs_pos) > 0 and len(partial_mags) > 0:
                        line2.set_data(partial_freqs_pos, partial_mags)
                
                progress_text.set_text(f'FFT Progress: {progress*100:.1f}%')
                return line1, line2, progress_text
            
            frames = min(100, len(signal))
            anim = FuncAnimation(fig, animate, frames=frames, interval=50, blit=True, repeat=False)
            
            fname = f"{self.artifact_dir}/fourier_animation_{uuid.uuid4().hex[:8]}.gif"
            anim.save(fname, writer='pillow', fps=self.fps)
            plt.close()
            
            logger.info(f"Fourier animation saved: {fname}")
            return fname
            
        except Exception as e:
            logger.error(f"Fourier animation creation failed: {e}")
            return ""
    
    def create_signal_analysis_animation(self, signal: np.ndarray, sr: float, 
                                       zscore_threshold: float, data_type: str) -> str:
        try:
            if len(signal) < 100:
                logger.warning("Signal too short for analysis animation")
                return ""
                
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
            
            time_axis = np.arange(len(signal)) / sr
            
            line_signal, = ax1.plot([], [], 'b-', linewidth=1, alpha=0.7, label='Signal')
            line_mean, = ax1.plot([], [], 'r--', linewidth=2, alpha=0.8, label='Moving Mean')
            line_std_upper, = ax1.plot([], [], 'g--', linewidth=1, alpha=0.6, label='±2σ')
            line_std_lower, = ax1.plot([], [], 'g--', linewidth=1, alpha=0.6)
            ax1.set_xlim(0, time_axis[-1])
            ax1.set_ylim(np.min(signal), np.max(signal))
            ax1.set_ylabel('Amplitude')
            ax1.set_title('Real-time Signal Analysis')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            line_zscore, = ax2.plot([], [], 'orange', linewidth=2, alpha=0.8, label='Z-score')
            line_threshold, = ax2.plot([], [], 'r--', linewidth=1, alpha=0.8, label=f'Threshold (±{zscore_threshold})')
            ax2.set_xlim(0, time_axis[-1])
            ax2.set_ylim(0, 10)
            ax2.set_ylabel('Z-score')
            ax2.set_title('Statistical Outlier Detection')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            feature_names = ['RMS', 'Kurtosis', 'Crest Factor']
            feature_lines = []
            colors = ['blue', 'red', 'green']
            for i, (name, color) in enumerate(zip(feature_names, colors)):
                line, = ax3.plot([], [], color=color, linewidth=2, alpha=0.8, label=name)
                feature_lines.append(line)
            ax3.set_xlim(0, time_axis[-1])
            ax3.set_ylim(0, 10)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Feature Value')
            ax3.set_title('Feature Evolution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            def animate(frame):
                progress = min((frame + 1) / len(signal), 1.0)
                current_samples = int(progress * len(signal))
                current_time = time_axis[:current_samples]
                current_signal = signal[:current_samples]
                
                line_signal.set_data(current_time, current_signal)
                
                if current_samples > 10:
                    window_size = min(50, current_samples)
                    moving_mean = pd.Series(current_signal).rolling(window=window_size, center=True).mean()
                    moving_std = pd.Series(current_signal).rolling(window=window_size, center=True).std()
                    
                    line_mean.set_data(current_time, moving_mean)
                    line_std_upper.set_data(current_time, moving_mean + 2 * moving_std)
                    line_std_lower.set_data(current_time, moving_mean - 2 * moving_std)
                    
                    z_scores = np.abs((current_signal - np.mean(current_signal)) / (np.std(current_signal) + 1e-8))
                    line_zscore.set_data(current_time, z_scores)
                    line_threshold.set_data([current_time[0], current_time[-1]], 
                                          [zscore_threshold] * 2)
                    
                    if current_samples >= 100:
                        feature_values = []
                        rms = np.sqrt(np.mean(current_signal**2))
                        feature_values.append(rms * 10)
                        if np.std(current_signal) > 1e-8:
                            kurt = self._compute_kurtosis(current_signal)
                            feature_values.append(min(kurt, 10))
                        else:
                            feature_values.append(0)
                        if rms > 1e-8:
                            crest = np.max(np.abs(current_signal)) / rms
                            feature_values.append(min(crest, 10))
                        else:
                            feature_values.append(0)
                        
                        for i, line in enumerate(feature_lines):
                            line.set_data(current_time, [feature_values[i]] * len(current_time))
                
                return [line_signal, line_mean, line_std_upper, line_std_lower, 
                       line_zscore, line_threshold] + feature_lines
            
            frames = min(80, len(signal))
            anim = FuncAnimation(fig, animate, frames=frames, interval=60, blit=True, repeat=False)
            
            fname = f"{self.artifact_dir}/signal_analysis_animation_{uuid.uuid4().hex[:8]}.gif"
            anim.save(fname, writer='pillow', fps=self.fps)
            plt.close()
            
            logger.info(f"Signal analysis animation saved: {fname}")
            return fname
            
        except Exception as e:
            logger.error(f"Signal analysis animation creation failed: {e}")
            return ""
    
    def create_anomaly_detection_animation(self, signal: np.ndarray, sr: float, 
                                         zscore_threshold: float, data_type: str) -> str:
        try:
            if len(signal) < 100:
                logger.warning("Signal too short for anomaly detection animation")
                return ""
                
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            time_axis = np.arange(len(signal)) / sr
            
            line_signal, = ax1.plot([], [], 'b-', linewidth=1, alpha=0.7, label='Signal')
            scatter_anomalies = ax1.scatter([], [], c='red', s=50, alpha=0.8, label='Detected Anomalies', zorder=5)
            ax1.set_xlim(0, time_axis[-1])
            ax1.set_ylim(np.min(signal), np.max(signal))
            ax1.set_ylabel('Amplitude')
            ax1.set_title('Real-time Anomaly Detection')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            line_score, = ax2.plot([], [], 'purple', linewidth=2, alpha=0.8, label='Anomaly Score')
            line_threshold, = ax2.plot([], [], 'r--', linewidth=1, alpha=0.8, label='Detection Threshold')
            ax2.set_xlim(0, time_axis[-1])
            ax2.set_ylim(0, 1)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Anomaly Score')
            ax2.set_title('Anomaly Confidence Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            counter_text = ax1.text(0.02, 0.95, 'Anomalies Detected: 0', transform=ax1.transAxes, 
                                  fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            def animate(frame):
                progress = min((frame + 1) / len(signal), 1.0)
                current_samples = int(progress * len(signal))
                current_time = time_axis[:current_samples]
                current_signal = signal[:current_samples]
                
                line_signal.set_data(current_time, current_signal)
                
                detected_anomaly_times = []
                anomaly_scores = []
                
                if current_samples > 50:
                    z_scores = np.abs((current_signal - np.mean(current_signal)) / (np.std(current_signal) + 1e-8))
                    anomaly_mask = z_scores > zscore_threshold
                    anomaly_indices = np.where(anomaly_mask)[0]
                    
                    for idx in anomaly_indices:
                        if idx < len(current_time):
                            detected_anomaly_times.append(current_time[idx])
                            anomaly_scores.append(min(z_scores[idx] / 10.0, 1.0))
                    
                    if detected_anomaly_times:
                        anomaly_values = [current_signal[i] for i in anomaly_indices if i < len(current_signal)]
                        scatter_anomalies.set_offsets(np.column_stack([detected_anomaly_times, anomaly_values]))
                    else:
                        scatter_anomalies.set_offsets(np.empty((0, 2)))
                    
                    if len(z_scores) >= window_size:
                        smoothed_scores = pd.Series(z_scores).rolling(window=window_size).mean() / 10.0
                        line_score.set_data(current_time, np.clip(smoothed_scores, 0, 1))
                        line_threshold.set_data([current_time[0], current_time[-1]], [0.3, 0.3])
                
                counter_text.set_text(f'Anomalies Detected: {len(detected_anomaly_times)}')
                return [line_signal, scatter_anomalies, line_score, line_threshold, counter_text]
            
            frames = min(100, len(signal))
            anim = FuncAnimation(fig, animate, frames=frames, interval=50, blit=True, repeat=False)
            
            fname = f"{self.artifact_dir}/anomaly_detection_animation_{uuid.uuid4().hex[:8]}.gif"
            anim.save(fname, writer='pillow', fps=self.fps)
            plt.close()
            
            logger.info(f"Anomaly detection animation saved: {fname}")
            return fname
            
        except Exception as e:
            logger.error(f"Anomaly detection animation creation failed: {e}")
            return ""
    
    def create_feature_evolution_animation(self, signal: np.ndarray, sr: float, data_type: str) -> str:
        try:
            if len(signal) < 100:
                logger.warning("Signal too short for feature evolution animation")
                return ""
                
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = axs.flatten()
            
            time_axis = np.arange(len(signal)) / sr
            feature_names = ['RMS', 'Variance', 'Kurtosis', 'Crest Factor']
            colors = ['blue', 'green', 'red', 'purple']
            
            lines = []
            for i, (ax, name, color) in enumerate(zip(axs, feature_names, colors)):
                line, = ax.plot([], [], color=color, linewidth=2, alpha=0.8)
                ax.set_xlim(0, time_axis[-1])
                ax.set_ylim(0, 10)
                ax.set_title(f'{name} Evolution')
                ax.grid(True, alpha=0.3)
                lines.append(line)
            
            plt.tight_layout()
            
            def animate(frame):
                progress = min((frame + 1) / len(signal), 1.0)
                current_samples = int(progress * len(signal))
                current_time = time_axis[:current_samples]
                current_signal = signal[:current_samples]
                
                if current_samples >= 100:
                    feature_values = []
                    rms = np.sqrt(np.mean(current_signal**2))
                    feature_values.append(rms * 50)
                    variance = np.var(current_signal)
                    feature_values.append(variance * 100)
                    if np.std(current_signal) > 1e-8:
                        kurtosis = self._compute_kurtosis(current_signal)
                        feature_values.append(min(kurtosis, 10))
                    else:
                        feature_values.append(0)
                    if rms > 1e-8:
                        crest = np.max(np.abs(current_signal)) / rms
                        feature_values.append(min(crest, 10))
                    else:
                        feature_values.append(0)
                    
                    for i, line in enumerate(lines):
                        line.set_data(current_time, [feature_values[i]] * len(current_time))
                return lines
            
            frames = min(80, len(signal))
            anim = FuncAnimation(fig, animate, frames=frames, interval=60, blit=True, repeat=False)
            
            fname = f"{self.artifact_dir}/feature_evolution_animation_{uuid.uuid4().hex[:8]}.gif"
            anim.save(fname, writer='pillow', fps=self.fps)
            plt.close()
            
            logger.info(f"Feature evolution animation saved: {fname}")
            return fname
            
        except Exception as e:
            logger.error(f"Feature evolution animation creation failed: {e}")
            return ""
    
    def create_comprehensive_analysis_animation(self, signal: np.ndarray, sr: float, 
                                                anomalies: List[Dict[str, Any]], 
                                                data_type: str) -> str:
        try:
            if len(signal) < 100:
                logger.warning("Signal too short for comprehensive animation")
                return ""
                
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            time_axis = np.arange(len(signal)) / sr
            
            line_signal, = ax1.plot([], [], 'g-', linewidth=1, alpha=0.7, label='Signal')
            scatter_anomalies = ax1.scatter([], [], c='red', s=40, alpha=0.8, 
                                          label='Detected Anomalies', zorder=5)
            ax1.set_xlim(0, time_axis[-1])
            ax1.set_ylim(np.min(signal), np.max(signal))
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.set_title('Real-time Signal Analysis with Anomaly Detection')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            line_score, = ax2.plot([], [], 'purple', linewidth=2, alpha=0.8, label='Anomaly Score')
            line_threshold, = ax2.plot([], [], 'r--', linewidth=1, alpha=0.8, label='Threshold')
            ax2.set_xlim(0, time_axis[-1])
            ax2.set_ylim(0, 1)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Anomaly Confidence')
            ax2.set_title('Anomaly Detection Confidence')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            feature_names = ['RMS', 'Kurtosis', 'Crest Factor']
            feature_lines = []
            colors = ['blue', 'red', 'green']
            for i, (name, color) in enumerate(zip(feature_names, colors)):
                line, = ax3.plot([], [], color=color, linewidth=2, alpha=0.8, label=name)
                feature_lines.append(line)
            ax3.set_xlim(0, time_axis[-1])
            ax3.set_ylim(0, 10)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Feature Value')
            ax3.set_title('Feature Evolution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            line_freq, = ax4.plot([], [], 'orange', linewidth=2, alpha=0.8, label='Spectrum')
            ax4.set_xlim(0, sr/2)
            ax4.set_ylim(0, 1)
            ax4.set_xlabel('Frequency (Hz)')
            ax4.set_ylabel('Magnitude')
            ax4.set_title('Frequency Spectrum Evolution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            progress_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=10,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            stats_text = ax1.text(0.02, 0.85, '', transform=ax1.transAxes, fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            def animate(frame):
                progress = min((frame + 1) / len(signal), 1.0)
                current_samples = int(progress * len(signal))
                current_time = time_axis[:current_samples]
                current_signal = signal[:current_samples]
                
                line_signal.set_data(current_time, current_signal)
                
                detected_anomaly_times = []
                anomaly_values = []
                anomaly_scores = []
                
                if current_samples > 50:
                    z_scores = np.abs((current_signal - np.mean(current_signal)) / (np.std(current_signal) + 1e-8))
                    anomaly_mask = z_scores > 3.0
                    anomaly_indices = np.where(anomaly_mask)[0]
                    
                    for idx in anomaly_indices:
                        if idx < len(current_time):
                            detected_anomaly_times.append(current_time[idx])
                            anomaly_values.append(current_signal[idx])
                            anomaly_scores.append(min(z_scores[idx] / 10.0, 1.0))
                    
                    if detected_anomaly_times:
                        scatter_anomalies.set_offsets(np.column_stack([detected_anomaly_times, anomaly_values]))
                    else:
                        scatter_anomalies.set_offsets(np.empty((0, 2)))
                    
                    if len(z_scores) >= 20:
                        smoothed_scores = pd.Series(z_scores).rolling(window=20).mean() / 10.0
                        line_score.set_data(current_time, np.clip(smoothed_scores, 0, 1))
                        line_threshold.set_data([current_time[0], current_time[-1]], [0.3, 0.3])
                
                if current_samples >= 100:
                    feature_values = []
                    rms = np.sqrt(np.mean(current_signal**2))
                    feature_values.append(rms * 10)
                    if np.std(current_signal) > 1e-8:
                        kurt = self._compute_kurtosis(current_signal)
                        feature_values.append(min(kurt, 10))
                    else:
                        feature_values.append(0)
                    if rms > 1e-8:
                        crest = np.max(np.abs(current_signal)) / rms
                        feature_values.append(min(crest, 10))
                    else:
                        feature_values.append(0)
                    
                    for i, line in enumerate(feature_lines):
                        line.set_data(current_time, [feature_values[i]] * len(current_time))
                
                if current_samples > 10:
                    fft_result = fft(current_signal)
                    freqs = fftfreq(current_samples, 1/sr)
                    pos_mask = freqs > 0
                    mags = np.abs(fft_result[pos_mask]) / current_samples
                    freqs_pos = freqs[pos_mask]
                    
                    if len(freqs_pos) > 0 and len(mags) > 0:
                        line_freq.set_data(freqs_pos, mags)
                        ax4.set_ylim(0, max(mags) * 1.1 if max(mags) > 0 else 1)
                
                progress_text.set_text(f'Progress: {progress*100:.1f}%')
                stats_text.set_text(f'Samples: {current_samples}\nAnomalies: {len(detected_anomaly_times)}')
                
                return [line_signal, scatter_anomalies, line_score, line_threshold] + feature_lines + [line_freq, progress_text, stats_text]
            
            frames = min(80, len(signal))
            anim = FuncAnimation(fig, animate, frames=frames, interval=60, blit=True, repeat=False)
            
            fname = f"{self.artifact_dir}/comprehensive_analysis_{uuid.uuid4().hex[:8]}.gif"
            anim.save(fname, writer='pillow', fps=self.fps)
            plt.close()
            
            logger.info(f"Comprehensive analysis animation saved: {fname}")
            return fname
            
        except Exception as e:
            logger.error(f"Comprehensive animation creation failed: {e}")
            return ""
    
    def _compute_kurtosis(self, sig: np.ndarray) -> float:
        if len(sig) < 4 or np.std(sig) < 1e-8:
            return 0.0
        m = np.mean(sig)
        return float(np.mean((sig - m) ** 4) / (np.std(sig) ** 4))
    
    def _compute_crest_factor(self, signal: np.ndarray) -> float:
        rms = np.sqrt(np.mean(signal ** 2))
        return float(np.max(np.abs(signal)) / rms) if rms > 1e-8 else 0.0
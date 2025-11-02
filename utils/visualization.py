import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import uuid
import os
import logging
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal as scipy_signal

logger = logging.getLogger("visualization")
plt.switch_backend('Agg')

def plot_trend_with_anomalies(signal: np.ndarray, 
                            sample_rate: float,
                            anomalies: List[Dict[str, Any]],
                            title: str = "Signal Trend with Anomaly Highlights",
                            figsize: Tuple[int, int] = (14, 8),
                            save_path: str = None) -> str:
    try:
        plt.figure(figsize=figsize)
        time_axis = np.arange(len(signal)) / sample_rate
        
        plt.plot(time_axis, signal, 'b-', alpha=0.7, linewidth=1, label='Signal')
        
        anomaly_regions_plotted = False
        
        for i, anomaly in enumerate(anomalies):
            if 'start_time' in anomaly and 'end_time' in anomaly:
                start_time = anomaly['start_time']
                end_time = anomaly['end_time']
                severity = anomaly.get('severity', 0.5)
                
                start_idx = int(start_time * sample_rate)
                end_idx = min(int(end_time * sample_rate), len(signal))
                
                if start_idx < len(signal) and end_idx > start_idx:
                    alpha = 0.2 + 0.3 * severity
                    color = 'red'
                    
                    plt.fill_between(time_axis[start_idx:end_idx], 
                                   np.min(signal) if len(signal) > 0 else 0,
                                   np.max(signal) if len(signal) > 0 else 1,
                                   alpha=alpha, color=color, 
                                   label='Anomaly Region' if not anomaly_regions_plotted else "")
                    anomaly_regions_plotted = True
        
        for i, anomaly in enumerate(anomalies):
            if 'start_time' in anomaly and ('end_time' not in anomaly or 
                                          anomaly['end_time'] - anomaly['start_time'] < 0.1):
                plt.axvline(x=anomaly['start_time'], color='red', linestyle='--', 
                           alpha=0.8, linewidth=2, label='Point Anomaly' if i == 0 else "")
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path is None:
            save_path = f"artifacts/trend_anomalies_{uuid.uuid4().hex[:8]}.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Trend plot saved to: {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"Trend plot creation failed: {e}")
        return f"Error creating trend plot: {str(e)}"

def plot_anomalies_with_highlights(signal: np.ndarray, 
                                 sample_rate: float,
                                 anomalies: List[Dict[str, Any]],
                                 title: str = "Signal with Anomaly Highlights",
                                 figsize: Tuple[int, int] = (14, 6),
                                 save_path: str = None) -> str:
    try:
        plt.figure(figsize=figsize)
        time_axis = np.arange(len(signal)) / sample_rate
        
        plt.plot(time_axis, signal, 'g-', alpha=0.3, linewidth=1, label='Normal Signal')
        
        anomaly_mask = np.zeros(len(signal), dtype=bool)
        anomaly_indices = []
        
        for anomaly in anomalies:
            if 'start_time' in anomaly and 'end_time' in anomaly:
                start_idx = int(anomaly['start_time'] * sample_rate)
                end_idx = min(int(anomaly['end_time'] * sample_rate), len(signal))
                
                if start_idx < len(signal) and end_idx > start_idx:
                    anomaly_mask[start_idx:end_idx] = True
                    anomaly_indices.extend(range(start_idx, min(end_idx, len(signal))))
            
            elif 'index' in anomaly:
                idx = anomaly['index']
                if idx < len(signal):
                    anomaly_mask[idx] = True
                    anomaly_indices.append(idx)
        
        if anomaly_indices:
            anomaly_times = [time_axis[i] for i in anomaly_indices if i < len(time_axis)]
            anomaly_values = [signal[i] for i in anomaly_indices if i < len(signal)]
            
            plt.scatter(anomaly_times, anomaly_values, color='red', s=50, 
                       alpha=0.8, label=f'Anomalies ({len(anomaly_times)} detected)', 
                       zorder=5, edgecolors='darkred', linewidth=1)
        
        for i, anomaly in enumerate(anomalies):
            severity = anomaly.get('severity', 0.5)
            if 'start_time' in anomaly:
                time_pos = anomaly['start_time']
                if 'index' in anomaly and anomaly['index'] < len(signal):
                    value_pos = signal[anomaly['index']]
                else:
                    idx = int(time_pos * sample_rate)
                    if idx < len(signal):
                        value_pos = signal[idx]
                    else:
                        continue
                
                plt.annotate(f'S:{severity:.2f}', 
                           xy=(time_pos, value_pos),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                           fontsize=8, color='white')
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title(f'{title} - {len(anomaly_indices)} Anomalies Detected')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path is None:
            save_path = f"artifacts/anomaly_highlights_{uuid.uuid4().hex[:8]}.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Anomaly highlights plot saved to: {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"Anomaly highlights plot creation failed: {e}")
        return f"Error creating anomaly highlights: {str(e)}"

def create_anomaly_heatmap(signal: np.ndarray,
                         anomalies: List[Dict[str, Any]],
                         sample_rate: float,
                         title: str = "Anomaly Heatmap",
                         figsize: Tuple[int, int] = (12, 6),
                         save_path: str = None) -> str:
    try:
        time_points = len(signal)
        
        anomaly_density = np.zeros(time_points)
        anomaly_severity = np.zeros(time_points)
        
        for anomaly in anomalies:
            if 'start_time' in anomaly and 'end_time' in anomaly:
                start_idx = int(anomaly['start_time'] * sample_rate)
                end_idx = min(int(anomaly['end_time'] * sample_rate), time_points)
                severity = anomaly.get('severity', 0.5)
                
                if start_idx < time_points and end_idx > start_idx:
                    anomaly_density[start_idx:end_idx] += 1
                    anomaly_severity[start_idx:end_idx] = np.maximum(
                        anomaly_severity[start_idx:end_idx], severity
                    )
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        cmap_density = LinearSegmentedColormap.from_list('density_cmap', ['blue', 'yellow', 'red'])
        im1 = ax1.imshow(anomaly_density.reshape(1, -1), aspect='auto', cmap=cmap_density,
                       extent=[0, time_points/sample_rate, 0, 1])
        ax1.set_title(f'{title} - Density')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Density')
        plt.colorbar(im1, ax=ax1, label='Anomaly Density')
        
        cmap_severity = LinearSegmentedColormap.from_list('severity_cmap', ['green', 'yellow', 'red'])
        im2 = ax2.imshow(anomaly_severity.reshape(1, -1), aspect='auto', cmap=cmap_severity,
                       extent=[0, time_points/sample_rate, 0, 1])
        ax2.set_title(f'{title} - Severity')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Severity')
        plt.colorbar(im2, ax=ax2, label='Anomaly Severity')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"artifacts/anomaly_heatmap_{uuid.uuid4().hex[:8]}.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Anomaly heatmap saved to: {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"Anomaly heatmap creation failed: {e}")
        return f"Error creating anomaly heatmap: {str(e)}"

def plot_spectral_analysis(signal: np.ndarray,
                         sample_rate: float,
                         anomalies: List[Dict[str, Any]] = None,
                         title: str = "Spectral Analysis",
                         figsize: Tuple[int, int] = (12, 8),
                         save_path: str = None) -> str:
    try:
        n = len(signal)
        frequencies = np.fft.fftfreq(n, 1/sample_rate)
        fft_result = np.fft.fft(signal)
        magnitudes = np.abs(fft_result) / n
        
        positive_idx = frequencies > 0
        frequencies = frequencies[positive_idx]
        magnitudes = magnitudes[positive_idx]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        ax1.plot(frequencies, magnitudes, 'b-', alpha=0.7, linewidth=1)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude')
        ax1.set_title('Frequency Spectrum')
        ax1.grid(True, alpha=0.3)
        
        if anomalies:
            for anomaly in anomalies:
                if 'frequency' in anomaly:
                    ax1.axvline(x=anomaly['frequency'], color='red', linestyle='--',
                               alpha=0.8, linewidth=2, label='Anomaly Frequency')
            if any('frequency' in anomaly for anomaly in anomalies):
                ax1.legend()
        
        f, Pxx = scipy_signal.welch(signal, sample_rate, nperseg=min(256, len(signal)))
        ax2.semilogy(f, Pxx)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('PSD (V²/Hz)')
        ax2.set_title('Power Spectral Density')
        ax2.grid(True, alpha=0.3)
        
        nperseg = min(256, len(signal))
        f, t, Sxx = scipy_signal.spectrogram(signal, sample_rate, nperseg=nperseg)
        im = ax3.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_xlabel('Time (sec)')
        ax3.set_title('Spectrogram')
        plt.colorbar(im, ax=ax3, label='Power Spectral Density [dB]')
        
        cumulative_power = np.cumsum(magnitudes**2)
        cumulative_power = cumulative_power / cumulative_power[-1]
        ax4.plot(frequencies, cumulative_power, 'g-', linewidth=2)
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Cumulative Power')
        ax4.set_title('Cumulative Spectral Power')
        ax4.grid(True, alpha=0.3)
        
        for percentile in [0.5, 0.9]:
            idx = np.where(cumulative_power >= percentile)[0]
            if len(idx) > 0:
                freq_cutoff = frequencies[idx[0]]
                ax4.axvline(x=freq_cutoff, color='red', linestyle='--', alpha=0.7,
                           label=f'{percentile*100:.0f}% Power: {freq_cutoff:.1f}Hz')
        ax4.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"artifacts/spectral_analysis_{uuid.uuid4().hex[:8]}.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Spectral analysis plot saved to: {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"Spectral analysis plot creation failed: {e}")
        return f"Error creating spectral analysis plot: {str(e)}"

def create_simple_signal_plot(signal: np.ndarray,
                            sample_rate: float,
                            title: str = "Signal Visualization",
                            save_path: str = None) -> str:
    try:
        plt.figure(figsize=(12, 4))
        time_axis = np.arange(len(signal)) / sample_rate
        plt.plot(time_axis, signal, 'b-', linewidth=1)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = f"artifacts/signal_plot_{uuid.uuid4().hex[:8]}.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Signal plot saved to: {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"Simple signal plot creation failed: {e}")
        return ""

def create_distribution_plot(signal: np.ndarray,
                           anomalies: List[Dict[str, Any]] = None,
                           title: str = "Signal Distribution Analysis",
                           figsize: Tuple[int, int] = (12, 8),
                           save_path: str = None) -> str:
    try:
        signal_clean = signal[~np.isnan(signal)]
        if len(signal_clean) == 0:
            return "Error: No valid data for distribution plot"
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        ax1.hist(signal_clean, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(signal_clean)
        x_range = np.linspace(np.min(signal_clean), np.max(signal_clean), 100)
        ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution with KDE')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.boxplot(signal_clean, vert=True)
        ax2.set_ylabel('Value')
        ax2.set_title('Box Plot')
        ax2.grid(True, alpha=0.3)
        
        from scipy.stats import probplot
        probplot(signal_clean, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normal Distribution)')
        ax3.grid(True, alpha=0.3)
        
        sorted_data = np.sort(signal_clean)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax4.plot(sorted_data, cdf, 'b-', linewidth=2)
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution Function')
        ax4.grid(True, alpha=0.3)
        
        if anomalies:
            anomaly_values = []
            for anomaly in anomalies:
                if 'start_time' in anomaly:
                    start_idx = int(anomaly.get('start_time', 0) * 1000)
                    if start_idx < len(signal):
                        anomaly_values.append(signal[start_idx])
            
            if anomaly_values:
                for ax in [ax1, ax4]:
                    for val in anomaly_values:
                        ax.axvline(x=val, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"artifacts/distribution_analysis_{uuid.uuid4().hex[:8]}.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Distribution plot saved to: {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"Distribution plot creation failed: {e}")
        return f"Error creating distribution plot: {str(e)}"

def generate_comprehensive_animation(signal: np.ndarray, 
                                   sample_rate: float, 
                                   anomalies: List[Dict[str, Any]] = None) -> str:
    try:
        from .animation_utils import AnimationGenerator
        
        animator = AnimationGenerator(artifact_dir="artifacts", fps=10)
        
        animation_path = animator.create_comprehensive_analysis_animation(
            signal, sample_rate, anomalies, "vibration"
        )
        
        if animation_path and os.path.exists(animation_path):
            logger.info(f"Comprehensive animation saved: {animation_path}")
            return animation_path
        else:
            return ""
            
    except Exception as e:
        logger.error(f"Comprehensive animation generation failed: {e}")
        return ""

def generate_analysis_visualizations(signal: np.ndarray, 
                                   sample_rate: float, 
                                   anomalies: List[Dict[str, Any]] = None,
                                   analysis_type: str = "comprehensive",
                                   include_animations: bool = True) -> Dict[str, str]:
    artifacts = {}
    
    try:
        signal_plot_path = create_simple_signal_plot(signal, sample_rate, "Vibration Signal")
        if signal_plot_path and not signal_plot_path.startswith("Error"):
            artifacts["signal_plot"] = signal_plot_path
        
        if anomalies and len(anomalies) > 0:
            highlights_path = plot_anomalies_with_highlights(signal, sample_rate, anomalies, 
                                                           "Signal with Anomaly Detection")
            if highlights_path and not highlights_path.startswith("Error"):
                artifacts["anomaly_highlights"] = highlights_path
        
        if anomalies and len(anomalies) > 0:
            trend_plot_path = plot_trend_with_anomalies(signal, sample_rate, anomalies, "Signal with Anomaly Detection")
            if trend_plot_path and not trend_plot_path.startswith("Error"):
                artifacts["anomaly_trend"] = trend_plot_path
            
            heatmap_path = create_anomaly_heatmap(signal, anomalies, sample_rate, "Anomaly Distribution")
            if heatmap_path and not heatmap_path.startswith("Error"):
                artifacts["anomaly_heatmap"] = heatmap_path
        
        if analysis_type == "comprehensive" and len(signal) >= 256:
            spectral_path = plot_spectral_analysis(signal, sample_rate, anomalies, "Spectral Analysis")
            if spectral_path and not spectral_path.startswith("Error"):
                artifacts["spectral_analysis"] = spectral_path
        
        if len(signal) >= 50:
            distribution_path = create_distribution_plot(signal, anomalies, "Signal Distribution Analysis")
            if distribution_path and not distribution_path.startswith("Error"):
                artifacts["distribution_analysis"] = distribution_path
        
        if include_animations and len(signal) >= 100:
            animated_artifact = generate_comprehensive_animation(signal, sample_rate, anomalies)
            if animated_artifact:
                artifacts["comprehensive_animation"] = animated_artifact
        
        logger.info(f"Generated {len(artifacts)} visualization artifacts")
        return artifacts
        
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        return artifacts

def create_plotly_signal_plot(signal: np.ndarray, 
                            sample_rate: float,
                            title: str = "Interactive Signal Plot") -> str:
    try:
        time_axis = np.arange(len(signal)) / sample_rate
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=signal,
            mode='lines',
            name='Signal',
            line=dict(color='blue', width=1)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            hovermode='x unified'
        )
        
        filename = f"artifacts/interactive_plot_{uuid.uuid4().hex[:8]}.html"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.write_html(filename)
        
        return filename
        
    except Exception as e:
        logger.error(f"Plotly plot creation failed: {e}")
        return ""

def save_plot_to_file(fig: plt.Figure, 
                     filename: str = None,
                     dpi: int = 150) -> str:
    try:
        if filename is None:
            filename = f"artifacts/plot_{uuid.uuid4().hex[:8]}.png"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Plot saved to: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Plot saving failed: {e}")
        return f"Error saving plot: {str(e)}"
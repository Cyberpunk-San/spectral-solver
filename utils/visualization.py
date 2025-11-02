# utils/visualization.py
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

# matplotlib backend to avoid GUI issues
plt.switch_backend('Agg')

def plot_trend_with_anomalies(signal: np.ndarray, 
                            sample_rate: float,
                            anomalies: List[Dict[str, Any]],
                            title: str = "Signal Trend with Anomaly Highlights",
                            figsize: Tuple[int, int] = (14, 8),
                            save_path: str = None) -> str:
    """
    Create trend plot with anomalies highlighted in red
    """
    try:
        plt.figure(figsize=figsize)
        time_axis = np.arange(len(signal)) / sample_rate
        
        # main signal
        plt.plot(time_axis, signal, 'b-', alpha=0.7, linewidth=1, label='Signal')
        
        # Highlight anomaly regions in red with different intensities based on severity
        anomaly_regions_plotted = False
        
        for i, anomaly in enumerate(anomalies):
            if 'start_time' in anomaly and 'end_time' in anomaly:
                start_time = anomaly['start_time']
                end_time = anomaly['end_time']
                severity = anomaly.get('severity', 0.5)
                
                start_idx = int(start_time * sample_rate)
                end_idx = min(int(end_time * sample_rate), len(signal))
                
                if start_idx < len(signal) and end_idx > start_idx:
                    # color intensity based on severity
                    alpha = 0.2 + 0.3 * severity  # 0.2 to 0.5 alpha based on severity
                    color = 'red'
                    
                    plt.fill_between(time_axis[start_idx:end_idx], 
                                   np.min(signal) if len(signal) > 0 else 0,
                                   np.max(signal) if len(signal) > 0 else 1,
                                   alpha=alpha, color=color, 
                                   label='Anomaly Region' if not anomaly_regions_plotted else "")
                    anomaly_regions_plotted = True
        
        # vertical lines for point anomalies
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

def create_anomaly_heatmap(signal: np.ndarray,
                         anomalies: List[Dict[str, Any]],
                         sample_rate: float,
                         title: str = "Anomaly Heatmap",
                         figsize: Tuple[int, int] = (12, 6),
                         save_path: str = None) -> str:
    """
    Create heatmap visualization of anomalies with severity mapping
    """
    try:
        time_points = len(signal)
        
        # Create anomaly density map with severity weighting
        anomaly_density = np.zeros(time_points)
        anomaly_severity = np.zeros(time_points)
        
        for anomaly in anomalies:
            if 'start_time' in anomaly and 'end_time' in anomaly:
                start_idx = int(anomaly['start_time'] * sample_rate)
                end_idx = min(int(anomaly['end_time'] * sample_rate), time_points)
                severity = anomaly.get('severity', 0.5)
                
                if start_idx < time_points and end_idx > start_idx:
                    # Add severity-weighted density
                    anomaly_density[start_idx:end_idx] += 1
                    anomaly_severity[start_idx:end_idx] = np.maximum(
                        anomaly_severity[start_idx:end_idx], severity
                    )
        
        # Create the heatmap
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Anomaly density
        cmap_density = LinearSegmentedColormap.from_list('density_cmap', ['blue', 'yellow', 'red'])
        im1 = ax1.imshow(anomaly_density.reshape(1, -1), aspect='auto', cmap=cmap_density,
                       extent=[0, time_points/sample_rate, 0, 1])
        ax1.set_title(f'{title} - Density')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Density')
        plt.colorbar(im1, ax=ax1, label='Anomaly Density')
        
        # Plot 2: Anomaly severity
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
    """
    Create comprehensive spectral analysis plot with anomaly highlights
    """
    try:
        # Compute FFT
        n = len(signal)
        frequencies = np.fft.fftfreq(n, 1/sample_rate)
        fft_result = np.fft.fft(signal)
        magnitudes = np.abs(fft_result) / n
        
        # Only positive frequencies
        positive_idx = frequencies > 0
        frequencies = frequencies[positive_idx]
        magnitudes = magnitudes[positive_idx]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Frequency spectrum
        ax1.plot(frequencies, magnitudes, 'b-', alpha=0.7, linewidth=1)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude')
        ax1.set_title('Frequency Spectrum')
        ax1.grid(True, alpha=0.3)
        
        # Highlight anomaly frequencies
        if anomalies:
            for anomaly in anomalies:
                if 'frequency' in anomaly:
                    ax1.axvline(x=anomaly['frequency'], color='red', linestyle='--',
                               alpha=0.8, linewidth=2, label='Anomaly Frequency')
            if any('frequency' in anomaly for anomaly in anomalies):
                ax1.legend()
        
        # Plot 2: Power Spectral Density
        f, Pxx = scipy_signal.welch(signal, sample_rate, nperseg=min(256, len(signal)))
        ax2.semilogy(f, Pxx)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('PSD (V²/Hz)')
        ax2.set_title('Power Spectral Density')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Spectrogram
        nperseg = min(256, len(signal))
        f, t, Sxx = scipy_signal.spectrogram(signal, sample_rate, nperseg=nperseg)
        im = ax3.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_xlabel('Time (sec)')
        ax3.set_title('Spectrogram')
        plt.colorbar(im, ax=ax3, label='Power Spectral Density [dB]')
        
        # Plot 4: Cumulative spectral power
        cumulative_power = np.cumsum(magnitudes**2)
        cumulative_power = cumulative_power / cumulative_power[-1]  # Normalize
        ax4.plot(frequencies, cumulative_power, 'g-', linewidth=2)
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Cumulative Power')
        ax4.set_title('Cumulative Spectral Power')
        ax4.grid(True, alpha=0.3)
        
        # Add 50% and 90% power lines
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
    """
    Create a simple signal plot - most reliable fallback
    """
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
    """
    Create distribution analysis plot with anomaly indicators
    """
    try:
        signal_clean = signal[~np.isnan(signal)]
        if len(signal_clean) == 0:
            return "Error: No valid data for distribution plot"
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Histogram with KDE
        ax1.hist(signal_clean, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        
        # Add KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(signal_clean)
        x_range = np.linspace(np.min(signal_clean), np.max(signal_clean), 100)
        ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution with KDE')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plot
        ax2.boxplot(signal_clean, vert=True)
        ax2.set_ylabel('Value')
        ax2.set_title('Box Plot')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Q-Q plot
        from scipy.stats import probplot
        probplot(signal_clean, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normal Distribution)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative distribution
        sorted_data = np.sort(signal_clean)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax4.plot(sorted_data, cdf, 'b-', linewidth=2)
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution Function')
        ax4.grid(True, alpha=0.3)
        
        # Highlight anomalies in distribution if provided
        if anomalies:
            anomaly_values = []
            for anomaly in anomalies:
                if 'start_time' in anomaly:
                    # Extract corresponding signal values for anomalies
                    start_idx = int(anomaly.get('start_time', 0) * 1000)  # Assuming 1000Hz
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

def generate_animated_visualizations(signal: np.ndarray, 
                                   sample_rate: float, 
                                   anomalies: List[Dict[str, Any]] = None,
                                   analysis_type: str = "comprehensive") -> Dict[str, str]:
    """
    Generate animated visualizations for analysis results
    """
    artifacts = {}
    
    try:
        from .animation_utils import AnimationGenerator
        
        # Create animation generator
        animator = AnimationGenerator(artifact_dir="artifacts", fps=10)
        
        # Generate different types of animations based on signal characteristics
        if len(signal) >= 100:  # Only create animations for sufficiently long signals
            
            # 1. Fourier transform animation
            try:
                fourier_anim = animator.create_fourier_animation(
                    signal, sample_rate, "vibration"
                )
                if fourier_anim and os.path.exists(fourier_anim):
                    artifacts["fourier_animation"] = fourier_anim
            except Exception as e:
                logger.warning(f"Fourier animation failed: {e}")
            
            # 2. Signal analysis animation
            try:
                signal_anim = animator.create_signal_analysis_animation(
                    signal, sample_rate, zscore_threshold=3.0, data_type="vibration"
                )
                if signal_anim and os.path.exists(signal_anim):
                    artifacts["signal_analysis_animation"] = signal_anim
            except Exception as e:
                logger.warning(f"Signal analysis animation failed: {e}")
            
            # 3. Anomaly detection animation (if we have anomalies)
            if anomalies and len(anomalies) > 0:
                try:
                    anomaly_anim = animator.create_anomaly_detection_animation(
                        signal, sample_rate, zscore_threshold=3.0, data_type="vibration"
                    )
                    if anomaly_anim and os.path.exists(anomaly_anim):
                        artifacts["anomaly_detection_animation"] = anomaly_anim
                except Exception as e:
                    logger.warning(f"Anomaly detection animation failed: {e}")
            
            # 4. Feature evolution animation
            try:
                feature_anim = animator.create_feature_evolution_animation(
                    signal, sample_rate, "vibration"
                )
                if feature_anim and os.path.exists(feature_anim):
                    artifacts["feature_evolution_animation"] = feature_anim
            except Exception as e:
                logger.warning(f"Feature evolution animation failed: {e}")
            
            # 5. Comprehensive dashboard animation
            try:
                dashboard_anim = animator.create_animated_dashboard(
                    signal, sample_rate, "vibration"
                )
                if dashboard_anim and os.path.exists(dashboard_anim):
                    artifacts["animated_dashboard"] = dashboard_anim
            except Exception as e:
                logger.warning(f"Dashboard animation failed: {e}")
        
        logger.info(f"Generated {len(artifacts)} animated visualizations")
        return artifacts
        
    except Exception as e:
        logger.error(f"Animated visualization generation failed: {e}")
        return artifacts

def generate_analysis_visualizations(signal: np.ndarray, 
                                   sample_rate: float, 
                                   anomalies: List[Dict[str, Any]] = None,
                                   analysis_type: str = "comprehensive",
                                   include_animations: bool = True) -> Dict[str, str]:
    """
    Main function to generate all visualizations for analysis results
    """
    artifacts = {}
    
    try:
        # Always create basic signal plot
        signal_plot_path = create_simple_signal_plot(signal, sample_rate, "Vibration Signal")
        if signal_plot_path and not signal_plot_path.startswith("Error"):
            artifacts["signal_plot"] = signal_plot_path
        
        # Create trend plot with anomalies if we have anomalies
        if anomalies and len(anomalies) > 0:
            trend_plot_path = plot_trend_with_anomalies(signal, sample_rate, anomalies, "Signal with Anomaly Detection")
            if trend_plot_path and not trend_plot_path.startswith("Error"):
                artifacts["anomaly_trend"] = trend_plot_path
            
            heatmap_path = create_anomaly_heatmap(signal, anomalies, sample_rate, "Anomaly Distribution")
            if heatmap_path and not heatmap_path.startswith("Error"):
                artifacts["anomaly_heatmap"] = heatmap_path
        
        # Create spectral analysis for comprehensive analysis
        if analysis_type == "comprehensive" and len(signal) >= 256:
            spectral_path = plot_spectral_analysis(signal, sample_rate, anomalies, "Spectral Analysis")
            if spectral_path and not spectral_path.startswith("Error"):
                artifacts["spectral_analysis"] = spectral_path
        
        # Create distribution plot
        if len(signal) >= 50:
            distribution_path = create_distribution_plot(signal, anomalies, "Signal Distribution Analysis")
            if distribution_path and not distribution_path.startswith("Error"):
                artifacts["distribution_analysis"] = distribution_path
        
        # Generate animations if requested and signal is long enough
        if include_animations and len(signal) >= 100:
            animated_artifacts = generate_animated_visualizations(
                signal, sample_rate, anomalies, analysis_type
            )
            artifacts.update(animated_artifacts)
        
        logger.info(f"Generated {len(artifacts)} visualization artifacts")
        return artifacts
        
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        # Return whatever artifacts we managed to create
        return artifacts

def create_plotly_signal_plot(signal: np.ndarray, 
                            sample_rate: float,
                            title: str = "Interactive Signal Plot") -> str:
    """
    Create interactive Plotly plot for web display
    """
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
        
        # Save as HTML
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
    """
    Utility function to save matplotlib figure to file
    """
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
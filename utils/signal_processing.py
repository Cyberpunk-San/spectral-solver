# utils/signal_processing.py
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from scipy.stats import zscore, kurtosis, skew
from typing import Dict, List, Any, Optional, Tuple, Union
import pywt
import logging

logger = logging.getLogger("signal_processing")

def compute_zscore_anomalies(signal: np.ndarray, 
                           threshold: float = 3.0, 
                           sample_rate: float = 1000.0,
                           window_size: int = None) -> List[Dict[str, Any]]:
    """
    Detect anomalies using Z-score method with adaptive thresholding
    
    Args:
        signal: Input signal array
        threshold: Z-score threshold for anomaly detection
        sample_rate: Sampling rate in Hz
        window_size: Rolling window size for adaptive thresholding
    
    Returns:
        List of anomaly dictionaries
    """
    anomalies = []
    
    if len(signal) < 10:
        return anomalies
    
    try:
        # Handle NaN values
        signal_clean = signal[~np.isnan(signal)]
        if len(signal_clean) < 10:
            return anomalies
        
        if window_size and window_size < len(signal_clean):
            # Adaptive z-score using rolling window
            series = pd.Series(signal_clean)
            rolling_mean = series.rolling(window=window_size, center=True, min_periods=1).mean()
            rolling_std = series.rolling(window=window_size, center=True, min_periods=1).std()
            
            # Avoid division by zero
            rolling_std = rolling_std.replace(0, np.finfo(float).eps)
            z_scores = np.abs((series - rolling_mean) / rolling_std)
        else:
            # Global z-score
            z_scores = np.abs(zscore(signal_clean))
        
        anomaly_indices = np.where(z_scores > threshold)[0]
        
        # Group consecutive anomalies
        if len(anomaly_indices) > 0:
            groups = []
            current_group = [anomaly_indices[0]]
            
            for i in range(1, len(anomaly_indices)):
                if anomaly_indices[i] - anomaly_indices[i-1] <= 5:  # Consecutive threshold
                    current_group.append(anomaly_indices[i])
                else:
                    groups.append(current_group)
                    current_group = [anomaly_indices[i]]
            
            groups.append(current_group)
            
            for group in groups:
                if group:
                    start_idx = group[0]
                    end_idx = group[-1]
                    max_z = np.max(z_scores[group])
                    
                    anomalies.append({
                        'type': 'zscore_anomaly',
                        'start_time': float(start_idx / sample_rate),
                        'end_time': float((end_idx + 1) / sample_rate),
                        'severity': float(min(max_z / (threshold * 2), 1.0)),
                        'max_zscore': float(max_z),
                        'description': f'Z-score anomaly (z={max_z:.2f})',
                        'confidence': min(0.3 + (max_z - threshold) / 10, 0.9),
                        'method': 'zscore',
                        'indices': group.tolist() if hasattr(group, 'tolist') else group
                    })
        
    except Exception as e:
        logger.error(f"Z-score anomaly detection failed: {e}")
    
    return anomalies

def compute_spectral_residual(signal: np.ndarray, 
                            sample_rate: float = 1000.0,
                            saliency_threshold: float = 2.0) -> List[Dict[str, Any]]:
    """
    Detect anomalies using spectral residual (saliency detection)
    
    Args:
        signal: Input signal array
        sample_rate: Sampling rate in Hz
        saliency_threshold: Threshold for spectral saliency
    
    Returns:
        List of spectral anomaly dictionaries
    """
    anomalies = []
    
    if len(signal) < 256:
        return anomalies
    
    try:
        # Remove NaN values
        signal_clean = signal[~np.isnan(signal)]
        if len(signal_clean) < 256:
            return anomalies
        
        n = len(signal_clean)
        
        # Compute FFT
        fft_vals = fft(signal_clean)
        frequencies = fftfreq(n, 1/sample_rate)
        magnitudes = np.abs(fft_vals)
        phases = np.angle(fft_vals)
        
        # Compute spectral residual (log spectrum - smoothed log spectrum)
        log_amplitude = np.log(magnitudes + 1e-8)
        smoothed_log_amp = scipy_signal.medfilt(log_amplitude, kernel_size=5)
        spectral_residual = log_amplitude - smoothed_log_amp
        
        # Compute saliency map
        saliency_map = np.exp(spectral_residual + 1j * phases)
        saliency_signal = np.abs(np.fft.ifft(saliency_map))
        
        # Normalize saliency
        saliency_signal = (saliency_signal - np.mean(saliency_signal)) / (np.std(saliency_signal) + 1e-8)
        
        # Find significant saliency peaks
        peak_indices, properties = scipy_signal.find_peaks(
            saliency_signal, 
            height=saliency_threshold,
            distance=10,
            prominence=1.0
        )
        
        for idx in peak_indices:
            if idx < len(saliency_signal):
                # Find corresponding frequency
                freq_idx = min(idx, len(frequencies) - 1)
                freq = abs(frequencies[freq_idx])
                
                if freq > 0 and freq < sample_rate / 2:  # Valid frequency range
                    severity = min(saliency_signal[idx] / (saliency_threshold * 2), 1.0)
                    
                    anomalies.append({
                        'type': 'spectral_anomaly',
                        'frequency': float(freq),
                        'start_time': float(idx / sample_rate),
                        'end_time': float((idx + 1) / sample_rate),
                        'severity': float(severity),
                        'saliency_score': float(saliency_signal[idx]),
                        'description': f'Spectral anomaly at {freq:.1f}Hz (saliency: {saliency_signal[idx]:.2f})',
                        'confidence': 0.65,
                        'method': 'spectral_residual'
                    })
        
    except Exception as e:
        logger.error(f"Spectral residual analysis failed: {e}")
    
    return anomalies

def compute_rolling_statistics(signal: np.ndarray, 
                             window_size: int = 100,
                             step_size: int = 10) -> Dict[str, np.ndarray]:
    """
    Compute rolling statistics for trend analysis
    
    Args:
        signal: Input signal array
        window_size: Size of rolling window
        step_size: Step size for rolling window
    
    Returns:
        Dictionary of rolling statistics
    """
    stats = {}
    
    if len(signal) < window_size:
        return stats
    
    try:
        series = pd.Series(signal)
        
        stats['rolling_mean'] = series.rolling(window=window_size, center=True).mean().values
        stats['rolling_std'] = series.rolling(window=window_size, center=True).std().values
        stats['rolling_median'] = series.rolling(window=window_size, center=True).median().values
        stats['rolling_min'] = series.rolling(window=window_size, center=True).min().values
        stats['rolling_max'] = series.rolling(window=window_size, center=True).max().values
        
        # Compute rolling kurtosis and skewness
        stats['rolling_kurtosis'] = series.rolling(window=window_size, center=True).apply(
            lambda x: kurtosis(x) if len(x) == window_size else np.nan
        ).values
        
        stats['rolling_skewness'] = series.rolling(window=window_size, center=True).apply(
            lambda x: skew(x) if len(x) == window_size else np.nan
        ).values
        
    except Exception as e:
        logger.error(f"Rolling statistics computation failed: {e}")
    
    return stats

def detect_peaks(signal: np.ndarray, 
                height: float = None,
                threshold: float = None,
                distance: int = 10,
                prominence: float = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detect peaks in signal with comprehensive properties
    
    Args:
        signal: Input signal array
        height: Minimum height of peaks
        threshold: Minimum threshold
        distance: Minimum distance between peaks
        prominence: Minimum prominence of peaks
    
    Returns:
        Tuple of (peak_indices, peak_properties)
    """
    try:
        if height is None:
            height = np.mean(signal) + np.std(signal)
        
        if prominence is None:
            prominence = np.std(signal) * 0.5
        
        peak_indices, properties = scipy_signal.find_peaks(
            signal,
            height=height,
            threshold=threshold,
            distance=distance,
            prominence=prominence
        )
        
        return peak_indices, properties
        
    except Exception as e:
        logger.error(f"Peak detection failed: {e}")
        return np.array([]), {}

def compute_spectral_entropy(signal: np.ndarray, 
                           sample_rate: float = 1000.0,
                           nperseg: int = 256) -> float:
    """
    Compute spectral entropy of signal
    
    Args:
        signal: Input signal array
        sample_rate: Sampling rate in Hz
        nperseg: Segment length for PSD calculation
    
    Returns:
        Spectral entropy value
    """
    try:
        if len(signal) < nperseg:
            nperseg = len(signal)
        
        # Compute power spectral density
        frequencies, psd = scipy_signal.welch(signal, sample_rate, nperseg=nperseg)
        
        # Normalize PSD to create probability distribution
        psd_normalized = psd / np.sum(psd)
        
        # Remove zeros to avoid log(0)
        psd_normalized = psd_normalized[psd_normalized > 0]
        
        # Compute spectral entropy
        spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(psd_normalized))
        normalized_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0
        
        return float(normalized_entropy)
        
    except Exception as e:
        logger.error(f"Spectral entropy computation failed: {e}")
        return 0.0

def compute_wavelet_coefficients(signal: np.ndarray,
                               wavelet: str = 'db4',
                               level: int = 5) -> Dict[str, np.ndarray]:
    """
    Compute wavelet transform coefficients
    
    Args:
        signal: Input signal array
        wavelet: Wavelet type
        level: Decomposition level
    
    Returns:
        Dictionary of wavelet coefficients
    """
    coefficients = {}
    
    try:
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        coefficients['approximation'] = coeffs[0]  # Lowest frequency component
        
        # Detail coefficients (high frequency components)
        for i in range(1, len(coeffs)):
            coefficients[f'detail_{i}'] = coeffs[i]
        
        # Compute energy distribution
        total_energy = np.sum(signal**2)
        coeff_energy = {}
        
        for key, coeff in coefficients.items():
            coeff_energy[f'{key}_energy'] = np.sum(coeff**2) / total_energy
        
        coefficients['energy_distribution'] = coeff_energy
        
    except Exception as e:
        logger.error(f"Wavelet decomposition failed: {e}")
    
    return coefficients

def bandpass_filter(signal: np.ndarray,
                   sample_rate: float,
                   lowcut: float = None,
                   highcut: float = None,
                   order: int = 4) -> np.ndarray:
    """
    Apply bandpass filter to signal
    
    Args:
        signal: Input signal array
        sample_rate: Sampling rate in Hz
        lowcut: Low cutoff frequency
        highcut: High cutoff frequency
        order: Filter order
    
    Returns:
        Filtered signal
    """
    try:
        if lowcut is None and highcut is None:
            return signal
        
        nyquist = 0.5 * sample_rate
        
        if lowcut is not None and highcut is not None:
            # Bandpass filter
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = scipy_signal.butter(order, [low, high], btype='band')
        elif lowcut is not None:
            # Highpass filter
            low = lowcut / nyquist
            b, a = scipy_signal.butter(order, low, btype='high')
        elif highcut is not None:
            # Lowpass filter
            high = highcut / nyquist
            b, a = scipy_signal.butter(order, high, btype='low')
        else:
            return signal
        
        # Apply filter
        filtered_signal = scipy_signal.filtfilt(b, a, signal)
        return filtered_signal
        
    except Exception as e:
        logger.error(f"Bandpass filtering failed: {e}")
        return signal

def compute_autocorrelation(signal: np.ndarray,
                          max_lag: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute autocorrelation function
    
    Args:
        signal: Input signal array
        max_lag: Maximum lag to compute
    
    Returns:
        Tuple of (lags, autocorrelation)
    """
    try:
        if max_lag is None:
            max_lag = min(len(signal) // 2, 1000)
        
        autocorr = scipy_signal.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:] / np.max(autocorr)
        lags = np.arange(len(autocorr))
        
        return lags[:max_lag], autocorr[:max_lag]
        
    except Exception as e:
        logger.error(f"Autocorrelation computation failed: {e}")
        return np.array([]), np.array([])

def compute_harmonic_ratios(signal: np.ndarray,
                          sample_rate: float,
                          fundamental_freq: float = None) -> Dict[str, float]:
    """
    Compute harmonic ratios and inharmonicity
    
    Args:
        signal: Input signal array
        sample_rate: Sampling rate in Hz
        fundamental_freq: Fundamental frequency (if known)
    
    Returns:
        Dictionary of harmonic metrics
    """
    metrics = {}
    
    try:
        # Compute FFT
        n = len(signal)
        frequencies = fftfreq(n, 1/sample_rate)
        magnitudes = np.abs(fft(signal)) / n
        
        # Only positive frequencies
        positive_idx = frequencies > 0
        frequencies = frequencies[positive_idx]
        magnitudes = magnitudes[positive_idx]
        
        if fundamental_freq is None:
            # Find fundamental frequency (peak with lowest frequency)
            peak_indices, _ = detect_peaks(magnitudes, height=np.max(magnitudes)*0.1)
            if len(peak_indices) > 0:
                fundamental_idx = peak_indices[0]
                fundamental_freq = frequencies[fundamental_idx]
            else:
                fundamental_freq = frequencies[np.argmax(magnitudes)]
        
        if fundamental_freq == 0:
            return metrics
        
        # Compute harmonic ratios
        harmonic_energy = 0
        total_energy = np.sum(magnitudes**2)
        harmonic_deviation = 0
        harmonic_count = 0
        
        for i in range(1, 11):  # Check first 10 harmonics
            harmonic_freq = i * fundamental_freq
            if harmonic_freq > sample_rate / 2:
                break
            
            # Find peak near harmonic frequency
            tolerance = fundamental_freq * 0.05  # 5% tolerance
            harmonic_idx = np.where((frequencies > harmonic_freq - tolerance) & 
                                  (frequencies < harmonic_freq + tolerance))[0]
            
            if len(harmonic_idx) > 0:
                harmonic_mag = np.max(magnitudes[harmonic_idx])
                harmonic_energy += harmonic_mag**2
                
                # Compute frequency deviation
                exact_harmonic_idx = np.argmin(np.abs(frequencies - harmonic_freq))
                deviation = abs(frequencies[exact_harmonic_idx] - harmonic_freq) / fundamental_freq
                harmonic_deviation += deviation
                harmonic_count += 1
        
        metrics['harmonic_ratio'] = harmonic_energy / total_energy if total_energy > 0 else 0
        metrics['inharmonicity'] = harmonic_deviation / harmonic_count if harmonic_count > 0 else 0
        metrics['fundamental_frequency'] = fundamental_freq
        metrics['harmonic_count'] = harmonic_count
        
    except Exception as e:
        logger.error(f"Harmonic ratio computation failed: {e}")
    
    return metrics

def compute_signal_features(signal: np.ndarray,
                          sample_rate: float = 1000.0) -> Dict[str, float]:
    """
    Compute comprehensive signal features for ML analysis
    
    Args:
        signal: Input signal array
        sample_rate: Sampling rate in Hz
    
    Returns:
        Dictionary of signal features
    """
    features = {}
    
    try:
        signal_clean = signal[~np.isnan(signal)]
        if len(signal_clean) < 10:
            return features
        
        # Time-domain features
        features['rms'] = float(np.sqrt(np.mean(signal_clean**2)))
        features['variance'] = float(np.var(signal_clean))
        features['kurtosis'] = float(kurtosis(signal_clean))
        features['skewness'] = float(skew(signal_clean))
        features['crest_factor'] = float(np.max(np.abs(signal_clean)) / features['rms'] if features['rms'] > 0 else 0)
        features['form_factor'] = features['rms'] / (np.mean(np.abs(signal_clean)) + 1e-8)
        
        # Zero-crossing rate
        zero_crossings = np.where(np.diff(np.signbit(signal_clean)))[0]
        features['zero_crossing_rate'] = float(len(zero_crossings) / len(signal_clean))
        
        # Frequency-domain features
        features['spectral_entropy'] = compute_spectral_entropy(signal_clean, sample_rate)
        
        # Autocorrelation features
        lags, autocorr = compute_autocorrelation(signal_clean, max_lag=100)
        if len(autocorr) > 1:
            features['autocorr_first_peak'] = float(np.max(autocorr[1:]))
            features['autocorr_decay'] = float(autocorr[0] - autocorr[-1] if len(autocorr) > 10 else 0)
        
        # Wavelet features
        wavelet_coeffs = compute_wavelet_coefficients(signal_clean)
        if 'energy_distribution' in wavelet_coeffs:
            features.update(wavelet_coeffs['energy_distribution'])
        
    except Exception as e:
        logger.error(f"Signal feature computation failed: {e}")
    
    return features
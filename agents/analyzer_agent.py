# agents/analyzer_agent.py 
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import asyncio
import logging
import json
import uuid
import io
import csv
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from scipy.fft import fft, fftfreq
from scipy import signal as scipy_signal
from scipy.stats import zscore, kurtosis, skew
import matplotlib.pyplot as plt
from utils.visualization import generate_analysis_visualizations

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Coral framework imports
from coral_s import (
    CoralBaseAgent, AgentCapability, MessageType, CoralMessage,
    TaskPriority, AgentStatus
)

class UniversalAnalyzerAgent(CoralBaseAgent):
    """
    Universal Analyzer Agent - Enhanced with comprehensive file format support
    """

    def __init__(self, agent_id: str = "analyzer_001", ollama_url: str = "http://localhost:11434"):
        capabilities = [
            AgentCapability(
                name="universal_analysis",
                version="7.2",
                description="Universal data analysis with comprehensive file format support",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_content": {"type": "string", "description": "Raw file content"},
                        "file_type": {"type": "string", "description": "File type override"},
                        "analysis_type": {"type": "string",
                                         "enum": ["quick", "comprehensive", "trend", "anomaly"]},
                        "use_llm": {"type": "boolean", "description": "Enable intelligent analysis"}
                    },
                    "required": []
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "data_profile": {"type": "object", "description": "Data characteristics"},
                        "anomalies": {"type": "array", "description": "Detected anomalies"},
                        "features": {"type": "object", "description": "Extracted features"},
                        "patterns": {"type": "object", "description": "Discovered patterns"},
                        "quality_metrics": {"type": "object", "description": "Data quality assessment"},
                        "summary": {"type": "string", "description": "Analysis summary"},
                        "recommendations": {"type": "array", "description": "Actionable insights"}
                    }
                },
                performance_metrics={"accuracy": 0.95, "latency_ms": 400}
            )
        ]

        super().__init__(agent_id, "Universal Data Analyzer", "analyzer", capabilities)
        
        self.ollama_url = ollama_url
        self.model = "phi3:mini"
        self.llm_enabled = HAS_REQUESTS
        
        if self.llm_enabled:
            self.logger.info(f"Ollama integration enabled: {self.ollama_url}, model: {self.model}")
        else:
            self.logger.warning("Ollama disabled - requests module not available")

        self.artifact_dir = "artifacts"
        os.makedirs(self.artifact_dir, exist_ok=True)
        
    async def generate_visualizations(self, df: pd.DataFrame, anomalies: List[Dict[str, Any]], 
                                    sample_rate: float = 1000.0, 
                                    include_animations: bool = True) -> Dict[str, str]:
        """Generate visualizations for analysis results"""
        try:
            self.logger.info(f"Generating visualizations for {len(df)} rows, {len(df.columns)} columns")
            
            # First numeric column for visualization
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                self.logger.warning("No numeric columns found for visualization")
                return {}
                
            signal_col = numeric_cols[0]
            signal = df[signal_col].dropna().values
            
            if len(signal) < 10:
                self.logger.warning(f"Signal too short for visualization: {len(signal)} samples")
                return {}
            
            self.logger.info(f"Generating visualizations for column '{signal_col}' with {len(signal)} samples")
            
            # Generate visualizations
            artifacts = generate_analysis_visualizations(
                signal=signal,
                sample_rate=sample_rate,
                anomalies=anomalies,
                analysis_type="comprehensive",
                include_animations=include_animations
            )
            
            self.logger.info(f"Generated {len(artifacts)} visualization artifacts")
            return artifacts
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            return {}

    async def query_ollama(self, prompt: str, system_prompt: str = None) -> str:
        """Query local Ollama with better timeout handling"""
        if not self.llm_enabled:
            self.logger.debug("Ollama disabled, skipping LLM query")
            return None
            
        try:
            url = f"{self.ollama_url}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_predict": 500  
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            # Timeout for faster fallback
            self.logger.debug(f"Sending query to Ollama: {prompt[:100]}...")
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get("response", "").strip()
            self.logger.debug(f"Ollama response received: {response_text[:100]}...")
            return response_text
            
        except requests.exceptions.Timeout:
            self.logger.warning("Ollama timeout - using fallback analysis")
            return None
        except Exception as e:
            self.logger.warning(f"Ollama query failed: {e}")
            return None

    async def analyze_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data structure and characteristics with detailed logging"""
        self.logger.info(f"Analyzing data structure: {df.shape[0]} rows × {df.shape[1]} columns")
        
        profile = {
            "data_type": "numeric",
            "dimensionality": f"{df.shape[0]} rows × {df.shape[1]} columns",
            "numeric_columns": [],
            "data_quality": {},
            "statistical_profile": {}
        }
        
        for col in df.columns:
            col_data = df[col].dropna()
            self.logger.debug(f"Analyzing column '{col}': {len(col_data)} non-null values")
            
            if pd.api.types.is_numeric_dtype(df[col]):
                profile["numeric_columns"].append(col)
                # Statistical profile for numeric columns
                if len(col_data) > 0:
                    with np.errstate(all='ignore'):  
                        profile["statistical_profile"][col] = {
                            "mean": float(np.nanmean(col_data)),
                            "std": float(np.nanstd(col_data)),
                            "min": float(np.nanmin(col_data)),
                            "max": float(np.nanmax(col_data)),
                            "missing_values": int(df[col].isna().sum())
                        }
                    self.logger.debug(f"Column '{col}' stats: mean={profile['statistical_profile'][col]['mean']:.3f}, "
                                    f"std={profile['statistical_profile'][col]['std']:.3f}")
        
        # Overall data quality
        missing_ratio = float(df.isna().sum().sum() / (len(df) * len(df.columns)) + 1e-8)
        profile["data_quality"]["overall"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "complete_cases": len(df.dropna()),
            "missing_data_ratio": missing_ratio
        }
        
        self.logger.info(f"Data analysis complete: {len(profile['numeric_columns'])} numeric columns, "
                        f"missing ratio: {missing_ratio:.3f}")
        
        return profile

    async def detect_anomalies_universal(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Universal anomaly detection with robust numerical handling and detailed logging"""
        all_anomalies = []
        self.logger.info(f"Starting anomaly detection on {len(df.columns)} columns")
        
        # Analyze numeric columns for anomalies
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.logger.info(f"Found {len(numeric_cols)} numeric columns for anomaly detection")
        
        for col in numeric_cols:
            signal = df[col].dropna().values
            self.logger.debug(f"Analyzing column '{col}' for anomalies: {len(signal)} samples")
            
            if len(signal) < 5:
                self.logger.debug(f"Column '{col}' has insufficient data for anomaly detection: {len(signal)} samples")
                continue
                
            # Method 1: Statistical outliers (Z-score) with safe numerical handling
            with np.errstate(all='ignore'): 
                try:
                    # Robust z-score calculation
                    signal_clean = signal[~np.isnan(signal)]
                    if len(signal_clean) < 5 or np.std(signal_clean) < 1e-10:
                        self.logger.debug(f"Column '{col}' has low variance, skipping z-score analysis")
                        continue
                        
                    z_scores = np.abs((signal_clean - np.mean(signal_clean)) / (np.std(signal_clean) + 1e-8))
                    statistical_outliers = np.where(z_scores > 3.0)[0]
                    
                    self.logger.debug(f"Column '{col}': found {len(statistical_outliers)} statistical outliers")
                    
                    for idx in statistical_outliers:
                        if idx < len(signal_clean):
                            all_anomalies.append({
                                'type': 'statistical_outlier',
                                'column': col,
                                'index': int(idx),
                                'value': float(signal_clean[idx]),
                                'z_score': float(z_scores[idx]),
                                'severity': min(z_scores[idx] / 6.0, 1.0),
                                'description': f'Statistical outlier in {col}: value {signal_clean[idx]:.3f} (z-score: {z_scores[idx]:.2f})',
                                'confidence': 0.8
                            })
                except Exception as e:
                    self.logger.debug(f"Z-score analysis failed for {col}: {e}")
                    continue
            
            # Method 2: Extreme values (beyond 4 sigma)
            try:
                with np.errstate(all='ignore'):
                    extreme_threshold = np.mean(signal) + 4 * np.std(signal)
                    if not np.isnan(extreme_threshold) and not np.isinf(extreme_threshold):
                        extreme_indices = np.where(np.abs(signal) > extreme_threshold)[0]
                        self.logger.debug(f"Column '{col}': found {len(extreme_indices)} extreme values")
                        
                        for idx in extreme_indices:
                            if idx < len(signal):
                                all_anomalies.append({
                                    'type': 'extreme_value',
                                    'column': col,
                                    'index': int(idx),
                                    'value': float(signal[idx]),
                                    'threshold': float(extreme_threshold),
                                    'description': f'Extreme value in {col}: {signal[idx]:.3f} exceeds threshold {extreme_threshold:.3f}',
                                    'confidence': 0.7
                                })
            except Exception as e:
                self.logger.debug(f"Extreme value detection failed for {col}: {e}")
        
        self.logger.info(f"Anomaly detection complete: found {len(all_anomalies)} total anomalies")
        return all_anomalies

    async def extract_features_universal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract universal features with safe numerical operations and logging"""
        features = {
            "statistical": {},
            "distributional": {},
            "temporal": {}
        }
        
        self.logger.info("Extracting features from data")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            signal = df[col].dropna().values
            self.logger.debug(f"Extracting features from column '{col}': {len(signal)} samples")
            
            if len(signal) < 5:
                continue
                
            # Statistical features
            with np.errstate(all='ignore'):
                features["statistical"][col] = {
                    "mean": float(np.nanmean(signal)),
                    "std": float(np.nanstd(signal)),
                    "variance": float(np.nanvar(signal)),
                    "range": float(np.nanmax(signal) - np.nanmin(signal)),
                }
                
                # Distribution features 
                if len(signal) > 10:
                    try:
                        features["distributional"][col] = {
                            "skewness": float(skew(signal)) if np.std(signal) > 1e-10 else 0.0,
                            "kurtosis": float(kurtosis(signal)) if np.std(signal) > 1e-10 else 0.0,
                        }
                        self.logger.debug(f"Column '{col}': skewness={features['distributional'][col]['skewness']:.3f}, "
                                        f"kurtosis={features['distributional'][col]['kurtosis']:.3f}")
                    except Exception as e:
                        self.logger.debug(f"Distribution analysis failed for {col}: {e}")
                        features["distributional"][col] = {
                            "skewness": 0.0,
                            "kurtosis": 0.0
                        }
        
        self.logger.info(f"Feature extraction complete: {len(features['statistical'])} columns processed")
        return features

    async def generate_intelligent_summary(self, analysis_results: Dict, data_profile: Dict) -> str:
        """Generate intelligent summary with fallback"""
        if not self.llm_enabled:
            self.logger.debug("LLM disabled, using fallback summary")
            return self._generate_fallback_summary(analysis_results, data_profile)
        
        prompt = f"""
        Please provide a brief analysis summary:

        DATA: {data_profile.get('dimensionality', 'Unknown')}
        COLUMNS: {data_profile.get('numeric_columns', [])}
        ANOMALIES: {len(analysis_results.get('anomalies', []))}
        
        Provide 2-3 sentences about:
        1. Data characteristics
        2. Anomaly significance  
        3. Key observations

        Be concise and professional.
        """

        try:
            self.logger.debug("Generating LLM-enhanced summary")
            summary = await self.query_ollama(prompt)
            if summary:
                self.logger.debug("LLM summary generated successfully")
                return summary
        except Exception as e:
            self.logger.warning(f"LLM summary generation failed: {e}")
            
        self.logger.debug("Using fallback summary")
        return self._generate_fallback_summary(analysis_results, data_profile)

    def _generate_fallback_summary(self, analysis_results: Dict, data_profile: Dict) -> str:
        """Generate fallback summary without LLM"""
        anomaly_count = len(analysis_results.get("anomalies", []))
        numeric_cols = len(data_profile.get("numeric_columns", []))
        
        self.logger.debug(f"Generating fallback summary: {anomaly_count} anomalies, {numeric_cols} numeric columns")
        
        # Detailed summary based on the actual anomalies found
        if anomaly_count > 0:
            # Analyze the anomalies to provide specific insights
            anomaly_columns = set()
            max_severity = 0
            
            for anomaly in analysis_results.get("anomalies", []):
                anomaly_columns.add(anomaly.get('column', 'unknown'))
                max_severity = max(max_severity, anomaly.get('severity', 0))
            
            return f"""**Analysis Complete - {anomaly_count} Anomalies Detected**

- **Data Scope**: Analyzed {numeric_cols} numeric columns with {data_profile.get('dimensionality', 'unknown size')}
- **Anomaly Distribution**: Found in {len(anomaly_columns)} different columns
- **Severity Level**: Maximum anomaly severity {max_severity:.1%}
- **Key Insight**: Data shows significant statistical outliers requiring investigation

**Recommendation**: Review anomalies in columns {list(anomaly_columns)} for potential data quality issues or significant events."""
        else:
            return f"""**Analysis Complete - No Significant Anomalies**

- **Data Scope**: Analyzed {numeric_cols} numeric columns with {data_profile.get('dimensionality', 'unknown size')}
- **Data Quality**: Good statistical consistency across all columns
- **Pattern Assessment**: No significant outliers detected using statistical methods

**Recommendation**: Data appears consistent. Consider more advanced pattern analysis if deeper insights are needed."""

    async def process_task(self, task_data: dict) -> dict:
        """Main analysis pipeline with robust error handling and comprehensive logging"""
        try:
            file_content = task_data.get("file_content", "")
            file_type = task_data.get("file_type", "auto")
            analysis_type = task_data.get("analysis_type", "comprehensive")
            use_llm = task_data.get("use_llm", True)
            
            self.logger.info(f"Starting universal analysis - Type: {file_type}, Analysis: {analysis_type}")
            self.logger.debug(f"File content preview: {file_content[:200]}...")

            # Step 1: Parse data
            df, parse_info = await self._parse_universal_data(file_content, file_type, task_data.get("metadata", {}))
            
            if df.empty:
                self.logger.error("No valid data found for analysis")
                return self._create_error_result("No valid data found for analysis")

            self.logger.info(f"Data parsed successfully: {parse_info.get('rows_processed', 0)} rows, "
                           f"{parse_info.get('columns_found', 0)} columns")

            # Step 2: Analyze data structure
            data_profile = await self.analyze_data_structure(df)
            
            # Step 3: Universal anomaly detection
            anomalies = await self.detect_anomalies_universal(df)
            
            # Step 4: Extract universal features
            features = await self.extract_features_universal(df)
            
            # Generate Visualizations
            sample_rate = task_data.get("metadata", {}).get("sample_rate", 1000.0)
            artifacts = await self.generate_visualizations(df, anomalies, sample_rate)
            
            # Step 5: Generate intelligent summary
            analysis_results = {
                "anomalies": anomalies,
                "features": features,
                "patterns": await self._discover_patterns(df)
            }
            
            summary = await self.generate_intelligent_summary(analysis_results, data_profile)

            # Step 6: Compile final results
            results = {
                "data_profile": data_profile,
                "anomalies": anomalies,
                "features": features,
                "patterns": analysis_results["patterns"],
                "quality_metrics": data_profile["data_quality"],
                "summary": summary,
                "recommendations": self._generate_recommendations(data_profile, anomalies),
                "artifacts": artifacts,  
                "metadata": {
                    "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                    "agent_id": self.agent_id,
                    "data_dimensions": data_profile["dimensionality"],
                    "analysis_scope": "universal",
                    "llm_enhanced": use_llm and self.llm_enabled,
                    "parse_info": parse_info
                }
            }

            self.logger.info(f"Universal analysis complete: {len(anomalies)} anomalies found in {len(df.columns)} columns")
            return results

        except Exception as e:
            self.logger.error(f"Universal analysis failed: {e}")
            return self._create_error_result(f"Analysis failed: {e}")

    async def _discover_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Discover patterns in data"""
        patterns = {
            "correlations": {},
            "trends": {}
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.logger.debug(f"Discovering patterns in {len(numeric_cols)} numeric columns")
        
        if len(numeric_cols) > 1:
            try:
                # Correlations between numeric columns
                correlation_matrix = df[numeric_cols].corr()
                patterns["correlations"] = {
                    "matrix": correlation_matrix.to_dict(),
                    "strong_pairs": self._find_strong_correlations(correlation_matrix)
                }
                self.logger.debug(f"Found {len(patterns['correlations']['strong_pairs'])} strong correlations")
            except Exception as e:
                self.logger.debug(f"Correlation analysis failed: {e}")
                patterns["correlations"] = {"error": "Could not compute correlations"}
        
        return patterns

    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Dict]:
        """Find strongly correlated column pairs"""
        strong_pairs = []
        columns = corr_matrix.columns
        
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                try:
                    corr = corr_matrix.iloc[i, j]
                    if not np.isnan(corr) and abs(corr) > threshold:
                        strong_pairs.append({
                            "column1": columns[i],
                            "column2": columns[j],
                            "correlation": float(corr),
                            "strength": "strong" if abs(corr) > 0.9 else "moderate"
                        })
                except Exception:
                    continue
        
        return strong_pairs

    def _generate_recommendations(self, data_profile: Dict, anomalies: List[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Data quality recommendations
        missing_ratio = data_profile.get("data_quality", {}).get("overall", {}).get("missing_data_ratio", 0)
        if missing_ratio > 0.1:
            recommendations.append(f"Consider data imputation for {missing_ratio:.1%} missing values")
        
        # Anomaly recommendations
        if anomalies:
            anomaly_columns = set(anomaly.get('column') for anomaly in anomalies)
            recommendations.append(f"Investigate {len(anomalies)} anomalies across {len(anomaly_columns)} columns")
        
        self.logger.debug(f"Generated {len(recommendations)} recommendations")
        return recommendations

    async def _parse_universal_data(self, file_content: str, file_type: str, metadata: dict) -> Tuple[pd.DataFrame, Dict]:
        """Parse any data format into DataFrame with comprehensive format support"""
        try:
            data_info = {
                "parse_success": False, 
                "rows_processed": 0, 
                "columns_found": 0,
                "file_type_detected": "unknown",
                "parse_method": "unknown"
            }
            
            self.logger.info(f"Parsing data - Type: {file_type}, Content length: {len(file_content)}")
            
            if not file_content or not file_content.strip():
                self.logger.warning("Empty file content provided")
                return pd.DataFrame(), data_info

            # === JSON PARSING ===
            if file_type == "json" or (file_type == "auto" and file_content.strip().startswith('{')):
                try:
                    self.logger.info("Attempting JSON parsing")
                    json_data = json.loads(file_content)
                    data_info["file_type_detected"] = "json"
                    
                    # Extract quality metrics arrays and flatten into DataFrame
                    if "quality_metrics" in json_data:
                        self.logger.info("Found quality_metrics in JSON data")
                        quality_data = json_data["quality_metrics"]
                        
                        # Create DataFrame with each metric as a column
                        df_data = {}
                        for metric_name, values in quality_data.items():
                            if isinstance(values, list):
                                df_data[metric_name] = values
                                self.logger.debug(f"Added metric '{metric_name}' with {len(values)} values")
                        
                        if df_data:
                            # Create DataFrame with all metrics as columns
                            max_length = max(len(values) for values in df_data.values())
                            
                            # Pad shorter arrays with NaN to make equal length
                            for key in df_data:
                                if len(df_data[key]) < max_length:
                                    padding = max_length - len(df_data[key])
                                    df_data[key] = df_data[key] + [np.nan] * padding
                                    self.logger.debug(f"Padded column '{key}' with {padding} NaN values")
                            
                            df = pd.DataFrame(df_data)
                            data_info.update({
                                "parse_success": True,
                                "rows_processed": len(df),
                                "columns_found": len(df.columns),
                                "data_type": "json_quality_metrics",
                                "parse_method": "json_quality_metrics"
                            })
                            self.logger.info(f"JSON parsing successful: {len(df)} rows, {len(df.columns)} columns")
                            return df, data_info
                    else:
                        # Try to flatten entire JSON structure
                        self.logger.info("Attempting to flatten JSON structure")
                        df = pd.json_normalize(json_data)
                        if not df.empty:
                            data_info.update({
                                "parse_success": True,
                                "rows_processed": len(df),
                                "columns_found": len(df.columns),
                                "data_type": "json_flattened",
                                "parse_method": "json_flattened"
                            })
                            self.logger.info(f"JSON flattening successful: {len(df)} rows, {len(df.columns)} columns")
                            return df, data_info
                            
                except json.JSONDecodeError as e:
                    self.logger.warning(f"JSON parsing failed: {e}")
                except Exception as e:
                    self.logger.warning(f"JSON processing failed: {e}")

            # === CSV PARSING ===
            if file_type == "csv" or (file_type == "auto" and ("," in file_content or "\t" in file_content)):
                try:
                    self.logger.info("Attempting CSV parsing")
                    # Try different delimiters
                    delimiters = [',', '\t', ';', '|']
                    
                    for delimiter in delimiters:
                        try:
                            if delimiter in file_content:
                                self.logger.debug(f"Trying delimiter: {repr(delimiter)}")
                                csv_reader = csv.reader(file_content.splitlines(), delimiter=delimiter)
                                rows = list(csv_reader)
                                
                                if len(rows) > 0:
                                    self.logger.debug(f"CSV parsing found {len(rows)} rows with delimiter {repr(delimiter)}")
                                    
                                    # Detect header
                                    has_header = any(not self._is_convertible_to_float(cell) for cell in rows[0]) if rows else False
                                    
                                    if has_header:
                                        headers = rows[0]
                                        data_rows = rows[1:]
                                        self.logger.debug(f"Detected header: {headers}")
                                    else:
                                        headers = [f"column_{i}" for i in range(len(rows[0]))]
                                        data_rows = rows
                                        self.logger.debug("No header detected, using generated column names")
                                    
                                    df_data = []
                                    for i, row in enumerate(data_rows):
                                        if row and len(row) == len(headers):
                                            try:
                                                numeric_row = [float(cell) if self._is_convertible_to_float(cell) else np.nan for cell in row]
                                                df_data.append(numeric_row)
                                            except Exception as e:
                                                self.logger.debug(f"Row {i} conversion failed: {e}")
                                                continue
                                    
                                    if df_data:
                                        df = pd.DataFrame(df_data, columns=headers)
                                        data_info.update({
                                            "parse_success": True,
                                            "rows_processed": len(df),
                                            "columns_found": len(df.columns),
                                            "file_type_detected": "csv",
                                            "parse_method": f"csv_delimiter_{repr(delimiter)}"
                                        })
                                        self.logger.info(f"CSV parsing successful: {len(df)} rows, {len(df.columns)} columns")
                                        return df, data_info
                        except Exception as e:
                            self.logger.debug(f"CSV parsing with delimiter {repr(delimiter)} failed: {e}")
                            continue
                            
                except Exception as e:
                    self.logger.warning(f"CSV parsing failed: {e}")

            # === TXT/RAW NUMERIC PARSING ===
            if file_type in ["txt", "log"] or (file_type == "auto" and not file_content.strip().startswith('{')):
                try:
                    self.logger.info("Attempting TXT/raw numeric parsing")
                    lines = file_content.splitlines()
                    self.logger.debug(f"Processing {len(lines)} lines as raw data")
                    
                    all_numeric_data = []
                    
                    for line_num, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Try comma-separated values
                        if "," in line:
                            values = [x.strip() for x in line.split(",")]
                        else:
                            # Try space/tab separated
                            values = line.split()
                        
                        numeric_values = []
                        for value in values:
                            if self._is_convertible_to_float(value):
                                numeric_values.append(float(value))
                        
                        if numeric_values:
                            all_numeric_data.append(numeric_values)
                    
                    if all_numeric_data:
                        # Check if all rows have same length (matrix) or single column
                        lengths = [len(row) for row in all_numeric_data]
                        if len(set(lengths)) == 1 and lengths[0] > 1:
                            # Matrix data
                            df = pd.DataFrame(all_numeric_data, columns=[f"col_{i}" for i in range(lengths[0])])
                            data_info.update({
                                "parse_success": True,
                                "rows_processed": len(df),
                                "columns_found": len(df.columns),
                                "file_type_detected": "txt_matrix",
                                "parse_method": "txt_matrix"
                            })
                            self.logger.info(f"TXT matrix parsing successful: {len(df)} rows, {len(df.columns)} columns")
                            return df, data_info
                        else:
                            # Single column data
                            flat_data = [val for row in all_numeric_data for val in row]
                            df = pd.DataFrame({"value": flat_data})
                            data_info.update({
                                "parse_success": True,
                                "rows_processed": len(df),
                                "columns_found": 1,
                                "file_type_detected": "txt_single_column",
                                "parse_method": "txt_single_column"
                            })
                            self.logger.info(f"TXT single column parsing successful: {len(df)} rows")
                            return df, data_info
                            
                except Exception as e:
                    self.logger.warning(f"TXT parsing failed: {e}")

            self.logger.error("All parsing methods failed")
            return pd.DataFrame(), data_info
            
        except Exception as e:
            self.logger.error(f"Data parsing failed: {e}")
            return pd.DataFrame(), data_info

    def _is_convertible_to_float(self, s: str) -> bool:
        """Check if string can be converted to float"""
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    def _create_error_result(self, error_message: str) -> dict:
        """Create error result"""
        self.logger.error(f"Creating error result: {error_message}")
        return {
            "error": error_message,
            "data_profile": {},
            "anomalies": [],
            "features": {},
            "summary": f"Analysis failed: {error_message}",
            "metadata": {
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_id": self.agent_id
            }
        }

    # Coral protocol handlers
    def _register_message_handlers(self):
        handlers = super()._register_message_handlers()
        handlers.update({
            MessageType.ANALYZE_REQUEST: self._handle_analyze_request,
        })
        return handlers

    async def _handle_analyze_request(self, message: CoralMessage) -> Optional[CoralMessage]:
        """Handle analysis requests"""
        try:
            self.status = AgentStatus.BUSY
            self.logger.info(f"Handling analysis request from {message.sender_id}")
            
            analysis_result = await self.process_task(message.payload)

            response = CoralMessage(
                message_id=self._generate_message_id(),
                thread_id=message.thread_id,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.ANALYZE_RESULT,
                payload=analysis_result,
                timestamp=datetime.now(timezone.utc),
                priority=message.priority,
                correlation_id=message.message_id
            )

            self.status = AgentStatus.ONLINE
            self.logger.info("Analysis request completed successfully")
            return response

        except Exception as e:
            self.logger.error(f"Analysis request failed: {e}")
            return self._create_error_message(message, str(e))
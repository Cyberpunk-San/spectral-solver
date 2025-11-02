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
import os
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
    Universal Analyzer Agent - FIXED with better error handling
    """

    def __init__(self, agent_id: str = "analyzer_001", ollama_url: str = "http://localhost:11434"):
        capabilities = [
            AgentCapability(
                name="universal_analysis",
                version="7.1",
                description="Universal data analysis with robust error handling",
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
            #first numeric column for visualization
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return {}
                
            signal_col = numeric_cols[0]
            signal = df[signal_col].dropna().values
            
            if len(signal) < 10:
                return {}
            
            # Generate visualizations
            artifacts = generate_analysis_visualizations(
                signal=signal,
                sample_rate=sample_rate,
                anomalies=anomalies,
                analysis_type="comprehensive",
                include_animations=include_animations
            )
            
            return artifacts
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            return {}

    async def query_ollama(self, prompt: str, system_prompt: str = None) -> str:
        """Query local Ollama with better timeout handling"""
        if not self.llm_enabled:
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
            
            # timeout for faster fallback
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.Timeout:
            self.logger.warning("Ollama timeout - using fallback analysis")
            return None
        except Exception as e:
            self.logger.warning(f"Ollama query failed: {e}")
            return None

    async def analyze_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data structure and characteristics"""
        profile = {
            "data_type": "numeric",
            "dimensionality": f"{df.shape[0]} rows × {df.shape[1]} columns",
            "numeric_columns": [],
            "data_quality": {},
            "statistical_profile": {}
        }
        
        
        for col in df.columns:
            col_data = df[col].dropna()
            
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
        
        # Overall data quality
        profile["data_quality"]["overall"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "complete_cases": len(df.dropna()),
            "missing_data_ratio": float(df.isna().sum().sum() / (len(df) * len(df.columns)) + 1e-8)
        }
        
        return profile

    async def detect_anomalies_universal(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Universal anomaly detection with robust numerical handling"""
        all_anomalies = []
        
        # Analyze numeric columns for anomalies
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            signal = df[col].dropna().values
            
            if len(signal) < 5:
                continue
                
            # Method 1: Statistical outliers (Z-score) with safe numerical handling
            with np.errstate(all='ignore'): 
                try:
                    # robust z-score calculation
                    signal_clean = signal[~np.isnan(signal)]
                    if len(signal_clean) < 5 or np.std(signal_clean) < 1e-10:
                        continue
                        
                    z_scores = np.abs((signal_clean - np.mean(signal_clean)) / (np.std(signal_clean) + 1e-8))
                    statistical_outliers = np.where(z_scores > 3.0)[0]
                    
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
        
        return all_anomalies

    async def extract_features_universal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract universal features with safe numerical operations"""
        features = {
            "statistical": {},
            "distributional": {},
            "temporal": {}
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            signal = df[col].dropna().values
            
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
                    except:
                        features["distributional"][col] = {
                            "skewness": 0.0,
                            "kurtosis": 0.0
                        }
        
        return features

    async def generate_intelligent_summary(self, analysis_results: Dict, data_profile: Dict) -> str:
        """Generate intelligent summary with fallback"""
        if not self.llm_enabled:
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
            summary = await self.query_ollama(prompt)
            if summary:
                return summary
        except:
            pass
            
        return self._generate_fallback_summary(analysis_results, data_profile)

    def _generate_fallback_summary(self, analysis_results: Dict, data_profile: Dict) -> str:
        """Generate fallback summary without LLM"""
        anomaly_count = len(analysis_results.get("anomalies", []))
        numeric_cols = len(data_profile.get("numeric_columns", []))
        
        # detailed summary based on the actual anomalies found
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
        """Main analysis pipeline with robust error handling"""
        try:
            file_content = task_data.get("file_content", "")
            file_type = task_data.get("file_type", "auto")
            analysis_type = task_data.get("analysis_type", "comprehensive")
            use_llm = task_data.get("use_llm", True)
            
            self.logger.info(f"Starting universal analysis for any data type")

            # Step 1: Parse data
            df, parse_info = await self._parse_universal_data(file_content, file_type, {})
            
            if df.empty:
                return self._create_error_result("No valid data found for analysis")

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
                    "llm_enhanced": use_llm and self.llm_enabled
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
        
        if len(numeric_cols) > 1:
            try:
                # correlations between numeric columns
                correlation_matrix = df[numeric_cols].corr()
                patterns["correlations"] = {
                    "matrix": correlation_matrix.to_dict(),
                    "strong_pairs": self._find_strong_correlations(correlation_matrix)
                }
            except:
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
                except:
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
        
        return recommendations

    async def _parse_universal_data(self, file_content: str, file_type: str, metadata: dict) -> Tuple[pd.DataFrame, Dict]:
        """Parse any data format into DataFrame"""
        try:
            data_info = {"parse_success": False, "rows_processed": 0, "columns_found": 0}
            
            if not file_content or not file_content.strip():
                return pd.DataFrame(), data_info

            # CSV parsing
            if file_type == "csv" or (file_type == "auto" and "," in file_content):
                csv_reader = csv.reader(file_content.splitlines())
                rows = list(csv_reader)
                
                if len(rows) < 1:
                    return pd.DataFrame(), data_info
                
                has_header = any(not self._is_convertible_to_float(cell) for cell in rows[0]) if rows else False
                
                if has_header:
                    headers = rows[0]
                    data_rows = rows[1:]
                else:
                    headers = [f"column_{i}" for i in range(len(rows[0]))]
                    data_rows = rows
                
                df_data = []
                for row in data_rows:
                    if row and len(row) == len(headers):
                        try:
                            numeric_row = [float(cell) if self._is_convertible_to_float(cell) else np.nan for cell in row]
                            df_data.append(numeric_row)
                        except:
                            continue
                
                if df_data:
                    df = pd.DataFrame(df_data, columns=headers)
                    data_info.update({
                        "parse_success": True,
                        "rows_processed": len(df),
                        "columns_found": len(df.columns)
                    })
                    return df, data_info

            # Fallback to raw numeric data
            try:
                if "," in file_content:
                    raw_data = [float(x.strip()) for x in file_content.split(",") if x.strip() and self._is_convertible_to_float(x.strip())]
                else:
                    raw_data = [float(x.strip()) for x in file_content.split() if x.strip() and self._is_convertible_to_float(x.strip())]
                
                if raw_data:
                    df = pd.DataFrame({"value": raw_data})
                    data_info.update({
                        "parse_success": True,
                        "rows_processed": len(df),
                        "columns_found": 1
                    })
                    return df, data_info
            except:
                pass
            
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
            return response

        except Exception as e:
            self.logger.error(f"Analysis request failed: {e}")
            return self._create_error_message(message, str(e))
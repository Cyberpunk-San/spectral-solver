# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import asyncio
import logging
from datetime import datetime
import sys
import os
from typing import Dict  


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.analyzer_agent import UniversalAnalyzerAgent
from agents.investigator_agent import InvestigatorAgent
from agents.reporter_agent import ReporterAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_app")

st.set_page_config(
    page_title="Spectral Solver - Industrial Analysis",
    page_icon="^_____^",
    layout="wide"
)

class StreamlitApp:
    def __init__(self):
        self.analyzer = UniversalAnalyzerAgent("streamlit_analyzer_001")
        self.investigator = InvestigatorAgent("streamlit_investigator_001")
        self.reporter = ReporterAgent("streamlit_reporter_001")
        self.initialized = False
    
    def initialize_agents(self):
        """Initialize agents synchronously"""
        try:
            
            self.initialized = True
            logger.info("All agents initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            st.error(f"Agent initialization failed: {str(e)}")
            return False
    
    def generate_sample_data(self, defect_type="bearing", duration=2, sample_rate=10000):
        """Generate sample vibration data with specified defects"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Base signal 
        base_signal = 0.1 * np.sin(2 * np.pi * 50 * t)  # 50Hz fundamental
        
        if defect_type == "bearing":
            # Bearing defect 
            defect_signal = 0.3 * np.sin(2 * np.pi * 160 * t)  # BPFO frequency
            defect_signal += 0.15 * np.sin(2 * np.pi * 320 * t)  # Harmonic
            # some spikes to simulate bearing defects
            spike_indices = [500, 1500, 2500]
            for idx in spike_indices:
                if idx < len(t):
                    defect_signal[idx] += 0.8
            description = "Bearing Outer Race Defect Simulation"
        elif defect_type == "imbalance":
            # Imbalance - increased 1x RPM
            defect_signal = 0.4 * np.sin(2 * np.pi * 50 * t) 
            description = "Rotor Imbalance Simulation"
        elif defect_type == "misalignment":
            # Misalignment - 2x RPM component
            defect_signal = 0.25 * np.sin(2 * np.pi * 100 * t)
            description = "Shaft Misalignment Simulation"
        else:
            # Normal operation with slight noise
            defect_signal = 0
            description = "Normal Operation Simulation"
        
        noise = 0.02 * np.random.normal(size=len(t))
        signal = base_signal + defect_signal + noise
        
        
        signal_list = [float(x) for x in signal]
        return signal_list, description
    
    async def run_direct_analysis(self, signal, sample_rate, rpm, machine_id, defect_type):
        """Run analysis using direct agent method calls (simpler approach)"""
        try:
            if not self.initialized:
                self.initialize_agents()
            
            # Step 1: analyzer 
            analyzer_data = {
                "file_content": json.dumps({"vibration_data": signal}),  
                "file_type": "json",
                "analysis_type": "comprehensive",
                "metadata": {
                    "sample_rate": sample_rate,
                    "rpm": rpm,
                    "machine_id": machine_id,
                    "signal_type": "vibration",
                    "defect_type": defect_type
                }
            }
            
            st.info(" Running signal analysis...")
            analyzer_result = await self.analyzer.process_task(analyzer_data)
            
            # Step 2: Investigator
            investigator_data = {
                "anomalies": analyzer_result.get("anomalies", []),
                "features": analyzer_result.get("features", {}),
                "metadata": analyzer_data["metadata"],
                "investigation_mode": "root_cause",
                "file_content": analyzer_result
            }
            
            st.info("Investigating root causes...")
            investigator_result = await self.investigator.process_task(investigator_data)
            
            # Step 3: Reporter
            reporter_data = {
                "analyzer_results": analyzer_result,
                "investigator_results": investigator_result,
                "metadata": {
                    **analyzer_data["metadata"],
                    "thread_id": f"streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
            }
            
            st.info("Generating reports...")
            reporter_result = await self.reporter.process_task(reporter_data)
            
            return {
                "analyzer": analyzer_result,
                "investigator": investigator_result,
                "reporter": reporter_result,
                "defect_type": defect_type
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            st.error(f"Analysis pipeline error: {str(e)}")
            return None
    
    async def process_uploaded_file(self, uploaded_file, sample_rate, rpm, machine_id):
        """Process uploaded file through analysis pipeline"""
        try:
            if not self.initialized:
                self.initialize_agents()
            
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # read file content based on file type
            if file_extension in ['csv', 'txt']:
                file_content = uploaded_file.getvalue().decode('utf-8')
            elif file_extension == 'json':
                file_content = uploaded_file.getvalue().decode('utf-8')
            else:
                file_content = uploaded_file.getvalue()
            
            # Step 1: Analyzer
            analyzer_data = {
                "file_content": file_content,
                "file_type": file_extension,
                "analysis_type": "comprehensive",
                "metadata": {
                    "sample_rate": sample_rate,
                    "rpm": rpm,
                    "machine_id": machine_id,
                    "signal_type": "vibration",
                    "original_filename": uploaded_file.name
                }
            }
            
            st.info("Analyzing file content...")
            analyzer_result = await self.analyzer.process_task(analyzer_data)
            
            # Step 2: Investigator
            investigator_data = {
                "anomalies": analyzer_result.get("anomalies", []),
                "features": analyzer_result.get("features", {}),
                "metadata": analyzer_data["metadata"],
                "file_content": analyzer_result,
                "investigation_mode": "root_cause"
            }
            
            st.info("Investigating root causes...")
            investigator_result = await self.investigator.process_task(investigator_data)
            
            # Step 3: Reporter
            reporter_data = {
                "analyzer_results": analyzer_result,
                "investigator_results": investigator_result,
                "metadata": {
                    **analyzer_data["metadata"],
                    "thread_id": f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
            }
            
            st.info("Generating reports...")
            reporter_result = await self.reporter.process_task(reporter_data)
            
            return {
                "analyzer": analyzer_result,
                "investigator": investigator_result,
                "reporter": reporter_result,
                "file_type": file_extension,
                "filename": uploaded_file.name
            }
            
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            st.error(f"File processing error: {str(e)}")
            return None

    def display_animated_visualizations(self, artifacts: Dict[str, str]):
        """Display animated visualizations in Streamlit"""
        animated_plots = {k: v for k, v in artifacts.items() if 'animation' in k or 'animated' in k}
        
        if not animated_plots:
            return
        
        st.subheader("Animated Analysis")
        st.write("Interactive animations showing the analysis process:")
        
        
        for anim_name, anim_path in animated_plots.items():
            if os.path.exists(anim_path):
                try:
                    
                    display_name = anim_name.replace('_', ' ').title()
                    
                    with st.expander(f"{display_name}"):
                        st.image(anim_path, use_column_width=True)
                        
                        
                        with open(anim_path, "rb") as file:
                            st.download_button(
                                label=f"Download {display_name}",
                                data=file,
                                file_name=os.path.basename(anim_path),
                                mime="image/gif"
                            )
                except Exception as e:
                    st.warning(f"Could not display {anim_name}: {str(e)}")

    def display_results(self, results, source_type="demo"):
        """Display analysis results in Streamlit"""
        if not results:
            st.error("Analysis failed. Please check the inputs and try again.")
            return
        
        st.header("📊 Analysis Results")
        
        # Show file info for uploaded files
        if source_type == "uploaded":
            st.subheader(f"📁 File: {results.get('filename', 'Unknown')}")
        
        # Get executive summary
        executive_report = results["reporter"].get("reports", {}).get("executive", {})
        if not executive_report:
            # Fallback to basic info
            executive_report = {
                "summary": results["investigator"].get("diagnosis", "Analysis completed"),
                "urgency": "medium",
                "confidence": results["investigator"].get("confidence", 0.5)
            }
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            urgency = executive_report.get("urgency", "unknown")
            urgency_icons = {
                "critical": " CRITICAL ALERT",
                "high": " HIGH PRIORITY", 
                "medium": "ℹ MEDIUM PRIORITY",
                "low": "NORMAL OPERATION"
            }
            st.metric("Status", urgency_icons.get(urgency, "UNKNOWN"))
        
        with col2:
            confidence = executive_report.get("confidence", 0)
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col3:
            anomalies = len(results["analyzer"].get("anomalies", []))
            st.metric("Anomalies Detected", anomalies)
            
        with col4:
            if source_type == "demo":
                st.metric("Scenario", results.get("defect_type", "Unknown").title())
            else:
                st.metric("File Type", results.get("file_type", "Unknown").upper())
        
        # Diagnosis
        st.subheader("🩺 Diagnosis")
        diagnosis = results["investigator"].get("diagnosis", "No diagnosis available")
        reasoning = results["investigator"].get("reasoning", "")
        
        st.info(f"**Primary Diagnosis:** {diagnosis}")
        
        if reasoning and len(reasoning) > 0:
            with st.expander("View Investigation Reasoning"):
                
                clean_reasoning = reasoning.replace('##', '###').replace('# ', '## ')
                st.markdown(clean_reasoning)
        
        # === UPDATED VISUALIZATIONS SECTION ===
        st.subheader("📈 Visualizations")
        
        # analyzer artifacts
        artifacts = results["analyzer"].get("artifacts", {})
        if artifacts:
            
            static_plots = {k: v for k, v in artifacts.items() if 'animation' not in k and 'animated' not in k}
            animated_plots = {k: v for k, v in artifacts.items() if 'animation' in k or 'animated' in k}
            
            #  static plots
            if static_plots:
                st.write("**Static Visualizations:**")
                col1, col2 = st.columns(2)
                
                displayed_artifacts = 0
                for artifact_name, artifact_path in static_plots.items():
                    if artifact_path and os.path.exists(artifact_path):
                        col = col1 if displayed_artifacts % 2 == 0 else col2
                        with col:
                            st.image(artifact_path, caption=f"{artifact_name.replace('_', ' ').title()}")
                            displayed_artifacts += 1
            
            # animated visualizations
            if animated_plots:
                self.display_animated_visualizations(artifacts)
            
            if displayed_artifacts == 0 and not animated_plots:
                self._create_fallback_visualization(results)
        else:
            self._create_fallback_visualization(results)
        
        
        # Recommendations
        st.subheader(" Recommended Actions")
        maintenance_report = results["reporter"].get("reports", {}).get("maintenance", {})
        recommendations = maintenance_report.get("recommendations", [])
        
        if not recommendations:
            recommendations = results["investigator"].get("recommendations", [])
        
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):
                if isinstance(rec, dict):
                    action = rec.get("action", "Unknown action")
                    priority = rec.get("priority", "medium")
                    icon = "🔴" if priority == "critical" else "🟡" if priority == "high" else "🟢"
                    st.write(f"{icon} {i}. **{action}** (*{priority} priority*)")
                    
                    if "estimated_time" in rec:
                        st.write(f"   ⏱️ Estimated time: {rec['estimated_time']}")
                    if "tools_required" in rec:
                        st.write(f"   🛠️ Tools needed: {', '.join(rec['tools_required'])}")
                else:
                    st.write(f"• {rec}")
        else:
            st.info("No specific recommendations generated.")
        
        # Detailed Findings
        with st.expander(" Detailed Technical Analysis"):
            tab1, tab2, tab3 = st.tabs(["Analyzer Results", "Investigator Results", "Reporter Results"])
            
            with tab1:
                st.json(results["analyzer"], expanded=False)
            with tab2:
                st.json(results["investigator"], expanded=False)
            with tab3:
                st.json(results["reporter"], expanded=False)
        
        # Download Results
        st.subheader("📥 Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="Download Full Analysis Report (JSON)",
                data=json.dumps(results, indent=2),
                file_name=f"spectral_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Generate executive summary text - FIXED INDENTATION
            exec_summary = f"""SPECTRAL SOLVER ANALYSIS REPORT
==============================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Machine: {results['analyzer'].get('metadata', {}).get('machine_id', 'Unknown')}
Status: {executive_report.get('urgency', 'unknown').upper()}
Confidence: {executive_report.get('confidence', 0):.1%}

DIAGNOSIS:
{diagnosis}

KEY FINDINGS:
• Anomalies Detected: {len(results['analyzer'].get('anomalies', []))}
• Primary Issue: {diagnosis}

RECOMMENDED ACTIONS:
{chr(10).join(['• ' + (rec.get('action', rec) if isinstance(rec, dict) else rec) for rec in recommendations[:3]])}
"""
            
            st.download_button(
                label="Download Executive Summary (TXT)",
                data=exec_summary,
                file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    def _create_fallback_visualization(self, results):
        """Create fallback visualizations when no artifacts are available"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Signal preview 
            signal_data = None
            
            if "signal_data" in results["analyzer"]:
                signal_data = results["analyzer"]["signal_data"]
            elif "features" in results["analyzer"] and "statistical" in results["analyzer"]["features"]:
                # reconstruct from features if available
                stats = results["analyzer"]["features"]["statistical"]
                if stats and len(stats) > 0:
                    key = list(stats.keys())[0]
                    mean_val = stats[key].get("mean", 0)
                    std_val = stats[key].get("std", 1)
                    signal_data = np.random.normal(mean_val, std_val, 1000)
            
            if signal_data is not None and len(signal_data) > 0:
                
                preview_data = signal_data[:1000] if len(signal_data) > 1000 else signal_data
                fig = px.line(y=preview_data, title="Signal Preview")
                fig.update_layout(xaxis_title="Samples", yaxis_title="Amplitude")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No signal data available for visualization")
        
        with col2:
            # Feature visualization
            features = results["analyzer"].get("features", {})
            numeric_features = {}
            
            # all numeric features recursively
            def extract_numeric_features(feature_dict, prefix=""):
                for key, value in feature_dict.items():
                    if isinstance(value, dict):
                        extract_numeric_features(value, f"{prefix}{key}.")
                    elif isinstance(value, (int, float)):
                        numeric_features[f"{prefix}{key}"] = value
            
            extract_numeric_features(features)
            
            if numeric_features:
                
                top_features = dict(sorted(numeric_features.items(), 
                                        key=lambda x: abs(x[1]), 
                                        reverse=True)[:8])
                fig = px.bar(x=list(top_features.keys()), 
                        y=list(top_features.values()),
                        title="Key Signal Features")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No feature data available for visualization")

def main():
    app = StreamlitApp()
    
    st.title(" Spectral Solver - Industrial Vibration Analysis")
    st.write("AI-powered mechanical fault detection and predictive maintenance")
    
    # Initialize agents
    if not app.initialized:
        if app.initialize_agents():
            st.sidebar.success("✅ Agents initialized")
        else:
            st.sidebar.error("❌ Agent initialization failed")
    
    # Sidebar Configuration
    st.sidebar.header("Configuration")
    
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["demo", "file_upload"],
        format_func=lambda x: " Demo Scenario" if x == "demo" else "📁 Upload File"
    )
    
    sample_rate = st.sidebar.number_input("Sample Rate (Hz)", 1000, 50000, 10000)
    rpm = st.sidebar.number_input("Machine RPM", 500, 5000, 1750)
    machine_id = st.sidebar.text_input("Machine ID", "Motor-001")
    
    if analysis_mode == "demo":
        st.sidebar.header("Demo Scenario")
        defect_type = st.sidebar.selectbox(
            "Select Defect Type",
            ["normal", "bearing", "imbalance", "misalignment"],
            format_func=lambda x: {
                "normal": " Normal Operation",
                "bearing": "Bearing Defect", 
                "imbalance": " Rotor Imbalance",
                "misalignment": " Shaft Misalignment"
            }[x]
        )
        duration = st.sidebar.slider("Signal Duration (seconds)", 1, 10, 2)
        
        if st.button(" Run Demo Analysis", type="primary"):
            with st.spinner("Generating sample data and running analysis..."):
                # Generate sample data
                signal, description = app.generate_sample_data(defect_type, duration, sample_rate)
                
                # Run analysis pipeline
                results = asyncio.run(app.run_direct_analysis(
                    signal, sample_rate, rpm, machine_id, defect_type
                ))
                
                # Display results
                if results:
                    st.success("✅ Analysis completed successfully!")
                    app.display_results(results, source_type="demo")
                else:
                    st.error("❌ Analysis failed. Please try again.")
    
    else:  # file_upload mode
        st.sidebar.header("File Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Upload Vibration Data File", 
            type=['csv', 'json', 'txt', 'log'],
            help="Supported formats: CSV, JSON, TXT, LOG files with vibration data"
        )
        
        if uploaded_file is not None:
            st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
            st.sidebar.write(f"**File type:** {uploaded_file.type}")
            st.sidebar.write(f"**Size:** {len(uploaded_file.getvalue())} bytes")
            
            if st.sidebar.button("🔍 Analyze Uploaded File", type="primary"):
                with st.spinner("Processing uploaded file through analysis pipeline..."):
                    # Process uploaded file
                    results = asyncio.run(app.process_uploaded_file(
                        uploaded_file, sample_rate, rpm, machine_id
                    ))
                    
                    # Display results
                    if results:
                        st.success("✅ Analysis completed successfully!")
                        app.display_results(results, source_type="uploaded")
                    else:
                        st.error("❌ Analysis failed. Please check the file format and try again.")

    # System Information
    with st.sidebar.expander("System Info"):
        st.write("**Agents Ready:**")
        if app.initialized:
            st.write("• ✅ Universal Analyzer")
            st.write("• ✅ AI Investigator") 
            st.write("• ✅ Multi-Format Reporter")
        else:
            st.write("• 🔄 Initializing agents...")
        st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
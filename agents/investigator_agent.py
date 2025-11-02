# agents/investigator_agent.py
import asyncio
import logging
import json
import aiohttp
import uuid
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import ollama

from coral_s import CoralBaseAgent, AgentCapability, MessageType, CoralMessage, TaskPriority, AgentStatus


class InvestigatorAgent(CoralBaseAgent):
    """
    Enhanced Investigator Agent with Ollama LLM integration
    Provides root cause analysis, correlation detection, and threat tracing
    """

    def __init__(self, agent_id: str = "investigator_001"):
        capabilities = [
            AgentCapability(
                name="diagnostic_reasoning",
                version="2.0",
                description="LLM-powered diagnostic reasoning with multi-mode investigation",
                input_schema={
                    "type": "object",
                    "properties": {
                        "anomalies": {"type": "array", "description": "Detected anomalies from analyzer"},
                        "features": {"type": "object", "description": "Signal features"},
                        "metadata": {"type": "object", "description": "Machine and context metadata"},
                        "investigation_mode": {"type": "string", "enum": [
                            "root_cause", "anomaly_correlation", "threat_tracing", 
                            "budget_investigation", "evidence_analysis"
                        ]},
                        "raw_data": {"type": "string", "description": "Raw data for analysis"},
                        "file_content": {"type": "object", "description": "File data for investigation"}
                    },
                    "required": []
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "diagnosis": {"type": "string", "description": "Primary diagnosis"},
                        "causes": {"type": "array", "description": "Likely causes"},
                        "recommendations": {"type": "array", "description": "Actionable recommendations"},
                        "confidence": {"type": "number", "description": "Confidence score 0-1"},
                        "reasoning": {"type": "string", "description": "Chain of thought"},
                        "patterns_detected": {"type": "array", "description": "Detected patterns"},
                        "risk_assessment": {"type": "object", "description": "Risk analysis"},
                        "visualizations": {"type": "object", "description": "Generated charts"}
                    }
                },
                performance_metrics={"accuracy": 0.92, "reasoning_depth": "advanced"}
            )
        ]

        super().__init__(agent_id, "Advanced Diagnostic Investigator", "investigator", capabilities)

        # Ollama configuration
        self.llm_model = "phi3:mini"  
        self.available_models = self._get_available_models()
        
        # Investigation modes configuration
        self.investigation_modes = {
            "root_cause": {
                "name": "Root Cause Investigation",
                "description": "Trace events and dependencies to uncover the true root cause",
                "color": "#7E57C2",
                "system_prompt": """You are a Root Cause Analysis Expert. Analyze the provided data to identify 
                the fundamental cause of issues. Focus on causal chains, dependency analysis, and underlying 
                system failures."""
            },
            "anomaly_correlation": {
                "name": "Anomaly Correlation", 
                "description": "Connect multiple anomalies to find deeper system links",
                "color": "#F4511E",
                "system_prompt": """You are an Anomaly Correlation Specialist. Identify relationships between 
                different anomalies and events. Look for temporal patterns, spatial correlations, and 
                systemic connections."""
            },
            "threat_tracing": {
                "name": "Cyber Threat Tracing",
                "description": "Track and analyze cyber threats or suspicious behaviors", 
                "color": "#26A69A",
                "system_prompt": """You are a Cyber Threat Investigator. Analyze patterns indicative of 
                security breaches, malicious activities, or system compromises. Focus on attack vectors, 
                persistence mechanisms, and impact assessment."""
            },
            "budget_investigation": {
                "name": "Budget Irregularity Investigation",
                "description": "Investigate spending anomalies and budget leaks",
                "color": "#00897B", 
                "system_prompt": """You are a Financial Investigation Specialist. Analyze financial data, 
                spending patterns, and budget allocations to identify irregularities, fraud indicators, 
                or inefficient resource usage."""
            },
            "evidence_analysis": {
                "name": "Evidence Cross-Analysis", 
                "description": "Combine datasets to detect reinforcing evidence or inconsistencies",
                "color": "#5C6BC0",
                "system_prompt": """You are an Evidence Analysis Expert. Cross-reference multiple data sources 
                to identify consistencies, contradictions, and supporting evidence patterns."""
            }
        }

        self.logger = logging.getLogger(f"investigator.{agent_id}")

    def _get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            models_response = ollama.list()
            return [model["name"] for model in models_response.get('models', [])]
        except Exception as e:
            self.logger.warning(f"Could not fetch Ollama models: {e}")
            return ["phi3:mini"]  

    def _register_message_handlers(self):
        """Register message handlers for investigator-specific messages"""
        handlers = super()._register_message_handlers()
        handlers.update({
            MessageType.ANALYZE_RESULT: self._handle_analyze_result,
            MessageType.INVESTIGATE_REQUEST: self._handle_investigate_request,
            MessageType.TASK_REQUEST: self._handle_task_request,
        })
        return handlers

    async def _handle_analyze_result(self, message: CoralMessage) -> Optional[CoralMessage]:
        """Automatically investigate when analyzer results come in"""
        try:
            self.logger.info(f"Starting investigation for thread {message.thread_id}")
            
            await self.update_status(AgentStatus.BUSY, 0.8)

            #  investigation
            investigation_result = await self.process_task(message.payload)

            response = CoralMessage(
                message_id=self._generate_message_id(),
                thread_id=message.thread_id,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.INVESTIGATE_RESULT,
                payload=investigation_result,
                timestamp=datetime.now(timezone.utc),
                priority=message.priority,
                correlation_id=message.message_id
            )

            await self.update_status(AgentStatus.ONLINE, 0.3)
            self.logger.info(f"Investigation completed for thread {message.thread_id}")
            return response

        except Exception as e:
            self.logger.error(f"Investigation failed: {str(e)}")
            await self.update_status(AgentStatus.ERROR, 0.5)
            
            error_response = CoralMessage(
                message_id=self._generate_message_id(),
                thread_id=message.thread_id,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.ERROR,
                payload={
                    "error": str(e), 
                    "status": "failed",
                    "original_message": message.to_dict()
                },
                timestamp=datetime.now(timezone.utc),
                priority=TaskPriority.HIGH,
                correlation_id=message.message_id
            )
            return error_response

    async def _handle_investigate_request(self, message: CoralMessage) -> Optional[CoralMessage]:
        """Handle direct investigation requests"""
        try:
            self.logger.info(f"Handling direct investigation request for thread {message.thread_id}")
            
            await self.update_status(AgentStatus.BUSY, 0.7)

            # investigation directly from request payload
            investigation_result = await self.process_task(message.payload)

            response = CoralMessage(
                message_id=self._generate_message_id(),
                thread_id=message.thread_id,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.INVESTIGATE_RESULT,
                payload=investigation_result,
                timestamp=datetime.now(timezone.utc),
                priority=message.priority,
                correlation_id=message.message_id
            )

            await self.update_status(AgentStatus.ONLINE, 0.3)
            self.logger.info(f"Direct investigation completed for thread {message.thread_id}")
            return response

        except Exception as e:
            self.logger.error(f"Direct investigation failed: {str(e)}")
            await self.update_status(AgentStatus.ERROR, 0.5)
            
            error_response = CoralMessage(
                message_id=self._generate_message_id(),
                thread_id=message.thread_id,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.ERROR,
                payload={
                    "error": str(e), 
                    "status": "failed",
                    "original_message": message.to_dict()
                },
                timestamp=datetime.now(timezone.utc),
                priority=TaskPriority.HIGH,
                correlation_id=message.message_id
            )
            return error_response

    async def _handle_task_request(self, message: CoralMessage) -> Optional[CoralMessage]:
        """Handle generic task requests"""
        try:
            payload = message.payload
            
            # is it an investigation task
            if (payload.get("anomalies") or 
                payload.get("analyzer_results") or 
                payload.get("raw_data") or
                payload.get("investigation_mode")):
                
                self.logger.info(f"Handling investigation task request for thread {message.thread_id}")
                return await self._handle_investigate_request(message)
            else:
                # error for invalid task data
                error_response = CoralMessage(
                    message_id=self._generate_message_id(),
                    thread_id=message.thread_id,
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type=MessageType.ERROR,
                    payload={
                        "error": "Invalid task data for investigator - missing anomalies, features, or investigation data",
                        "received_data": list(payload.keys())
                    },
                    timestamp=datetime.now(timezone.utc),
                    priority=TaskPriority.HIGH,
                    correlation_id=message.message_id
                )
                return error_response
                
        except Exception as e:
            self.logger.error(f"Task request handling failed: {str(e)}")
            error_response = CoralMessage(
                message_id=self._generate_message_id(),
                thread_id=message.thread_id,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.ERROR,
                payload={"error": str(e)},
                timestamp=datetime.now(timezone.utc),
                priority=TaskPriority.HIGH,
                correlation_id=message.message_id
            )
            return error_response

    async def receive_message(self, message: CoralMessage) -> Optional[CoralMessage]:
        """Handle incoming Coral message - override to add specific logging"""
        self.logger.debug(f"Received message type: {message.message_type.value} from {message.sender_id}")
        return await super().receive_message(message)

    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main investigation pipeline with Ollama LLM integration
        """
        try:
            # investigation parameters
            investigation_mode = task_data.get("investigation_mode", "root_cause")
            anomalies = task_data.get("anomalies", [])
            features = task_data.get("features", {})
            metadata = task_data.get("metadata", {})
            raw_data = task_data.get("raw_data", "")
            file_content = task_data.get("file_content", {})

            self.logger.info(f"Starting {investigation_mode} investigation with {len(anomalies)} anomalies")

            # Step 1: investigation context
            investigation_context = self._prepare_investigation_context(
                anomalies, features, metadata, raw_data, file_content
            )

            # Step 2: LLM reasoning
            llm_analysis = await self._perform_llm_analysis(investigation_context, investigation_mode)

            # Step 3: visualizations if data available
            visualizations = await self._generate_visualizations(anomalies, features, file_content)

            # Step 4: final results
            final_results = self._compile_investigation_results(
                llm_analysis, investigation_mode, len(anomalies), visualizations
            )

            return final_results
            
        except Exception as e:
            self.logger.error(f"Investigation processing failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "diagnosis": "Investigation failed due to technical error",
                "causes": ["System error during analysis"],
                "recommendations": ["Retry investigation", "Check system configuration"],
                "confidence": 0.0,
                "reasoning": f"Error during investigation: {str(e)}",
                "patterns_detected": [],
                "risk_assessment": {"level": "unknown", "factors": []}
            }

    def _prepare_investigation_context(self, anomalies: List[Dict], features: Dict[str, Any], 
                                        metadata: Dict[str, Any], raw_data: str, 
                                        file_content: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare comprehensive context for LLM analysis"""
        
        context = {
            "investigation_summary": {
                "anomaly_count": len(anomalies),
                "feature_categories": list(features.keys()),
                "metadata_context": metadata,
                "data_sources": []
            },
            "detailed_analysis": {
                "anomalies": anomalies[:10],  
                "key_features": self._extract_key_features(features),
                "temporal_patterns": self._analyze_temporal_patterns(anomalies),
                "severity_distribution": self._analyze_severity_distribution(anomalies)
            }
        }

        # file content 
        if file_content:
            context["file_analysis"] = {
                "data_type": file_content.get("data_type", "unknown"),
                "sample_data": self._extract_file_sample(file_content),
                "data_quality": self._assess_data_quality(file_content)
            }

        # raw data summary
        if raw_data:
            context["raw_data_summary"] = {
                "length": len(raw_data),
                "content_preview": raw_data[:500] + "..." if len(raw_data) > 500 else raw_data
            }

        return context

    async def _perform_llm_analysis(self, context: Dict[str, Any], 
                                    investigation_mode: str) -> Dict[str, Any]:
        """Perform LLM analysis using Ollama"""
        
        mode_config = self.investigation_modes.get(investigation_mode, self.investigation_modes["root_cause"])
        
        system_prompt = f"""
        {mode_config['system_prompt']}
        
        RESPONSE FORMAT REQUIREMENTS:
        - Use structured markdown sections
        - Include confidence assessments
        - Provide actionable recommendations
        - Highlight critical findings
        - Use bullet points for clarity
        
        STRUCTURED SECTIONS:
        # Investigation Summary
        # Key Findings  
        # Root Cause Analysis
        # Risk Assessment
        # Recommendations
        # Confidence Level
        """

        user_prompt = f"""
        INVESTIGATION MODE: {mode_config['name']}
        CONTEXT: {json.dumps(context, indent=2)}
        
        Please analyze this investigation context and provide a comprehensive report.
        Focus on patterns, causes, and actionable insights specific to {investigation_mode}.
        """

        try:
            self.logger.info(f"Calling Ollama with model: {self.llm_model}")
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "top_k": 40
                }
            )
            
            llm_response = response["message"]["content"]
            return self._parse_llm_response(llm_response, investigation_mode)
            
        except Exception as e:
            self.logger.error(f"Ollama analysis failed: {e}")
            return self._fallback_analysis(context, investigation_mode)

    def _parse_llm_response(self, llm_response: str, investigation_mode: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        
        # Basic parsing - in production, you'd want more sophisticated parsing
        return {
            "raw_response": llm_response,
            "summary": self._extract_section(llm_response, "Investigation Summary"),
            "findings": self._extract_section(llm_response, "Key Findings"),
            "root_causes": self._extract_section(llm_response, "Root Cause Analysis"),
            "risk_assessment": self._extract_section(llm_response, "Risk Assessment"),
            "recommendations": self._extract_section(llm_response, "Recommendations"),
            "confidence": self._extract_confidence(llm_response)
        }

    async def _generate_visualizations(self, anomalies: List[Dict], features: Dict[str, Any], 
                                     file_content: Dict[str, Any]) -> Dict[str, Any]:
        """Generate investigation visualizations"""
        visualizations = {}
        
        try:
            # anomaly severity distribution
            if anomalies:
                severity_data = [{"anomaly": i, "severity": a.get("severity", 0)} 
                               for i, a in enumerate(anomalies)]
                df_severity = pd.DataFrame(severity_data)
                
                # plotly chart data 
                fig_severity = px.histogram(df_severity, x="severity", 
                                          title="Anomaly Severity Distribution")
                visualizations["severity_distribution"] = "available"
            
            #correlation if enough numeric features
            if features and len(features) > 1:
                numeric_features = {k: v for k, v in features.items() 
                                  if isinstance(v, (int, float))}
                if len(numeric_features) >= 2:
                    visualizations["feature_correlation"] = "available"
                    
        except Exception as e:
            self.logger.warning(f"Visualization generation failed: {e}")
            
        return visualizations

    def _compile_investigation_results(self, llm_analysis: Dict[str, Any], 
                                        investigation_mode: str, anomaly_count: int,
                                        visualizations: Dict[str, Any]) -> Dict[str, Any]:
        """Compile final investigation results"""
        
        mode_config = self.investigation_modes.get(investigation_mode, self.investigation_modes["root_cause"])
        
        return {
            "status": "success",
            "investigation_mode": investigation_mode,
            "mode_display_name": mode_config["name"],
            "mode_description": mode_config["description"],
            "mode_color": mode_config["color"],
            "diagnosis": llm_analysis.get("summary", "Analysis completed"),
            "causes": self._parse_list_items(llm_analysis.get("root_causes", "")),
            "recommendations": self._parse_list_items(llm_analysis.get("recommendations", "")),
            "confidence": llm_analysis.get("confidence", 0.7),
            "reasoning": llm_analysis.get("raw_response", ""),
            "patterns_detected": self._extract_patterns(llm_analysis.get("findings", "")),
            "risk_assessment": {
                "level": self._assess_risk_level(llm_analysis.get("risk_assessment", "")),
                "factors": self._parse_list_items(llm_analysis.get("risk_assessment", ""))
            },
            "visualizations": visualizations,
            "metadata": {
                "agent_id": self.agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "llm_model": self.llm_model,
                "anomalies_processed": anomaly_count,
                "investigation_depth": "comprehensive"
            }
        }

    # Helper methods for data processing
    def _extract_key_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract most relevant features for investigation"""
        key_features = {}
        for key, value in features.items():
            if isinstance(value, (int, float)):
                key_features[key] = round(float(value), 4)
            elif isinstance(value, list) and len(value) > 0:
                key_features[key] = value[:3]  
            else:
                key_features[key] = str(value)[:100]  
        return key_features

    def _analyze_temporal_patterns(self, anomalies: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal patterns in anomalies"""
        if not anomalies:
            return {}
            
        times = [a.get("start_time", 0) for a in anomalies if "start_time" in a]
        if times:
            return {
                "time_range": f"{min(times):.1f}s - {max(times):.1f}s",
                "frequency": len(times) / (max(times) - min(times)) if max(times) > min(times) else 0,
                "clustering": len(times) / 10  
            }
        return {}

    def _analyze_severity_distribution(self, anomalies: List[Dict]) -> Dict[str, Any]:
        """Analyze severity distribution of anomalies"""
        if not anomalies:
            return {}
            
        severities = [a.get("severity", 0) for a in anomalies]
        return {
            "mean_severity": sum(severities) / len(severities),
            "max_severity": max(severities),
            "critical_count": len([s for s in severities if s > 0.8])
        }

    def _extract_file_sample(self, file_content: Dict[str, Any]) -> Any:
        """Extract sample data from file content"""
        if "dataframe" in file_content:
            return file_content["dataframe"].head(3).to_dict()
        elif "raw_data" in file_content:
            return str(file_content["raw_data"])[:200]
        return "No sample available"

    def _assess_data_quality(self, file_content: Dict[str, Any]) -> str:
        """Assess quality of file data"""
        if "error" in file_content:
            return "poor"
        elif "dataframe" in file_content and len(file_content["dataframe"]) > 0:
            return "good"
        elif "raw_data" in file_content and len(file_content["raw_data"]) > 100:
            return "fair"
        return "unknown"

    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract specific section from LLM response"""
        lines = text.split('\n')
        in_section = False
        section_lines = []
        
        for line in lines:
            if line.strip().startswith('#') and section_name.lower() in line.lower():
                in_section = True
                continue
            elif in_section and line.strip().startswith('#'):
                break
            elif in_section:
                section_lines.append(line.strip())
                
        return '\n'.join(section_lines) if section_lines else "Not specified"

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence level from LLM response"""
        import re
        confidence_patterns = [
            r'confidence[:\s]*(\d+\.?\d*)%',
            r'confidence[:\s]*(\d+\.?\d*)/10',
            r'confidence[:\s]*(\d+\.?\d*)\s*out of 10'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, text.lower())
            if match:
                confidence = float(match.group(1))
                if '%' in pattern:
                    return confidence / 100
                elif '/10' in pattern or 'out of 10' in pattern:
                    return confidence / 10
                    
        return 0.7  # Default confidence

    def _parse_list_items(self, text: str) -> List[str]:
        """Parse bullet points or numbered lists from text"""
        import re
        items = re.findall(r'[-*•]\s*(.+?)(?=\n[-*•]|\n#|\n\n|$)', text, re.DOTALL)
        if not items:
            # Try numbered lists
            items = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\n#|\n\n|$)', text, re.DOTALL)
        return [item.strip() for item in items if item.strip()]

    def _extract_patterns(self, findings: str) -> List[str]:
        """Extract patterns from findings text"""
        patterns = []
        if "pattern" in findings.lower():
            lines = findings.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['pattern', 'trend', 'correlation', 'cluster']):
                    patterns.append(line.strip())
        return patterns[:5]  
    def _assess_risk_level(self, risk_text: str) -> str:
        """Assess risk level from risk assessment text"""
        risk_text_lower = risk_text.lower()
        if any(word in risk_text_lower for word in ['critical', 'severe', 'emergency']):
            return "critical"
        elif any(word in risk_text_lower for word in ['high', 'elevated', 'serious']):
            return "high"
        elif any(word in risk_text_lower for word in ['medium', 'moderate']):
            return "medium"
        elif any(word in risk_text_lower for word in ['low', 'minor']):
            return "low"
        return "unknown"

    def _fallback_analysis(self, context: Dict[str, Any], investigation_mode: str) -> Dict[str, Any]:
        """Provide fallback analysis when LLM fails"""
        return {
            "raw_response": f"## Fallback Analysis\n\nInvestigation mode: {investigation_mode}\nAnomalies processed: {context['investigation_summary']['anomaly_count']}\n\n*Note: LLM analysis unavailable, using rule-based fallback.*",
            "summary": f"Processed {context['investigation_summary']['anomaly_count']} anomalies in {investigation_mode} mode",
            "findings": "Limited analysis available due to system constraints",
            "root_causes": ["System limitations prevented deep analysis"],
            "risk_assessment": "Risk level unknown - manual review recommended",
            "recommendations": ["Verify data quality", "Retry analysis", "Consult domain expert"],
            "confidence": 0.5
        }


# Streamlit-compatible synchronous wrapper
class StreamlitInvestigator:
    """Synchronous wrapper for Streamlit compatibility"""
    
    def __init__(self, agent_id: str = "streamlit_investigator"):
        self.agent = InvestigatorAgent(agent_id)
        self.investigation_history = []
    
    def investigate(self, investigation_mode: str, input_data: str = "", 
                   file_content: Dict[str, Any] = None) -> Dict[str, Any]:
        """Synchronous investigation for Streamlit"""
        
        task_data = {
            "investigation_mode": investigation_mode,
            "raw_data": input_data,
            "file_content": file_content or {},
            "metadata": {
                "source": "streamlit_app",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        
        # async function in sync context
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.agent.process_task(task_data))
            loop.close()
            
            
            self.investigation_history.append({
                "mode": investigation_mode,
                "input": input_data,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "diagnosis": "Investigation failed",
                "reasoning": f"Error during investigation: {str(e)}"
            }


# Test function
async def test_investigator():
    """Test the investigator agent"""
    investigator = InvestigatorAgent("test_investigator_001")
    
    # Test data
    test_data = {
        "investigation_mode": "root_cause",
        "anomalies": [
            {
                "type": "performance_degradation",
                "severity": 0.8,
                "start_time": 100.5,
                "description": "System response time increased by 300%"
            }
        ],
        "features": {
            "response_time": 2.5,
            "error_rate": 0.15,
            "throughput": 45.2
        },
        "metadata": {
            "system": "web_application",
            "timeframe": "last_24_hours"
        },
        "raw_data": "System logs showing increased latency and error rates"
    }
    
    result = await investigator.process_task(test_data)
    print("Investigation Result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(test_investigator())
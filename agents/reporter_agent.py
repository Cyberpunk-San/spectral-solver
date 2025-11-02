# agents/reporter_agent.py
import asyncio
import logging
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
from coral_s import CoralBaseAgent, AgentCapability, MessageType, CoralMessage, TaskPriority, AgentStatus


@dataclass
class Report:
    """Structured report for different audiences"""
    report_id: str
    thread_id: str
    created_at: datetime
    audience: str  # engineer, manager, maintenance, executive
    title: str
    summary: str
    findings: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    urgency: str  # critical, high, medium, low
    confidence: float
    artifacts: Dict[str, str]


class ReporterAgent(CoralBaseAgent):
    """
    Reporter Agent - Generates actionable reports for different audiences
    Converts technical findings into business-ready deliverables
    """

    def __init__(self, agent_id: str = "reporter_001"):
        capabilities = [
            AgentCapability(
                name="report_generation",
                version="1.0",
                description="Multi-audience report generation with actionable insights",
                input_schema={
                    "type": "object",
                    "properties": {
                        "analyzer_results": {"type": "object", "description": "Analyzer agent findings"},
                        "investigator_results": {"type": "object", "description": "Investigator agent diagnosis"},
                        "metadata": {"type": "object", "description": "Context and metadata"},
                        "audience": {"type": "string", "enum": ["engineer", "manager", "maintenance", "executive"]}
                    },
                    "required": ["analyzer_results", "investigator_results"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "report": {"type": "object", "description": "Structured report"},
                        "artifacts": {"type": "object", "description": "Generated files and URLs"},
                        "alert_level": {"type": "string", "description": "Urgency indicator"},
                        "deliverables": {"type": "object", "description": "Formatted outputs"}
                    }
                },
                performance_metrics={"report_quality": 0.92, "generation_time": "30s"}
            )
        ]

        super().__init__(agent_id, "Multi-Format Reporter", "reporter", capabilities)

        self.report_templates = {
            "engineer": {
                "sections": ["Technical Summary", "Anomaly Details", "Signal Analysis", "Root Cause", "Immediate Actions"],
                "depth": "detailed",
                "include_technical": True
            },
            "maintenance": {
                "sections": ["Problem Summary", "Required Actions", "Parts Needed", "Time Estimate", "Safety Notes"],
                "depth": "actionable", 
                "include_technical": False
            },
            "manager": {
                "sections": ["Executive Summary", "Business Impact", "Recommended Response", "Cost Implications", "Timeline"],
                "depth": "strategic",
                "include_technical": False
            },
            "executive": {
                "sections": ["Situation Overview", "Financial Impact", "Strategic Recommendations", "Risk Assessment"],
                "depth": "high-level",
                "include_technical": False
            }
        }

        self.reports_dir = "reports"
        os.makedirs(self.reports_dir, exist_ok=True)

        self.logger = logging.getLogger(f"reporter.{agent_id}")

    def _register_message_handlers(self):
        """Register message handlers for reporter-specific messages"""
        handlers = super()._register_message_handlers()
        handlers.update({
            MessageType.INVESTIGATE_RESULT: self._handle_investigate_result,
            MessageType.REPORT_REQUEST: self._handle_report_request,
            MessageType.TASK_REQUEST: self._handle_task_request,
        })
        return handlers

    async def _handle_investigate_result(self, message: CoralMessage) -> Optional[CoralMessage]:
        """Automatically generate reports when investigation completes"""
        try:
            self.logger.info(f"Generating report for thread {message.thread_id}")
            
            await self.update_status(AgentStatus.BUSY, 0.7)

            # Generate reports for different audiences
            report_results = await self.process_task(message.payload)

            # Create response message
            response = CoralMessage(
                message_id=self._generate_message_id(),
                thread_id=message.thread_id,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.REPORT_RESULT,
                payload=report_results,
                timestamp=datetime.now(timezone.utc),
                priority=message.priority,
                correlation_id=message.message_id
            )

            await self.update_status(AgentStatus.ONLINE, 0.3)

            self.logger.info(f"Report generation completed for thread {message.thread_id}")
            return response

        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            await self.update_status(AgentStatus.ERROR, 0.5)
            
            error_response = CoralMessage(
                message_id=self._generate_message_id(),
                thread_id=message.thread_id,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.ERROR,
                payload={"error": str(e), "status": "failed"},
                timestamp=datetime.now(timezone.utc),
                priority=TaskPriority.HIGH,
                correlation_id=message.message_id
            )
            return error_response

    async def _handle_report_request(self, message: CoralMessage) -> Optional[CoralMessage]:
        """Handle direct report requests"""
        try:
            self.logger.info(f"Handling direct report request for thread {message.thread_id}")
            
            await self.update_status(AgentStatus.BUSY, 0.8)

            # Generate reports from request payload
            report_results = await self.process_task(message.payload)

            response = CoralMessage(
                message_id=self._generate_message_id(),
                thread_id=message.thread_id,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.REPORT_RESULT,
                payload=report_results,
                timestamp=datetime.now(timezone.utc),
                priority=message.priority,
                correlation_id=message.message_id
            )

            await self.update_status(AgentStatus.ONLINE, 0.2)
            self.logger.info(f"Direct report generation completed for thread {message.thread_id}")
            return response

        except Exception as e:
            self.logger.error(f"Direct report generation failed: {str(e)}")
            await self.update_status(AgentStatus.ERROR, 0.5)
            
            error_response = CoralMessage(
                message_id=self._generate_message_id(),
                thread_id=message.thread_id,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.ERROR,
                payload={"error": str(e), "status": "failed"},
                timestamp=datetime.now(timezone.utc),
                priority=TaskPriority.HIGH,
                correlation_id=message.message_id
            )
            return error_response

    async def _handle_task_request(self, message: CoralMessage) -> Optional[CoralMessage]:
        """Handle generic task requests"""
        try:
            payload = message.payload
            
            # Check if this is a report generation task
            if (payload.get("analyzer_results") or 
                payload.get("investigator_results") or
                payload.get("audience")):
                
                self.logger.info(f"Handling report generation task for thread {message.thread_id}")
                return await self._handle_report_request(message)
            else:
                # Return error for invalid task data
                error_response = CoralMessage(
                    message_id=self._generate_message_id(),
                    thread_id=message.thread_id,
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type=MessageType.ERROR,
                    payload={
                        "error": "Invalid task data for reporter - missing analyzer_results, investigator_results, or audience",
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
        Generate comprehensive reports for multiple audiences
        """
        try:
            analyzer_results = task_data.get("analyzer_results", {})
            investigator_results = task_data.get("investigator_results", {})
            metadata = task_data.get("metadata", {})
            specific_audience = task_data.get("audience")

            self.logger.info(f"Generating reports from {len(analyzer_results.get('anomalies', []))} anomalies")

            #  reports for different audiences
            reports = {}
            
            if specific_audience:
                # specified audience
                reports[specific_audience] = await self._generate_single_report(
                    specific_audience, analyzer_results, investigator_results, metadata
                )
            else:
                # \all audiences
                for audience in self.report_templates.keys():
                    reports[audience] = await self._generate_single_report(
                        audience, analyzer_results, investigator_results, metadata
                    )

            #  alert if high severity
            alert_level = self._determine_alert_level(analyzer_results, investigator_results)

            # deliverables
            deliverables = await self._create_deliverables(reports, alert_level, metadata)

            return {
                "status": "success",
                "reports": reports,
                "alert_level": alert_level,
                "deliverables": deliverables,
                "metadata": {
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "audiences_generated": list(reports.keys())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "reports": {},
                "alert_level": "unknown",
                "deliverables": {}
            }

    async def _generate_single_report(self, audience: str, analyzer_results: Dict[str, Any],
                                    investigator_results: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a report for a specific audience"""
        template = self.report_templates[audience]
        report_id = f"report_{audience}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract key information
        anomalies = analyzer_results.get("anomalies", [])
        features = analyzer_results.get("features", {})
        diagnosis = investigator_results.get("diagnosis", "No specific diagnosis")
        causes = investigator_results.get("causes", [])
        recommendations = investigator_results.get("recommendations", [])
        confidence = investigator_results.get("confidence", 0.0)

        #\ audience-specific content
        if audience == "engineer":
            report_content = self._generate_engineer_report(anomalies, features, diagnosis, causes, recommendations, metadata)
        elif audience == "maintenance":
            report_content = self._generate_maintenance_report(anomalies, diagnosis, causes, recommendations, metadata)
        elif audience == "manager":
            report_content = self._generate_manager_report(anomalies, diagnosis, recommendations, confidence, metadata)
        elif audience == "executive":
            report_content = self._generate_executive_report(diagnosis, recommendations, confidence, metadata)
        else:
            report_content = self._generate_default_report(anomalies, diagnosis, recommendations, metadata)

        # report object
        report = Report(
            report_id=report_id,
            thread_id=metadata.get("thread_id", "unknown"),
            created_at=datetime.now(timezone.utc),
            audience=audience,
            title=report_content["title"],
            summary=report_content["summary"],
            findings=report_content["findings"],
            recommendations=report_content["recommendations"],
            urgency=report_content["urgency"],
            confidence=confidence,
            artifacts={}
        )

        # artifacts
        artifacts = await self._generate_report_artifacts(report, analyzer_results, audience)
        report.artifacts = artifacts

        # report
        report_path = self._save_report(report, report_content)

        return {
            "report_id": report_id,
            "title": report.title,
            "summary": report.summary,
            "findings": report.findings,
            "recommendations": report.recommendations,
            "urgency": report.urgency,
            "confidence": report.confidence,
            "artifacts": report.artifacts,
            "report_path": report_path
        }

    def _generate_engineer_report(self, anomalies: List[Dict], features: Dict[str, Any],
                                diagnosis: str, causes: List[str], recommendations: List[str],
                                metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed technical report for engineers"""
        
        # Technical findings
        findings = []
        for anomaly in anomalies:
            findings.append({
                "type": anomaly.get("type", "Unknown"),
                "frequency": anomaly.get("frequency", 0),
                "severity": anomaly.get("severity", 0),
                "description": anomaly.get("description", ""),
                "confidence": anomaly.get("confidence", 0)
            })

        # Feature analysis
        feature_analysis = {
            "rms_vibration": features.get("rms", 0),
            "kurtosis": features.get("kurtosis", 0),
            "crest_factor": features.get("crest_factor", 0),
            "dominant_frequencies": [pf["frequency"] for pf in features.get("peak_frequencies", [])[:3]]
        }

        return {
            "title": f"Technical Diagnostic Report - {metadata.get('machine_id', 'Unknown Machine')}",
            "summary": f"Analysis detected {len(anomalies)} anomalies. Primary diagnosis: {diagnosis}",
            "findings": [
                {"section": "Anomaly Analysis", "content": findings},
                {"section": "Signal Features", "content": feature_analysis},
                {"section": "Root Cause Analysis", "content": causes},
                {"section": "Technical Details", "content": features}
            ],
            "recommendations": [{"action": rec, "priority": "high"} for rec in recommendations],
            "urgency": self._calculate_urgency(anomalies)
        }

    def _generate_maintenance_report(self, anomalies: List[Dict], diagnosis: str,
                                  causes: List[str], recommendations: List[str],
                                  metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable report for maintenance teams"""
        
        #  maintenance actions
        maintenance_actions = []
        for rec in recommendations:
            if "inspect" in rec.lower() or "check" in rec.lower():
                priority = "high"
            elif "replace" in rec.lower() or "repair" in rec.lower():
                priority = "critical"
            else:
                priority = "medium"
            
            maintenance_actions.append({
                "action": rec,
                "priority": priority,
                "estimated_time": "1-2 hours", 
                "tools_required": ["Vibration meter", "Basic hand tools"]
            })

        return {
            "title": f"Maintenance Work Order - {metadata.get('machine_id', 'Unknown Machine')}",
            "summary": f"Required: {diagnosis}. {len(maintenance_actions)} maintenance actions identified.",
            "findings": [
                {"section": "Problem Description", "content": diagnosis},
                {"section": "Likely Causes", "content": causes},
                {"section": "Anomalies Detected", "content": f"{len(anomalies)} issues found"}
            ],
            "recommendations": maintenance_actions,
            "urgency": self._calculate_urgency(anomalies)
        }

    def _generate_manager_report(self, anomalies: List[Dict], diagnosis: str,
                              recommendations: List[str], confidence: float,
                              metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate business-focused report for managers"""
        
        # business impact
        highest_severity = max([a.get("severity", 0) for a in anomalies]) if anomalies else 0
        if highest_severity > 0.8:
            impact = "High - Potential downtime risk"
            cost_implication = "$$$ - Urgent repair needed"
        elif highest_severity > 0.5:
            impact = "Medium - Schedule maintenance soon"
            cost_implication = "$$ - Planned maintenance"
        else:
            impact = "Low - Monitor condition"
            cost_implication = "$ - Routine check"

        return {
            "title": f"Equipment Health Report - {metadata.get('machine_id', 'Unknown Machine')}",
            "summary": f"Diagnosis: {diagnosis}. Business Impact: {impact}",
            "findings": [
                {"section": "Executive Summary", "content": diagnosis},
                {"section": "Business Impact", "content": impact},
                {"section": "Financial Implications", "content": cost_implication},
                {"section": "Confidence Level", "content": f"{confidence:.1%}"}
            ],
            "recommendations": [{"action": rec, "business_case": "Preventive maintenance"} for rec in recommendations],
            "urgency": self._calculate_urgency(anomalies)
        }

    def _generate_executive_report(self, diagnosis: str, recommendations: List[str],
                                 confidence: float, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level report for executives"""
        
        return {
            "title": f"Asset Health Briefing - {metadata.get('machine_id', 'Unknown Machine')}",
            "summary": f"Equipment requires attention: {diagnosis}",
            "findings": [
                {"section": "Situation", "content": diagnosis},
                {"section": "Confidence", "content": f"{confidence:.1%} certainty"},
                {"section": "Strategic Importance", "content": "Maintain operational continuity"}
            ],
            "recommendations": [
                {"action": rec, "rationale": "Risk mitigation"} for rec in recommendations[:2]  # Top 2 only
            ],
            "urgency": "medium"  \
        }

    def _generate_default_report(self, anomalies: List[Dict], diagnosis: str,
                               recommendations: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a default report format"""
        return {
            "title": f"Analysis Report - {metadata.get('machine_id', 'Unknown Machine')}",
            "summary": f"Found {len(anomalies)} issues. Diagnosis: {diagnosis}",
            "findings": [
                {"section": "Detection Summary", "content": f"{len(anomalies)} anomalies detected"},
                {"section": "Primary Diagnosis", "content": diagnosis}
            ],
            "recommendations": [{"action": rec} for rec in recommendations],
            "urgency": self._calculate_urgency(anomalies)
        }

    def _calculate_urgency(self, anomalies: List[Dict]) -> str:
        """Calculate urgency based on anomaly severity"""
        if not anomalies:
            return "low"
        
        max_severity = max([a.get("severity", 0) for a in anomalies])
        
        if max_severity > 0.8:
            return "critical"
        elif max_severity > 0.6:
            return "high"
        elif max_severity > 0.4:
            return "medium"
        else:
            return "low"

    def _determine_alert_level(self, analyzer_results: Dict[str, Any], investigator_results: Dict[str, Any]) -> str:
        """Determine overall alert level for the system"""
        anomalies = analyzer_results.get("anomalies", [])
        confidence = investigator_results.get("confidence", 0)
        
        if not anomalies:
            return "info"
        
        max_severity = max([a.get("severity", 0) for a in anomalies])
        
        if max_severity > 0.8 and confidence > 0.8:
            return "critical"
        elif max_severity > 0.6 and confidence > 0.7:
            return "warning"
        elif max_severity > 0.4:
            return "info"
        else:
            return "normal"

    async def _generate_report_artifacts(self, report: Report, analyzer_results: Dict[str, Any],
                                       audience: str) -> Dict[str, str]:
        """Generate visual artifacts for the report"""
        artifacts = {}
        
        try:
            # technical audiences, include more artifacts
            if audience in ["engineer", "maintenance"]:
                # summary chart
                chart_path = self._create_summary_chart(analyzer_results, report.report_id)
                if chart_path:
                    artifacts["summary_chart"] = chart_path
                
                # analyzer artifacts if available
                if "artifacts" in analyzer_results:
                    artifacts.update(analyzer_results["artifacts"])

            # generate JSON report
            json_path = os.path.join(self.reports_dir, f"{report.report_id}.json")
            with open(json_path, 'w') as f:
                json.dump({
                    "report_id": report.report_id,
                    "audience": report.audience,
                    "title": report.title,
                    "summary": report.summary,
                    "findings": report.findings,
                    "recommendations": report.recommendations,
                    "urgency": report.urgency,
                    "confidence": report.confidence,
                    "timestamp": report.created_at.isoformat()
                }, f, indent=2)
            
            artifacts["json_report"] = json_path

        except Exception as e:
            self.logger.error(f"Artifact generation failed: {e}")

        return artifacts

    def _create_summary_chart(self, analyzer_results: Dict[str, Any], report_id: str) -> Optional[str]:
        """Create a summary visualization chart"""
        try:
            anomalies = analyzer_results.get("anomalies", [])
            if not anomalies:
                return None

            # \severity distribution chart
            severities = [a.get("severity", 0) for a in anomalies]
            types = [a.get("type", "Unknown") for a in anomalies]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(range(len(anomalies)), severities, color=['#ff6b6b' if s > 0.7 else '#ffa726' if s > 0.4 else '#66bb6a' for s in severities])
            ax.set_xlabel('Anomalies')
            ax.set_ylabel('Severity')
            ax.set_title('Anomaly Severity Distribution')
            ax.set_xticks(range(len(anomalies)))
            ax.set_xticklabels(types, rotation=45, ha='right')
            
            plt.tight_layout()
            chart_path = os.path.join(self.reports_dir, f"{report_id}_chart.png")
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return chart_path

        except Exception as e:
            self.logger.error(f"Chart creation failed: {e}")
            return None

    def _save_report(self, report: Report, content: Dict[str, Any]) -> str:
        """Save report to file system"""
        report_path = os.path.join(self.reports_dir, f"{report.report_id}_full.json")
        
        full_report = {
            "metadata": {
                "report_id": report.report_id,
                "thread_id": report.thread_id,
                "audience": report.audience,
                "created_at": report.created_at.isoformat(),
                "urgency": report.urgency,
                "confidence": report.confidence
            },
            "content": content,
            "artifacts": report.artifacts
        }
        
        with open(report_path, 'w') as f:
            json.dump(full_report, f, indent=2)
        
        return report_path

    async def _create_deliverables(self, reports: Dict[str, Dict], alert_level: str,
                                 metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create final deliverables package"""
        deliverables = {
            "alert_notification": self._create_alert_notification(alert_level, reports, metadata),
            "summary_dashboard": self._create_summary_dashboard(reports),
            "api_endpoints": {
                "reports": f"/api/reports/{metadata.get('thread_id', 'unknown')}",
                "alerts": f"/api/alerts/{metadata.get('thread_id', 'unknown')}",
                "artifacts": f"/api/artifacts/{metadata.get('thread_id', 'unknown')}"
            }
        }

        # consolidated report
        consolidated = self._create_consolidated_report(reports, metadata)
        deliverables["consolidated_report"] = consolidated

        return deliverables

    def _create_alert_notification(self, alert_level: str, reports: Dict[str, Dict],
                                 metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create alert notification for immediate action"""
        engineer_report = reports.get("engineer", {})
        maintenance_report = reports.get("maintenance", {})
        
        return {
            "level": alert_level,
            "message": f"Equipment alert: {engineer_report.get('summary', 'Anomalies detected')}",
            "machine": metadata.get("machine_id", "Unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actions": maintenance_report.get("recommendations", [])[:3],  # Top 3 actions
            "urgency": engineer_report.get("urgency", "medium")
        }

    def _create_summary_dashboard(self, reports: Dict[str, Dict]) -> Dict[str, Any]:
        """Create summary data for dashboard display"""
        engineer_report = reports.get("engineer", {})
        
        return {
            "alert_level": engineer_report.get("urgency", "low"),
            "confidence": engineer_report.get("confidence", 0),
            "anomaly_count": len(engineer_report.get("findings", [{}])[0].get("content", []) if engineer_report.get("findings") else []),
            "primary_issue": engineer_report.get("summary", "No issues").split(":")[-1].strip(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def _create_consolidated_report(self, reports: Dict[str, Dict], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a consolidated report combining all audiences"""
        return {
            "machine_id": metadata.get("machine_id", "Unknown"),
            "analysis_date": datetime.now(timezone.utc).isoformat(),
            "reports_generated": list(reports.keys()),
            "executive_summary": reports.get("executive", {}).get("summary", ""),
            "technical_summary": reports.get("engineer", {}).get("summary", ""),
            "maintenance_actions": [rec["action"] for rec in reports.get("maintenance", {}).get("recommendations", [])],
            "business_impact": reports.get("manager", {}).get("findings", [{}])[1].get("content", "Unknown") if reports.get("manager", {}).get("findings") else "Unknown"
        }


# Test function
async def test_reporter():
    """Test the reporter agent"""
    reporter = ReporterAgent("test_reporter_001")
    
    # Test data
    test_data = {
        "analyzer_results": {
            "anomalies": [
                {
                    "type": "bearing_outer_race_defect",
                    "frequency": 160.2,
                    "amplitude": 0.78,
                    "severity": 0.92,
                    "description": "BPFO detected at 160.2Hz",
                    "confidence": 0.85
                }
            ],
            "features": {
                "rms": 0.045,
                "kurtosis": 4.2,
                "crest_factor": 5.8,
                "peak_frequencies": [
                    {"frequency": 50.1, "magnitude": 0.15},
                    {"frequency": 160.2, "magnitude": 0.78}
                ]
            },
            "artifacts": {
                "spectrogram": "artifacts/spec_123.png",
                "spectrum": "artifacts/spectrum_123.png"
            }
        },
        "investigator_results": {
            "diagnosis": "Bearing outer race defect confirmed",
            "causes": ["Fatigue", "Contamination", "Improper installation"],
            "recommendations": [
                "Replace bearing within 24 hours",
                "Check lubrication system",
                "Verify proper installation"
            ],
            "confidence": 0.91,
            "reasoning": "Frequency match with BPFO formula"
        },
        "metadata": {
            "thread_id": "thread_test_123",
            "machine_id": "Motor-A",
            "rpm": 1750,
            "signal_type": "vibration"
        }
    }
    
    result = await reporter.process_task(test_data)
    print("Reporter Result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(test_reporter())
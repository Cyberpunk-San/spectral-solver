# main.py
import asyncio
import logging
import json
import numpy as np
from datetime import datetime, timezone

from coral_manager import get_coral_manager
from agents import UniversalAnalyzerAgent, InvestigatorAgent, ReporterAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Main function to run the Spectral Solver system"""
    logger = logging.getLogger("main")
    
    try:
        logger.info("Starting Spectral Solver System...")
        
        # Initialize Coral Manager
        coral_mgr = await get_coral_manager()
        logger.info("Coral Manager initialized")
        
        # Create and register agents
        analyzer = UniversalAnalyzerAgent("analyzer_001")
        investigator = InvestigatorAgent("investigator_001") 
        reporter = ReporterAgent("reporter_001")
        
        await coral_mgr.register_agent(analyzer)
        await coral_mgr.register_agent(investigator)
        await coral_mgr.register_agent(reporter)
        
        logger.info("All agents registered and ready")
        
        # Demo: Create a test analysis pipeline
        await run_demo_pipeline(coral_mgr, analyzer, investigator, reporter)
        
        # Keep the system running
        logger.info("Spectral Solver system is running. Press Ctrl+C to stop.")
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        if 'coral_mgr' in locals():
            await coral_mgr.shutdown()

async def run_demo_pipeline(coral_mgr, analyzer, investigator, reporter):
    """Run a demo pipeline with synthetic data"""
    logger = logging.getLogger("demo")
    
    # synthetic vibration data with bearing defect
    sample_rate = 10000
    duration = 2
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Normal vibration + bearing defect
    normal_vibration = 0.1 * np.sin(2 * np.pi * 50 * t)  # 50Hz rotation
    bearing_defect = 0.3 * np.sin(2 * np.pi * 160 * t)   # 160Hz defect
    noise = 0.02 * np.random.normal(size=len(t))
    
    signal = normal_vibration + bearing_defect + noise
    
    # Create thread
    thread_id = await coral_mgr.create_thread(
        created_by="demo_user",
        metadata={
            "purpose": "demo_bearing_analysis",
            "machine_id": "Demo-Motor-001", 
            "rpm": 1750,
            "signal_type": "vibration"
        }
    )
    
    logger.info(f"Demo pipeline started with thread: {thread_id}")
    
    # Run analysis through the pipeline
    task_data = {
        "file_content": json.dumps(signal.tolist()),
        "file_type": "json",
        "analysis_type": "comprehensive",
        "metadata": {
            "rpm": 1750,
            "machine_id": "Demo-Motor-001",
            "sample_rate": sample_rate,
            "signal_type": "vibration"
        }
    }
    
    try:
        # Step 1: Analysis
        analyzer_result = await analyzer.process_task(task_data)
        logger.info(f"Analyzer completed: {len(analyzer_result.get('anomalies', []))} anomalies found")
        
        # Step 2: Investigation  
        investigator_data = {
            "anomalies": analyzer_result.get("anomalies", []),
            "features": analyzer_result.get("features", {}),
            "metadata": task_data["metadata"]
        }
        investigator_result = await investigator.process_task(investigator_data)
        logger.info(f"Investigator completed: {investigator_result.get('diagnosis', 'Unknown')}")
        
        # Step 3: Reporting
        reporter_data = {
            "analyzer_results": analyzer_result,
            "investigator_results": investigator_result, 
            "metadata": {
                **task_data["metadata"],
                "thread_id": thread_id
            }
        }
        reporter_result = await reporter.process_task(reporter_data)
        logger.info(f"Reporter completed: {len(reporter_result.get('reports', {}))} reports generated")
        
        # Print final results
        print("\n" + "="*60)
        print("🎯 SPECTRAL SOLVER - DEMO RESULTS")
        print("="*60)
        
        if "reports" in reporter_result and "executive" in reporter_result["reports"]:
            executive_report = reporter_result["reports"]["executive"]
            print(f"Executive Summary: {executive_report.get('summary', 'No summary')}")
            print(f"Confidence: {executive_report.get('confidence', 0):.1%}")
            print(f"Urgency: {executive_report.get('urgency', 'unknown').upper()}")
        
        if "reports" in reporter_result and "maintenance" in reporter_result["reports"]:
            maintenance_report = reporter_result["reports"]["maintenance"] 
            print(f"\nMaintenance Actions:")
            recommendations = maintenance_report.get("recommendations", [])
            for i, action in enumerate(recommendations[:3], 1):
                if isinstance(action, dict):
                    print(f"  {i}. {action.get('action', 'Unknown action')} ({action.get('priority', 'medium')})")
                else:
                    print(f"  {i}. {action}")
        
        print(f"\n📁 Analysis completed for thread: {thread_id}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Demo pipeline failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
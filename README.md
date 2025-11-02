# Spectral Solver: Industrial AI Analysis Platform

**Advanced multi-agent AI system for industrial diagnostics**  
**Coral Protocol-powered agent coordination framework**  
**Automated anomaly detection, root cause analysis, and reporting**

## The Data Scientist's Challenge & Solution

**Problem**: Data scientists waste 60% of time on data preprocessing, visualization, and report generation instead of actual analysis. Industrial data requires specialized signal processing expertise that most data scientists lack.

**How Spectral Solver Helps**:
- **Universal Data Parser**: Automatically handles JSON, CSV, TXT with intelligent format detection
- **Pre-built Signal Processing**: Advanced FFT, wavelet analysis, statistical features out-of-the-box
- **Automated Visualization**: Production-ready charts, animations, and dashboards
- **LLM-Enhanced Analysis**: Local Ollama integration for intelligent diagnostics
- **Multi-format Reporting**: Technical, executive, and maintenance reports automatically generated
- **Pipeline Automation**: End-to-end analysis from raw data to actionable insights

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Streamlit UI  │◄──►│  Coral Manager   │◄──►│   Redis Broker   │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                              │
               ┌──────────────┼──────────────┐
               │              │              │
         ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
         │  Analyzer  │  │Investigator│  │  Reporter │
         │   Agent    │  │   Agent    │  │   Agent   │
         └────────────┘  └────────────┘  └───────────┘
```

## Coral Protocol: Enterprise-Grade Agent Framework

### Core Protocol Engine (`coral_s.py`)
- **Cryptographic Security**: RSA-2048 signed messages with Fernet encryption
- **Message Orchestration**: 20+ message types with priority-based routing
- **Agent Management**: Dynamic registration, capability discovery, load balancing
- **Thread-based Workflows**: Complete audit trails for complex analysis pipelines

### Multi-Agent Coordination (`coral_manager.py`)
- **Redis-Backed Messaging**: High-performance pub/sub communication
- **Automatic Pipeline Routing**: Analyzer → Investigator → Reporter workflow
- **Fault Tolerance**: Graceful error handling and agent recovery
- **Real-time Monitoring**: Agent status, performance metrics, system health

## Specialized Agent Ecosystem

### Universal Analyzer Agent (`analyzer_agent.py`)
**Industrial-Grade Signal Intelligence**
- **Format Agnostic**: Auto-detects JSON vibration data, CSV timeseries, raw numeric streams
- **Advanced Detection**: Z-score anomalies, spectral residuals, rolling statistics
- **Feature Extraction**: 15+ statistical features including kurtosis, crest factor, harmonic ratios
- **Visualization Engine**: Automated spectrograms, trend analysis, distribution plots

### AI Investigator Agent (`investigator_agent.py`) 
**LLM-Powered Diagnostic Reasoning**
- **Multi-Mode Analysis**: Root cause, anomaly correlation, threat tracing, budget investigation
- **Local Ollama Integration**: Phi-3 model with structured reasoning prompts
- **Confidence Scoring**: Probabilistic diagnosis with evidence chains
- **Risk Assessment**: Criticality evaluation with actionable prioritization

### Multi-Format Reporter Agent (`reporter_agent.py`)
**Audience-Specific Intelligence Delivery**
- **Four Audience Targets**: Engineering (technical), Maintenance (actionable), Management (strategic), Executive (briefing)
- **Automated Artifacts**: Charts, summaries, API endpoints, alert notifications
- **Business Alignment**: Cost implications, timeline estimates, resource requirements
- **Export Ready**: JSON, HTML, PNG, and integrated dashboard delivery

## Advanced Processing Capabilities

### Signal Processing Engine (`signal_processing.py`)
- **Real-time Analysis**: Rolling statistics with adaptive windowing
- **Frequency Domain**: FFT, power spectral density, harmonic analysis
- **Wavelet Processing**: Multi-resolution decomposition for transient detection
- **Quality Metrics**: Data integrity validation and missing data handling

### Visualization Suite (`visualization.py`, `animation_utils.py`)
- **Interactive Plotly**: Web-ready charts with hover diagnostics
- **Animated Analysis**: Fourier transform progressions, real-time detection sequences
- **Multi-panel Dashboards**: Comprehensive signal analysis in single view
- **Export Flexibility**: Static reports and interactive HTML for client delivery

## Technical Stack

**Core Framework**: Python 3.11+, Asyncio for high-concurrency agent operations  
**AI/ML**: Ollama (Phi-3), NumPy, SciPy, Scikit-learn compatible feature extraction  
**Data Processing**: Pandas, PyWavelets, advanced FFT algorithms  
**Visualization**: Plotly, Matplotlib with animation capabilities  
**Backend**: Redis for message brokering, Streamlit for web interface  
**Security**: Cryptography (RSA-2048, Fernet), secure message signing  
**Deployment**: Docker containerization, microservices architecture  

## Quick Start Deployment

```bash
# 1. Clone and setup
git clone https://github.com/Cyberpunk-San/spectral-solver/tree/main
cd spectral-solver-private
pip install -r requirements.txt

# 2. Start Redis (message broker)
docker run -d -p 6379:6379 redis:alpine

# 3. Launch Coral multi-agent system
python main.py

# 4. Start web dashboard (separate terminal)
cd dashboard
streamlit run app.py

# 5. Access interface at http://localhost:8501

**Demo Mode**: Use built-in synthetic data for immediate testing  
**Production Ready**: Upload CSV/JSON vibration data for real analysis  
**API Access**: Coral Protocol endpoints available for system integration  

The system automatically coordinates all agents - from data ingestion through analysis to executive reporting - delivering complete industrial diagnostics in under 30 seconds.
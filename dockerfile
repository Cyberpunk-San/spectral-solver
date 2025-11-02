FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including build tools and pkg-config
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p reports artifacts data

# Expose ports
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501 || exit 1

# Start command
CMD ["sh", "-c", "python main.py & streamlit run app/dashboard.py --server.port=8501 --server.address=0.0.0.0"]
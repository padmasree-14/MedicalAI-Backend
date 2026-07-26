FROM python:3.11-slim

# Install system dependencies needed for OpenCV, Matplotlib, and general builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency specifications
COPY requirements.txt .

# Install dependencies (use --no-cache-dir to minimize image size)
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend codebase
COPY server/ ./server/
COPY training/ ./training/
COPY model/ ./model/

# Create folders for uploads, reports, etc.
RUN mkdir -p server/uploads server/reports server/static/metrics

# Expose port
EXPOSE 8000

# Set python path
ENV PYTHONPATH=/app

# Command to start backend
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]

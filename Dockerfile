FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU version first to prevent heavy download failures
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install clean requirements
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

CMD ["python", "main.py"]
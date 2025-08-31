FROM python:3.10-slim

WORKDIR /app

# Install basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY handler.py .

CMD ["python", "-u", "handler.py"]

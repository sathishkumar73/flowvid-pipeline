FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code AND test input
COPY handler.py . 

CMD ["python", "-u", "handler.py"]

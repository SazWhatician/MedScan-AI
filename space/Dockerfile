FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# HuggingFace Spaces runs on port 7860
EXPOSE 7860

CMD ["python", "main.py"]

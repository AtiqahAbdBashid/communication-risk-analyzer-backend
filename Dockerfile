FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files including models
COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]

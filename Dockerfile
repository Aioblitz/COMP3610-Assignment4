# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY requirements.txt .
COPY app.py .
COPY models ./models

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
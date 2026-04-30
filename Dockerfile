# Use a stable Python version (3.14 is too new for some ML libs in Docker)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY backend_requirements.txt .
RUN pip install --no-cache-dir -r backend_requirements.txt

# Copy the application and the models we just generated
COPY ./app ./app
COPY ./models ./models

# Expose the port Uvicorn runs on
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
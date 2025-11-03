FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Copy dependencies file
COPY Mlops/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "Mlops.src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]

FROM agrigorev/zoomcamp-model:2025

# Set working directory
WORKDIR /app

# Copy pyproject.toml and install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy the FastAPI application
COPY app.py .

# Copy the trained model
# COPY pipeline_v1.bin .

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

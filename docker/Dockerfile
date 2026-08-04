FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app

EXPOSE 8000

# NOTE: run the training pipeline (scripts/run_pipeline.sh) at least once
# before/while building the image if you want models_registry/ pre-populated,
# or mount it as a volume so the container serves whatever the host trained.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

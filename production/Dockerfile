# =============================================================================
# TGCF-IDS Production Container
# Temporal Graph Contrastive Feature-Transformer Intrusion Detection System
# =============================================================================

FROM python:3.10-slim-bullseye AS base

# System dependencies for packet capture and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python production dependencies
COPY production/requirements.txt /app/production/requirements.txt
RUN pip install --no-cache-dir -r production/requirements.txt

# Copy research baseline dependencies (read-only model weights & preprocessor)
COPY data/processed/preprocessor.joblib /app/data/processed/preprocessor.joblib
COPY data/processed/label_mapping.json /app/data/processed/label_mapping.json
COPY data/processed/cat_mapping.json /app/data/processed/cat_mapping.json
COPY results/checkpoints/final_tgcf_ids_model.pt /app/results/checkpoints/final_tgcf_ids_model.pt

# Copy research source code for model definition loading
COPY src/ /app/src/

# Copy production system
COPY production/ /app/production/

# Expose REST & WebSocket port
EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default execution
ENTRYPOINT ["python", "-m", "production"]
CMD ["run", "--host", "0.0.0.0", "--port", "8000"]

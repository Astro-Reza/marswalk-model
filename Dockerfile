# Image for Google Cloud Batch: simulation shards (default) and check_walker.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEPRL_BASELINE_DIR=/app/baselines_DEPRL/myoLegWalk_20230514/myoLeg/

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch keeps the image small (the policy is a small MLP).
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
    || pip install torch==2.15.0

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY marswalk_sim ./marswalk_sim
COPY scripts ./scripts
COPY kinesis_backend ./kinesis_backend
COPY tests ./tests

# Bake the DEP-RL walking weights into the image so cloud tasks never hit
# Google Drive download limits.
RUN python -c "from marswalk_sim.policies import ensure_deprl_weights; ensure_deprl_weights()"

# Fail the build early if the env or patch code is broken.
RUN python -m pytest -q tests -p no:cacheprovider

ENTRYPOINT ["python"]
CMD ["scripts/generate.py", "--help"]

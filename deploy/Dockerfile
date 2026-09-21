# Fake-review forensics backend.
#
# Built to stay small: the CPU-only PyTorch wheels are ~1.5 GB instead of the ~2.5 GB CUDA
# ones, and none of this runs on a GPU. Measured footprint of the running server: about
# 1.1 GB resident, 2.3 GB committed, so give the host at least 2 GB of RAM (4 GB is
# comfortable). Azure's free B1s has 1 GiB and will be killed by the OOM reaper.
FROM python:3.10-slim

# opencv needs libGL and libglib even in headless use; git is not needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU wheels only. Installed before the code so this layer is cached across code changes.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/backend/requirements.txt \
    && pip install --no-cache-dir transformers sentence-transformers timm

# Only what the server actually reads. The 349 MB of dataset photos are NOT copied: they are
# only used by /demo. The image memory works from stored hashes and vectors, which is why the
# payload here is ~75 MB rather than 1.8 GB. deploy/make_bundle.py builds this subset.
COPY backend/                          /app/backend/
COPY fake_review_dataset_v6/code/      /app/fake_review_dataset_v6/code/
COPY fake_review_dataset_v6/out/model/ /app/fake_review_dataset_v6/out/model/
COPY deploy/bundle/                    /app/fake_review_dataset_v6/out/

# Pre-download the pretrained encoders so the first request is not a 60 s cold start.
ENV HF_HOME=/app/.cache TRANSFORMERS_NO_TF=1 USE_TF=0
RUN python -c "\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \
AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \
import timm; timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=0); \
import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')" \
    || echo 'pre-download skipped; models will be fetched on first request'

# Bind to every interface (a container's 127.0.0.1 is unreachable from outside) and read the
# port from the environment, which is how Container Apps / App Service / Codespaces assign it.
ENV RPF_HOST=0.0.0.0 PORT=8000
EXPOSE 8000

# RPF_TOKEN must be supplied at run time. Without it /analyze is open to anyone who finds
# the URL; see deploy/README.md.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -fs http://localhost:${PORT}/health || exit 1

WORKDIR /app/backend
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT}"]

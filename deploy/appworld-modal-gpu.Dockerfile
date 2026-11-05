# Separate training and serving environments; no credentials or task data in this image.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv ca-certificates curl git build-essential util-linux procps \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/bootstrap \
    && /opt/bootstrap/bin/pip install uv==0.13.3 \
    && ln -s /opt/bootstrap/bin/uv /usr/local/bin/uv
RUN uv venv --python 3.11 /opt/venv-train && uv venv --python 3.11 /opt/venv-serve
RUN uv pip install --python /opt/venv-train/bin/python \
    torch==2.13.0 transformers==5.19.0 peft==0.19.1 accelerate==1.13.0 datasets==4.5.0 trl==0.29.1 \
    && uv pip install --python /opt/venv-serve/bin/python vllm==0.32.0
RUN uv pip check --python /opt/venv-train/bin/python \
    && uv pip check --python /opt/venv-serve/bin/python \
    && uv pip freeze --python /opt/venv-train/bin/python > /opt/train-installed.txt \
    && uv pip freeze --python /opt/venv-serve/bin/python > /opt/serve-installed.txt
RUN useradd --create-home --shell /bin/bash cebtrain \
    && useradd --create-home --shell /bin/bash cebserve
WORKDIR /persist
CMD ["sleep", "infinity"]

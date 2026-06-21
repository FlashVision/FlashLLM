FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY flashllm/ flashllm/

RUN pip install --no-cache-dir -e ".[all]"

COPY configs/ configs/
COPY examples/ examples/

ENTRYPOINT ["flashllm"]
CMD ["--help"]

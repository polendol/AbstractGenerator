# this stage avoids using MBs that are not necessary to execute later on
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS builder

#compilation tools that will not be included int eh final image stage
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3-pip git build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

#install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
      -r requirements.txt


#final image stage
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 \
 && rm -rf /var/lib/apt/lists/*

#copy the environment without unnecessary compilers
COPY --from=builder /opt/venv /opt/venv

# copy the path and add variables recommended for all python dockers + gradio
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME="0.0.0.0"

WORKDIR /app

#not root
RUN useradd --create-home --uid 1001 appuser \
 && mkdir -p /home/appuser/.cache/huggingface \
 && chown -R appuser:appuser /app /home/appuser/.cache
USER appuser

COPY --chown=appuser:appuser . .

#Gradio cache
ENV HF_HOME=/home/appuser/.cache/huggingface

#expose gradio port
EXPOSE 7860

CMD ["python", "./AbstractGenerator.py"]
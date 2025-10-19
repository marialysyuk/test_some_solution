FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

RUN pip3 install --no-cache-dir \
    autoawq==0.2.7 \
    transformers==4.45.2 \
    accelerate==0.34.2 \
    safetensors==0.4.5 \
    tqdm==4.66.5 \
    sentencepiece==0.2.0 protobuf==5.28.2

# Optional but helpful if the runner disallows egress
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

WORKDIR /workspace
COPY . .

ENTRYPOINT ["python3", "solution.py"]

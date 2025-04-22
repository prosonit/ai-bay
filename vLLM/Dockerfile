FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install tools
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip git curl build-essential \
    libaio-dev && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install cmake, ninja, etc
RUN pip install --upgrade pip && \
    pip install cmake ninja

# Optional: Set correct pip/pip3 links
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

# Set env to avoid Python warnings
ENV PYTHONUNBUFFERED=1

# Install torch - must be built with CUDA 11.8
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --extra-index-url https://download.pytorch.org/whl/cu118

# Patch torch's CUDA capability check
RUN python -c "import torch; \
    import os; \
    torch_cuda_ext = os.path.dirname(torch.__file__) + '/cuda/__init__.py'; \
    text = open(torch_cuda_ext).read(); \
    text = text.replace('major_cc >= 7', 'major_cc >= 6'); \
    open(torch_cuda_ext, 'w').write(text)"
 
# Install necessary build tools for vLLM
RUN pip install ninja

# Install vllm source, and patch for compute capability 6.1
WORKDIR /workspace
RUN git clone https://github.com/vllm-project/vllm.git && cd vllm \
    && sed -i 's/sm_70/sm_61/g' vllm/transformers/custom_kernels/CMakeLists.txt \
    && sed -i 's/sm_70/sm_61/g' cmake/torch.cmake \
    && pip install -e .

# Set compute capability lower bound for P1000 in env
ENV TORCH_CUDA_ARCH_LIST="6.1"

CMD ["python"]
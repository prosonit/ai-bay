## NVIDIA

### Install Cuda (Ubuntu 22)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

### Install drivers
sudo apt-get install -y cuda-drivers

### Install Docker
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get update
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin


### Install Nvidia Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

#### Verify docker gpu support for nvidia
sudo docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi


### Install Nvidia gpu-operator

### Metrics
sudo microk8s enable metrics-server

### Models
https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct




sudo microk8s.ctr image pull  docker.io/charmedkubeflow/huggingfaceserver@sha256:2907486cfc390aa7b6a85f4738e79b7a24b569fe5601c102e2b7c874182fd5f3

sudo microk8s.ctr run --rm --tty   --snapshotter overlayfs   docker.io/charmedkubeflow/huggingfaceserver@sha256:2907486cfc390aa7b6a85f4738e79b7a24b569fe5601c102e2b7c874182fd5f3   debug-llama   python3 -u -m huggingfaceserver --model_name=tinyllama --model_id=TinyLlama/TinyLlama-1.1B-Chat-v0.6

sudo microk8s.kubectl run -it --rm --image=docker.io/charmedkubeflow/huggingfaceserver@sha256:2907486cfc390aa7b6a85f4738e79b7a24b569fe5601c102e2b7c874182fd5f3 -- bash



sudo microk8s.ctr image pull docker.io/charmedkubeflow/huggingfaceserver:0.14.1-790009b

sudo microk8s.kubectl run debugdns4  -it --image=docker.io/charmedkubeflow/huggingfaceserver:0.14.1-790009b  --restart=Never   -- python3 -m huggingfaceserver --model_name=tinyllama --model_id=TinyLlama/TinyLlama-1.1B-Chat-v0.6




### add to groups
prosonit@seste-ai-1:~$ sudo usermod -aG sudo $USER
prosonit@seste-ai-1:~$ sudo usermod -aG docker $USER
prosonit@seste-ai-1:~$


## Troubleshooting

## CUDA error: no kernel image is available for execution on the device
RuntimeError: CUDA error: no kernel image is available for execution on the device
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.

This means that the nvidia gpu is compatable with the cuda version required for pyTorch. vLLM requires compute compability 7.0 and example Quadrio P1000 supports 6.1

import torch
print(torch.cuda.get_arch_list())
print(torch.version.cuda)

### Try to fix install of drivers in docker
_______________________________________________
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo snap restart docker



1. PyTorch (pytorch_model.bin)
What is it?
The standard (full-precision) model format for Hugging Face Transformers and native PyTorch.
File extension:
pytorch_model.bin (sometimes .pt)
Typical precision:
16-bit (fp16) or 32-bit (fp32)
Compatibility:
Works everywhere PyTorch is supported (CPU, GPU, TPU).
Use case:
Training, fine-tuning, and highest-accuracy inference.
Size:
Larger file sizes; models can be tens of gigabytes.
Summary:
Most flexible and widely used, but requires more memory and compute.

2. AWQ
What is it?
Activation-aware Weight Quantization – a newer quantization technique designed to reduce model size and speed up inference without much loss in accuracy.
File extension:
Often .safetensors or .bin, but marked "AWQ".
Typical precision:
4-bit or 8-bit
Compatibility:
Supported by vLLM, autoawq, and some optimized inference engines.
Use case:
Memory-efficient, fast inference at near-original accuracy.
Size:
Much smaller, e.g., a 7B model could fit in ~4GB VRAM.
Summary:
For users who want fast, memory-efficient inference with very little accuracy drop.

3. GPTQ
What is it?
GPT Quantization – one of the first major methods for effective 4-bit quantization of large language models, especially Llama-family.
File extension:
Usually .safetensors or .bin, marked "GPTQ".
Typical precision:
4-bit
Compatibility:
Supported by vLLM, exllama, GPTQ-for-LLaMa, etc.
Use case:
Extremely efficient inference, especially popular in LLM deployment.
Size:
Even smaller than AWQ (similar size to 4-bit AWQ).
Summary:
Mainstream for running big LLMs on commodity GPUs, sometimes requires special runners.


4. bitsandbytes
What is it?
A library for quantized/efficient inference and training (by Tim Dettmers).
Supports 8-bit, 4-bit and other quantization for both training and inference.
File extension:
Still often .bin, .safetensors, or Hugging Face format, but marked as 8-bit, 4-bit etc.
Precision:
8-bit, 4-bit, etc.
Compatibility:
Hugging Face Transformers (with special loading), some inference engines.
Use case:
Allows full fine-tuning and inference with much less memory.
Summary:
Lets you load and train (and run) big models on consumer hardware with a big cut in RAM/VRAM needs.

5. GGUF
What is it?
A file format ("GGML Unified Format") for language models – designed for CPU inference with llama.cpp, Koboldcpp, and other lightweight runners outside mainstream ML frameworks.
File extension:
.gguf
Typical precision:
8-bit, 6-bit, 5-bit, 4-bit quantization
Compatibility:
Not for PyTorch. For llama.cpp family tools: run models on Mac/Windows/Linux (CPU or Apple Silicon, some support for CUDA).
Use case:
Run LLMs on laptops/desktops/phones without any ML frameworks or GPUs.
Limitations:
Most often CPU-based, occasionally limited CUDA support (very recent).
Not compatible with vLLM or standard PyTorch.






 docker run --runtime nvidia --gpus all     -v ~/.cache/huggingface:/root/.cache/huggingface     --env "HUGGING_FACE_HUB_TOKEN=<TOKEN>"     -p 8000:8000     --ipc=host     ghcr.io/sasha0552/vllm:v0.8.1     --model jakiAJK/DeepSeek-R1-Distill-Llama-8B_GPTQ-int4 --dtype=half  --cpu-offload-gb 16


 deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B


 docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=<TOKEN>" \
  -p 8000:8000 \
  --ipc=host \
  ghcr.io/sasha0552/vllm:v0.8.1 \
  python3 -m vllm.entrypoints.openai.api_server \
    --model TheBloke/deepseek-llm-7B-chat-GPTQ \
    --dtype=half \
    --cpu-offload-gb 16


docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=<TOKEN>" -p 8000:8000 --ipc=host ghcr.io/sasha0552/vllm:v0.8.1 python3 -m vllm.entrypoints.openai.api_server --model TheBloke/deepseek-llm-7B-chat-GPTQ --dtype=half --cpu-offload-gb 16

SEEMS TO WORK :)
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=<TOKEN>" \
  -p 8000:8000 \
  --ipc=host \
  --entrypoint python3 \
  ghcr.io/sasha0552/vllm:v0.8.1 \
  -m vllm.entrypoints.openai.api_server \
    --model TheBloke/deepseek-llm-7B-chat-GPTQ \
    --dtype=half \
    --cpu-offload-gb 16



1. PyTorch (pytorch_model.bin)
What is it?
The standard (full-precision) model format for Hugging Face Transformers and native PyTorch.
File extension:
pytorch_model.bin (sometimes .pt)
Typical precision:
16-bit (fp16) or 32-bit (fp32)
Compatibility:
Works everywhere PyTorch is supported (CPU, GPU, TPU).
Use case:
Training, fine-tuning, and highest-accuracy inference.
Size:
Larger file sizes; models can be tens of gigabytes.
Summary:
Most flexible and widely used, but requires more memory and compute.

2. AWQ
What is it?
Activation-aware Weight Quantization – a newer quantization technique designed to reduce model size and speed up inference without much loss in accuracy.
File extension:
Often .safetensors or .bin, but marked "AWQ".
Typical precision:
4-bit or 8-bit
Compatibility:
Supported by vLLM, autoawq, and some optimized inference engines.
Use case:
Memory-efficient, fast inference at near-original accuracy.
Size:
Much smaller, e.g., a 7B model could fit in ~4GB VRAM.
Summary:
For users who want fast, memory-efficient inference with very little accuracy drop.

3. GPTQ
What is it?
GPT Quantization – one of the first major methods for effective 4-bit quantization of large language models, especially Llama-family.
File extension:
Usually .safetensors or .bin, marked "GPTQ".
Typical precision:
4-bit
Compatibility:
Supported by vLLM, exllama, GPTQ-for-LLaMa, etc.
Use case:
Extremely efficient inference, especially popular in LLM deployment.
Size:
Even smaller than AWQ (similar size to 4-bit AWQ).
Summary:
Mainstream for running big LLMs on commodity GPUs, sometimes requires special runners.


4. bitsandbytes
What is it?
A library for quantized/efficient inference and training (by Tim Dettmers).
Supports 8-bit, 4-bit and other quantization for both training and inference.
File extension:
Still often .bin, .safetensors, or Hugging Face format, but marked as 8-bit, 4-bit etc.
Precision:
8-bit, 4-bit, etc.
Compatibility:
Hugging Face Transformers (with special loading), some inference engines.
Use case:
Allows full fine-tuning and inference with much less memory.
Summary:
Lets you load and train (and run) big models on consumer hardware with a big cut in RAM/VRAM needs.

5. GGUF
What is it?
A file format ("GGML Unified Format") for language models – designed for CPU inference with llama.cpp, Koboldcpp, and other lightweight runners outside mainstream ML frameworks.
File extension:
.gguf
Typical precision:
8-bit, 6-bit, 5-bit, 4-bit quantization
Compatibility:
Not for PyTorch. For llama.cpp family tools: run models on Mac/Windows/Linux (CPU or Apple Silicon, some support for CUDA).
Use case:
Run LLMs on laptops/desktops/phones without any ML frameworks or GPUs.
Limitations:
Most often CPU-based, occasionally limited CUDA support (very recent).
Not compatible with vLLM or standard PyTorch.






 docker run --runtime nvidia --gpus all     -v ~/.cache/huggingface:/root/.cache/huggingface     --env "HUGGING_FACE_HUB_TOKEN=<TOKEN>"     -p 8000:8000     --ipc=host     ghcr.io/sasha0552/vllm:v0.8.1     --model jakiAJK/DeepSeek-R1-Distill-Llama-8B_GPTQ-int4 --dtype=half  --cpu-offload-gb 16


 deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B


 docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=<TOKEN>" \
  -p 8000:8000 \
  --ipc=host \
  ghcr.io/sasha0552/vllm:v0.8.1 \
  python3 -m vllm.entrypoints.openai.api_server \
    --model TheBloke/deepseek-llm-7B-chat-GPTQ \
    --dtype=half \
    --cpu-offload-gb 16


docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=<TOKEN>" -p 8000:8000 --ipc=host ghcr.io/sasha0552/vllm:v0.8.1 python3 -m vllm.entrypoints.openai.api_server --model TheBloke/deepseek-llm-7B-chat-GPTQ --dtype=half --cpu-offload-gb 16

SEEMS TO WORK :)
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=<TOKEN>" \
  -p 8000:8000 \
  --ipc=host \
  --entrypoint python3 \
  ghcr.io/sasha0552/vllm:v0.8.1 \
  -m vllm.entrypoints.openai.api_server \
    --model TheBloke/deepseek-llm-7B-chat-GPTQ \
    --dtype=half \
    --cpu-offload-gb 16


ssh-copy-id -i ~/.ssh/id_ed25519.pub nitro@10.22.5.138    



# Trying
sudo docker run --gpus all -it --rm     -p 8000:8000     -v /home/nitro/hf_cache:/root/.cache/huggingface     ghcr.io/sasha0552/vllm:v0.8.1 --model    TheBloke/deepseek-llm-7B-chat-GPTQ  --tensor-parallel-size 2 --dtype=half

# Another try
sudo docker run --gpus all -it --rm  --shm-size=8g   -p 8000:8000     -v /home/nitro/hf_cache:/root/.cache/huggingface     ghcr.io/sasha0552/vllm:v0.8.1 --model    TheBloke/deepseek-llm-7B-chat-GPTQ  --tensor-parallel-size 2 --dtype=half --max_model_len=2048

                                            


sudo docker run --gpus all -it -p 8000:8000   -v /home/nitro/hf_cache:/root/.cache/huggingface     vllm/vllm-openai:latest --model unsloth/DeepSeek-R1-Distill-Llama-7B-unsloth-bnb-4bit --max_model_len=1000

sudo docker run --gpus all -it -p 8000:8000  -v /home/nitro/hf_cache:/root/.cache/huggingface     vllm/vllm-openai:latest --model unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit --max_model_len=1000



 

sudo docker pull ghcr.io/huggingface/text-generation-inference:latest



sudo docker run --gpus all -it --rm --shm-size 1g -p 8000:80  -v ~/tgi-models:/data ghcr.io/huggingface/text-generation-inference:latest  --model-id TheBloke/Llama-2-7B-AWQ --quantize awq 

sudo docker run --gpus all -it --rm --shm-size 1g -p 8000:80  -v ~/tgi-models:/data ghcr.io/huggingface/text-generation-inference:latest  --model-id unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit --max-batch-prefill-tokens 128 --max-input-length 256  --max-total-tokens 512

cpu only: Remove gpu stuff and add --disable-custom-kernel

sudo docker pull ghcr.io/huggingface/text-generation-inference:3.2.3-intel-cpu


sudo docker run -it --rm  -p 8000:80  -v ~/tgi-models:/data ghcr.io/huggingface/text-generation-inference:3.2.3-intel-cpu --model-id microsoft/Phi-3.5-mini-instruct


### CPU Only (?)
sudo docker run --rm --privileged --cap-add=sys_nice --device=/dev/dri --ipc=host --shm-size 1g -p 8000:80 -v ~/tgi-models:/data  ghcr.io/huggingface/text-generation-inference:3.2.3-intel-cpu --model-id microsoft/Phi-3.5-mini-instruct --cuda-graphs 0

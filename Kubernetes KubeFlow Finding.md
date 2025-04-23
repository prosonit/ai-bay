## NVIDIA

### Install drivers

### Install Nvidia Container Toolkit

### Install Nvidia gpu-operator

### Install Cuda (Ubuntu 22)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8



### Metrics
sudo microk8s enable metrics-server

### Models
https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct




sudo microk8s.ctr image pull  docker.io/charmedkubeflow/huggingfaceserver@sha256:2907486cfc390aa7b6a85f4738e79b7a24b569fe5601c102e2b7c874182fd5f3

sudo microk8s.ctr run --rm --tty   --snapshotter overlayfs   docker.io/charmedkubeflow/huggingfaceserver@sha256:2907486cfc390aa7b6a85f4738e79b7a24b569fe5601c102e2b7c874182fd5f3   debug-llama   python3 -u -m huggingfaceserver --model_name=tinyllama --model_id=TinyLlama/TinyLlama-1.1B-Chat-v0.6

sudo microk8s.kubectl run -it --rm --image=docker.io/charmedkubeflow/huggingfaceserver@sha256:2907486cfc390aa7b6a85f4738e79b7a24b569fe5601c102e2b7c874182fd5f3 -- bash



sudo microk8s.ctr image pull docker.io/charmedkubeflow/huggingfaceserver:0.14.1-790009b

sudo microk8s.kubectl run debugdns4  -it --image=docker.io/charmedkubeflow/huggingfaceserver:0.14.1-790009b  --restart=Never   -- python3 -m huggingfaceserver --model_name=tinyllama --model_id=TinyLlama/TinyLlama-1.1B-Chat-v0.6


### Verify docker gpu support for nvidia
sudo docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

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
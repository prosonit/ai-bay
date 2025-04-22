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

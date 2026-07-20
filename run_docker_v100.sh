#!/usr/bin/env bash
set -e
IMAGE_NAME="byzfl_snn"
sudo docker build -t ${IMAGE_NAME} .
sudo docker run --gpus all --rm \
    -v "$(pwd)":/home/bendouro \
    ${IMAGE_NAME} \
    bash -c "source venv/bin/activate && python run_full_sweep_docker.py"

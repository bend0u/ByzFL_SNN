#!/usr/bin/env bash
set -e

IMAGE_NAME="byzfl_snn"

echo "Building docker image if needed..."
docker build -t ${IMAGE_NAME} .

echo "Running full sweep in docker with 4 V100 GPUs..."
docker run --gpus all --rm \
    --user $(id -u):$(id -g) \
    -v "$(pwd)":/home/bendouro \
    ${IMAGE_NAME} \
    bash -c "python run_full_sweep_docker.py"

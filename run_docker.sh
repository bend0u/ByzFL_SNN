#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

IMAGE_NAME="byzfl_snn"

echo "========================================="
echo "Building SNN Docker Image..."
echo "========================================="
sudo docker build -t ${IMAGE_NAME} .

echo ""
echo "========================================="
echo "Running SNN Container with GPU Support..."
echo "========================================="
echo "Mounting current directory to /home/bendouro"
echo "All outputs (results/, plots/, logs) will be saved on your host."
echo "Press Ctrl+D or type 'exit' to leave the container."
echo "========================================="

sudo docker run --gpus all --rm -it \
    -v "$(pwd)":/home/bendouro \
    ${IMAGE_NAME}

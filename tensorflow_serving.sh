#!/bin/bash

MODEL_DIR="$(pwd)/export_dir" # 修改pb模型存放地址
NAME="Extract Property"

echo "Pull tensorflow 1.15.2 images."
docker pull tensorflow/tensorflow:1.15.2-gpu

echo "Start tensorflow-serving."

docker run -t --rm -p 8513:8501 \
	-v "$MODEL_DIR/:/models/serving_default" \
	-e MODEL_NAME=serving_default \
	--name $NAME \  # 修改容器名称
	tensorflow/serving &

#!/bin/bash

# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 安装依赖
echo "正在安装依赖..."
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 运行程序
echo "启动训练..."
python src/modelTrans.py

#!/bin/bash

# KANO 一键安装脚本

echo "开始安装KANO依赖..."

# 升级pip
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip

# 安装核心依赖
echo "安装PyTorch..."
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.13.1 torchvision torchaudio

echo "安装torch-scatter..."
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch-scatter==2.0.9

echo "安装科学计算包..."
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy==1.20.3 \
    scipy \
    pandas \
    matplotlib \
    seaborn

echo "安装RDKit..."
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple rdkit-pypi

echo "安装NLP包..."
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "gensim==4.2.0" \
    "nltk==3.4.5"

echo "安装OWL相关包..."
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "Owlready2==0.37" \
    "Click>=7.0" \
    "rdflib>=4.2.2" \
    "scikit-learn~=0.24.2" \
    "pyparsing==2.4.7" \
    "owl2vec-star==0.2.1"

echo "安装其他依赖..."
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    tqdm \
    tensorboard \
    jupyter

echo "验证安装..."
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import rdkit; print('RDKit: OK')"
python -c "import owl2vec_star; print('OWL2Vec-Star: OK')"
python -c "import numpy; print('NumPy:', numpy.__version__)"

echo "安装完成！"

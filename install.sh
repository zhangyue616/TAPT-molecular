#!/bin/bash

echo "Starting TAPT dependency installation..."

python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip

echo "Installing PyTorch..."
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.13.1 torchvision torchaudio

echo "Installing torch-scatter..."
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch-scatter==2.0.9

echo "Installing scientific computing packages..."
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy==1.20.3 \
    scipy \
    pandas \
    matplotlib \
    seaborn

echo "Installing RDKit..."
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple rdkit-pypi

echo "Installing NLP packages..."
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "gensim==4.2.0" \
    "nltk==3.4.5"

echo "Installing OWL-related packages..."
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "Owlready2==0.37" \
    "Click>=7.0" \
    "rdflib>=4.2.2" \
    "scikit-learn~=0.24.2" \
    "pyparsing==2.4.7" \
    "owl2vec-star==0.2.1"

echo "Installing remaining dependencies..."
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    tqdm \
    tensorboard \
    jupyter

echo "Verifying installation..."
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import rdkit; print('RDKit: OK')"
python -c "import owl2vec_star; print('OWL2Vec-Star: OK')"
python -c "import numpy; print('NumPy:', numpy.__version__)"

echo "Installation complete!"

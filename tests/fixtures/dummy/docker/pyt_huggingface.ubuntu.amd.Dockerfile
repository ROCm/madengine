# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
#
# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################
ARG BASE_DOCKER=rocm/pytorch:latest
FROM $BASE_DOCKER

USER root
ENV WORKSPACE_DIR=/workspace
ENV DEBIAN_FRONTEND=noninteractive

# Create workspace directory
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

# Install system dependencies first (better caching)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg2 \
    sudo \
    unzip \
    jq \
    sshpass \
    sshfs \
    netcat-traditional \
    locales \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configure locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Install huggingface transformers - using official repo with latest stable release
# Note: Using official huggingface/transformers instead of ROCm fork for better compatibility
RUN cd /workspace && \
    git clone https://github.com/huggingface/transformers transformers && \
    cd transformers && \
    # Checkout latest stable release tag (adjust as needed)
    git checkout $(git describe --tags --abbrev=0) && \
    git show --oneline -s && \
    pip install -e . && \
    cd ..

# Install core dependencies with compatible versions
# Pin huggingface-hub to compatible range to avoid conflicts
RUN pip3 install --no-cache-dir \
    'huggingface_hub>=0.20.0' \
    'tokenizers>=0.13.0' \
    'datasets>=2.0.0' \
    'accelerate>=0.20.0' \
    && pip3 list

# Intentionally skip torchaudio to prevent torch version conflicts
RUN if [ -f /workspace/transformers/examples/pytorch/_tests_requirements.txt ]; then \
        sed -i 's/torchaudio//g' /workspace/transformers/examples/pytorch/_tests_requirements.txt && \
        sed -i 's/torch[>=<].*//g' /workspace/transformers/examples/pytorch/_tests_requirements.txt; \
    fi

# Install transformers example dependencies
RUN if [ -f /workspace/transformers/examples/pytorch/_tests_requirements.txt ]; then \
        cd /workspace/transformers/examples/pytorch && \
        pip3 install -r _tests_requirements.txt || true; \
    fi

# Install additional ML and utility packages
RUN pip3 install --no-cache-dir \
    GPUtil \
    azureml \
    azureml-core \
    ninja \
    cerberus \
    sympy \
    sacremoses \
    'sacrebleu>=2.0.0' \
    sentencepiece \
    scipy \
    scikit-learn \
    evaluate \
    tensorboard \
    && pip3 list

# Verify installation and dependencies
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'ROCm/HIP: {torch.version.hip if hasattr(torch.version, \"hip\") else \"N/A\"}')" && \
    python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" && \
    python3 -c "import huggingface_hub; print(f'HuggingFace Hub: {huggingface_hub.__version__}')" && \
    python3 -c "from transformers import AutoModel, AutoTokenizer; print('Transformers import successful')"

# Record final configuration
RUN pip3 list > /workspace/pip_packages.txt && \
    echo "=== Environment Configuration ===" && \
    cat /workspace/pip_packages.txt

# Reset frontend to avoid issues
ENV DEBIAN_FRONTEND=

WORKDIR $WORKSPACE_DIR

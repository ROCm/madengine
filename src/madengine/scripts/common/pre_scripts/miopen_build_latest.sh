#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#

set -e
set -x

#Install cget
pip install cget

# Install rclone
pip install https://github.com/pfultz2/rclone/archive/master.tar.gz

MIOPEN_DIR='/root/dMIOpen'
#Clone MIOpen
git clone https://github.com/ROCm/MIOpen.git $MIOPEN_DIR
cd $MIOPEN_DIR

PREFIX='/opt/rocm'
MIOPEN_DEPS="${MIOPEN_DIR}/cget"

# Install dependencies
cget install pfultz2/rocm-recipes
cget install -f min-requirements.txt
export CXXFLAGS="-isystem ${PREFIX}/include"
cget install -f ./mlir-requirements.txt

USER="miopenpdb"
BACKEND="HIP"

# Build MIOpen
git remote update
if [[ "$1" != "" ]]; then
    git checkout --track origin/"$1"
else
    git checkout origin/develop
fi
git pull
git branch

mkdir "${MIOPEN_DIR}/build"
cd "${MIOPEN_DIR}/build"
export CXX=/opt/rocm/llvm/bin/clang++
cmake -DMIOPEN_BACKEND=$BACKEND -DCMAKE_PREFIX_PATH=$MIOPEN_DEPS $MIOPEN_DIR

make -j $(nproc)
make install

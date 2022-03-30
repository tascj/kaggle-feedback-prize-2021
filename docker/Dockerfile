FROM nvcr.io/nvidia/pytorch:21.10-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"


# RUN pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
RUN pip install cython scikit-learn pandas transformers sentencepiece
RUN pip install -U numpy
RUN pip install mmcv==1.4.6
RUN pip uninstall -y opencv-python && pip install opencv-python-headless

WORKDIR /workspace

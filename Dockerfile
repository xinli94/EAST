FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

WORKDIR /workspace/

# install basics 
RUN apt-get update -y --fix-missing
RUN apt-get install -y --fix-missing git curl ca-certificates bzip2 cmake tree htop bmon iotop vim wget unzip
RUN apt-get install -y --fix-missing libglib2.0-0 libsm6 libxext6 libxrender1 libfontconfig1 build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libgeos-dev

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.4.10-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN /miniconda/bin/conda install conda-build \
 && /miniconda/bin/conda create -y --name py36 python=3.6.4 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Ensure conda version is at least 4.4.11
# (because of this issue: https://github.com/conda/conda/issues/6811)
ENV CONDA_AUTO_UPDATE_CONDA=false
RUN conda install -y "conda>=4.4.11" && conda clean -ya

# gcc
RUN apt-get update && apt-get install build-essential software-properties-common -y && add-apt-repository ppa:ubuntu-toolchain-r/test -y && apt-get update && apt-get install gcc-snapshot -y && apt-get update && apt-get install gcc-6 g++-6 -y && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-6 && apt-get install gcc-4.8 g++-4.8 -y && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 60 --slave /usr/bin/g++ g++ /usr/bin/g++-4.8

# requirements
RUN git clone --recursive https://github.com/xinli94/EAST.git
RUN cd /workspace/EAST; git fetch --all && pip install -r requirements.txt

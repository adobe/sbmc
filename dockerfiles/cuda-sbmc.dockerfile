# CUDA-capable Docker configuration to run "Sample-Based Monte Carlo
# denoising..." [Gharbi2016]
#
# This dockerfile requires `nvidia-docker` check this url:
# <https://github.com/NVIDIA/nvidia-docker> or install using
# `make nvidia_docker`
# FROM nvidia/cuda:9.0-devel
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
# FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
MAINTAINER Michael Gharbi <mgharbi@adobe.com>

# Download and update required packages
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y \
        build-essential \
        vim \
        git \
        bash \
        liblz4-dev \
        libopenexr-dev \
        curl \
        bison \
        libomp-dev \
        cmake \
        flex \
        qt5-default \
        libeigen3-dev \
        wget \
        unzip \
        libncurses5-dev \
        liblz4-dev

# Change default shell
SHELL ["/bin/bash", "-c"]

# Create directories and copy data
RUN mkdir -p /sbmc_app /sbmc_app/output /sbmc_app/data
COPY pbrt_patches /sbmc_app/patches

WORKDIR /sbmc_app

# Download, patch and install PBRTv2 with our changes for data generation.
RUN git clone https://github.com/mmp/pbrt-v2 pbrt_tmp
RUN cd pbrt_tmp && git checkout e6f6334f3c26ca29eba2b27af4e60fec9fdc7a8d
RUN mv pbrt_tmp/src pbrt
RUN rm -rf pbrt_tmp
# The patch allows to save samples while path-tracing
RUN patch -d pbrt -p1 -i /sbmc_app/patches/sbmc_pbrt.diff
RUN cd pbrt && make -j 4

# Install a few previous denoising works for comparison -----------------------

# [Sen2011] 
# "On Filtering the Noise from the Random Parameters in Monte Carlo Rendering"
RUN (wget http://cvc.ucsb.edu/graphics/Papers/Sen2011_RPF/PaperData/RPF-v1.0.zip && \
            unzip RPF-v1.0.zip && \
            mv RPF-v1.0/pbrt-v2-rpf/src 2011_sen_rpf && \
            rm -rf RPF-v1.0*  && \
# Patch to fix compilation errors
            patch -d 2011_sen_rpf -p1 -i /sbmc_app/patches/2011_sen_rpf.diff && \
            cd 2011_sen_rpf && make -j 4) || echo "Sen2011 could not be downloaded"

# [Rousselle2012] :requires CUDA
# "Adaptive Rendering with Non-Local Means Filtering"
RUN (wget http://cgg-zwicker.inf.unibe.ch/downloads/nlm-code-data.zip && \
            unzip nlm-code-data.zip && \
            mv nlm-code+data/pbrt-v2-nlm/src 2012_rousselle_nlm && \
            rm -rf nlm-code+data nlm-code-data.zip && \
# Patch to fix compilation errors
            patch -d 2012_rousselle_nlm -p1 -i /sbmc_app/patches/2012_rousselle_nlm.diff  && \
            cd 2012_rousselle_nlm && make -j 4 ) || echo "Rousselle2012 could not be downloaded"

# [Kalantari2015]: requires CUDA.
# "A Machine Learning Approach for Filtering Monte Carlo Noise"
RUN (wget http://cvc.ucsb.edu/graphics/Papers/SIGGRAPH2015_LBF/PaperData/SIGGRAPH15_LBF_v1.0.zip && \
            unzip SIGGRAPH15_LBF_v1.0.zip && \
            mv SIGGRAPH15_LBF_v1.0/pbrt-v2-lbf/src 2015_kalantari_lbf && \
            mkdir 2015_kalantari_lbf/pretrained && \
            mv SIGGRAPH15_LBF_v1.0/pbrt-v2-lbf/scenes/*.dat 2015_kalantari_lbf/pretrained && \
            rm -rf SIGGRAPH15_LBF_v1.0*  && \
            patch -d 2015_kalantari_lbf -p1 -i /sbmc_app/patches/2015_kalantari_lbf.diff  && \\
            cd 2015_kalantari_lbf && make -j 4 ) || echo "Kalantari2015 could not be downloaded"

# [Bitterli2016]
RUN git clone https://github.com/tunabrain/tungsten.git 2016_bitterli_nfor
RUN cd 2016_bitterli_nfor && git checkout 88ea02044dbaf20472a8173b6752460b50c096d8 && rm -rf .git
RUN patch -d 2016_bitterli_nfor -p1 -i /sbmc_app/patches/2016_bitterli_nfor.diff
RUN cd 2016_bitterli_nfor && mkdir build && cd build && cmake .. && make -j 4
# -----------------------------------------------------------------------------


# Install Halide
RUN wget -O halide.tgz https://github.com/halide/Halide/releases/download/v8.0.0/halide-linux-64-gcc53-800-65c26cba6a3eca2d08a0bccf113ca28746012cc3.tgz
RUN tar zvxf halide.tgz
RUN rm -rf halide.tgz
ENV HALIDE_DISTRIB_DIR /sbmc_app/halide


# Python Environment ----------------------------------------------------------
RUN curl -o /sbmc_app/anaconda.sh -O \
        https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh && \
    sha256sum /sbmc_app/anaconda.sh && \
    chmod a+x /sbmc_app/anaconda.sh && \
    /sbmc_app/anaconda.sh -b -p /sbmc_app/anaconda
ENV PATH /sbmc_app/anaconda/bin:$PATH

RUN source activate
RUN conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch  
# RUN conda install pytorch torchvision cudatoolkit=9.2 -c pytorch  
RUN pip install --upgrade pip && pip install pytest
# -----------------------------------------------------------------------------

# Set the environment variables so that the `demo/*` commands in the Makefile
# point to the right directories.
ENV OUTPUT /sbmc_app/output
ENV DATA /sbmc_app/data
ENV PBRT /sbmc_app/pbrt/bin/pbrt
ENV OBJ2PBRT /sbmc_app/pbrt/bin/obj2pbrt
ENV SEN2011 /sbmc_app/2011_sen_rpf
ENV ROUSSELLE2012 /sbmc_app/2012_rousselle_nlm
ENV KALANTARI2015 /sbmc_app/2015_kalantari_lbf
ENV BITTERLI2016 /sbmc_app/2016_bitterli_nfor

# Install our code
COPY . /sbmc_app/sbmc
RUN cd sbmc/halide_pytorch && python setup.py install
RUN cd sbmc && python setup.py develop
WORKDIR /sbmc_app/sbmc

# NOTE(mgharbi):
# Command used to compute the patch: diff -uarN pbrt rendernet_pbrt > # sbmc_pbrt.diff

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/Miniconda3-latest-Linux-x86_64.sh && \
    bash /root/Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3

ENV PATH="/root/miniconda3/bin:${PATH}"

RUN conda init bash && \
    echo "conda activate annstention" >> ~/.bashrc

RUN conda create -n annstention python=3.10 -y

RUN apt-get install ca-certificates gpg wget -y

RUN test -f /usr/share/doc/kitware-archive-keyring/copyright || wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null

RUN apt-get update

RUN apt-get install kitware-archive-keyring -y

RUN apt-get install cmake -y

RUN apt-get install libopenblas-dev -y

RUN apt-get install libgflags-dev -y

SHELL ["conda", "run", "-n", "annstention", "/bin/bash", "-c"]

RUN pip3 install torch torchvision torchaudio && \
    pip install swig numpy transformers datasets accelerate && \
    pip install flash-attn --no-build-isolation

RUN conda install pytorch::faiss-gpu

RUN python -c "import flash_attn"
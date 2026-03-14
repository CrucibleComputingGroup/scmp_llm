# ANNS-Attention

## Build FAISS from source

1. ~~Install `cmake` in docker. 
https://apt.kitware.com/~~

2. ~~Install OpenBLAS or other BLAS library. 
http://www.openmathlib.org/OpenBLAS/docs/install/~~

    ~~`apt install libopenblas-dev`~~

3. ~~Install swig, gflags~~
    
    ~~`pip install swig`~~

    ~~`apt-get install libgflags-dev`~~

4. Follow the guide in https://github.com/facebookresearch/faiss/blob/main/INSTALL.md. 

    `cmake -B build . -DFAISS_ENABLE_GPU=ON -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_PYTHON=ON`

    `make -C build -j8 faiss`

    `sudo make -C build install`

5. `cp ../faiss/faiss/impl/maybe_owned_vector.h /usr/local/include/faiss/impl/`

7. `conda install -c conda-forge libstdcxx-ng`

8. Install transformers, datasets, accelerate, flash-attn. 

## Transformers Llama

### Remarks

Test on wolverine: `export HF_HOME=/mnt/data/huggingface/`

source intel mkl. `source /opt/intel/oneapi/setvars.sh`

Flash-infer floating point exception. 

Using flash-attn building from source. 
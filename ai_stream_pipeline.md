## 3090 编译指令
```
    cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" ..
    cmake --build . -j$(nproc)
```
## WSL sevnce 编译指令
```
    cmake --build . -j$(nproc)
```
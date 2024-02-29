# nn_simulator

CUDA 11.8 (export PATH="/usr/local/cuda-11.8/bin:$PATH") 
Libtorch: No need to install anything just download and extract 


To build: 
mkdir build 
cd build 
cmake ../ -DCMAKE_INSTALL_PREFIX='release' -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/home/bargun2/Programs/libtorch_wcuda/libtorch .. -D CMAKE_CUDA_COMPILER=$(which nvcc)
cmake --build . --config Release





cmake_minimum_required(VERSION 3.28.3)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_C_COMPILER gcc)

project(BFS LANGUAGES CUDA CXX)

add_executable(BFS main.cu)
add_executable(CPU_version bfgs.cpp)

set_target_properties(BFS PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xlinker=-rpath,/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu")
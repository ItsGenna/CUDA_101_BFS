cmake_minimum_required(VERSION 3.28.3)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_C_COMPILER gcc)

project(BFS LANGUAGES CUDA CXX)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
endif()

add_executable(BFS main.cu)
add_executable(CPU_version bfgs.cpp)
add_executable(CUDA_debugger main.cu)

set_target_properties(BFS PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xlinker=-rpath,/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(CUDA_debugger PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Escludi alcune configurazioni dalla modalità Debug
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Skipping debug configuration")
else()
    message(STATUS "Debugging enabled for main")
endif()

# Messaggi utili per la configurazione
message(STATUS "CMake Version: ${CMAKE_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
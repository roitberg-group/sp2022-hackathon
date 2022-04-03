cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr")

find_package(Torch REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python_INCLUDE_DIRS})

project(cuaev LANGUAGES CXX CUDA)
# Define our library target
add_library(cuaev SHARED cuaev.cpp aev.cu aev.h cuaev_cub.cuh)
target_include_directories(cuaev PRIVATE ./)
# set_property(TARGET cuaev PROPERTY CUDA_ARCHITECTURES 80)
set(CMAKE_CXX_STANDARD 14)
# Link against LibTorch
target_link_libraries(cuaev ${TORCH_LIBRARIES})
target_link_libraries(cuaev ${Python_LIBRARIES})

project(ani LANGUAGES CXX)
# Define our library target
find_package(OpenMP REQUIRED)
add_library(ani SHARED mnp.cpp)
target_include_directories(ani PRIVATE ./)
set(CMAKE_CXX_STANDARD 14)
# Link against LibTorch
target_link_libraries(ani ${TORCH_LIBRARIES})
target_link_libraries(ani ${Python_LIBRARIES})
target_link_libraries(ani OpenMP::OpenMP_CXX)

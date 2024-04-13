cmake_minimum_required(VERSION 3.18)
project(SGEMM LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

enable_language(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 80 86)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall,-Wextra")

# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# Configure the source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# generate executable
add_executable(sgemm_main sgemm_main.cu ${SRC})
target_link_libraries(sgemm_main ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY})
set_target_properties(sgemm_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# in debug mode, add debug symbols to device code
# this disables most optimizations and kills performance
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sgemm_main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

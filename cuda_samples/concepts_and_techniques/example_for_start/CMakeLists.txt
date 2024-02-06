cmake_minimum_required(VERSION 3.18)
project(SimpleCUDASample LANGUAGES CXX CUDA)

enable_language(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--std=c++14)

# ���� C++ ��������־
# set(CMAKE_CXX_FLAGS_RELEASE "/MT")
# set(CMAKE_CXX_FLAGS_DEBUG "/MTd")

# ���� CUDA ��������־�����ʹ�� CUDA��
# set(CUDA_NVCC_FLAGS_RELEASE -Xcompiler /MT)
# set(CUDA_NVCC_FLAGS_DEBUG -Xcompiler /MTd)


add_executable(SimpleCUDASample example.cpp kernel.cu)

target_link_libraries(SimpleCUDASample ${CUDA_LIBRARIES})


# CUDA specific settings
set_target_properties(SimpleCUDASample PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_property(TARGET SimpleCUDASample PROPERTY CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86)

# Compiler and linker options for different configurations
# if(MSVC)
#     # Debug configuration
#     target_compile_definitions(SimpleCUDASample PRIVATE "$<$<CONFIG:DEBUG>:WIN32;_MBCS>")
#     target_compile_options(SimpleCUDASample PRIVATE "$<$<CONFIG:DEBUG>:/wd4819>")
#     set_target_properties(SimpleCUDASample PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG "../../../bin/win64/Debug")

#     # Release configuration
#     target_compile_definitions(SimpleCUDASample PRIVATE "$<$<CONFIG:RELEASE>:WIN32;_MBCS>")
#     set_target_properties(SimpleCUDASample PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "../../../bin/win64/Release")
# endif()
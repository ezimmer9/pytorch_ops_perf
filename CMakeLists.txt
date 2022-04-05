cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(cmake_gen)
set(CMAKE_CXX_STANDARD 14)
option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" OFF)
if (DOWNLOAD_MNIST)
  message(STATUS 'Downloading MNIST dataset')
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/../download_mnist.py
      -d ${CMAKE_BINARY_DIR}/data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()
include_directories(${CMAKE_SOURCE_DIR}/include)
find_package(Torch REQUIRED)
add_executable(code_gen cpp_gen.cpp src/perf_events.cpp)
target_compile_features(code_gen PUBLIC cxx_range_for)
target_link_libraries(code_gen ${TORCH_LIBRARIES})
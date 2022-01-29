cmake_minimum_required(VERSION 3.18)
project(Spin1GpeCube LANGUAGES CXX CUDA)

file(GLOB SOURCES
    *.hpp
    *.cpp
	*.cu
	*.h
)

set(CMAKE_CUDA_ARCHITECTURES 52)

add_executable(Spin1GpeCube ${SOURCES})

target_include_directories(Spin1GpeCube PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(Spin1GpeCube Mesh)
target_link_libraries(Spin1GpeCube Output)
target_link_libraries(Spin1GpeCube Types)
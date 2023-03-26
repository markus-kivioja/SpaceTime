cmake_minimum_required(VERSION 3.18)
project(RelativisticSpin2Gpe LANGUAGES CXX CUDA)

file(GLOB SOURCES
    *.hpp
    *.cpp
	*.cu
	*.h
)

set(CMAKE_CUDA_ARCHITECTURES 52)

add_executable(RelativisticSpin2Gpe ${SOURCES})

target_include_directories(RelativisticSpin2Gpe PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(RelativisticSpin2Gpe Mesh)
target_link_libraries(RelativisticSpin2Gpe Output)
target_link_libraries(RelativisticSpin2Gpe Types)
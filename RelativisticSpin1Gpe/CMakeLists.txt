cmake_minimum_required(VERSION 3.18)
project(RelativisticSpin1Gpe LANGUAGES CXX CUDA)

file(GLOB SOURCES
    *.hpp
    *.cpp
	*.cu
	*.h
)

set(CMAKE_CUDA_ARCHITECTURES 52)

add_executable(RelativisticSpin1Gpe ${SOURCES})

target_include_directories(RelativisticSpin1Gpe PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(RelativisticSpin1Gpe Mesh)
target_link_libraries(RelativisticSpin1Gpe Output)
target_link_libraries(RelativisticSpin1Gpe Types)
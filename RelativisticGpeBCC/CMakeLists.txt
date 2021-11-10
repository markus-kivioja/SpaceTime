cmake_minimum_required(VERSION 3.18)
project(RelativisticGpeBCC LANGUAGES CXX CUDA)

file(GLOB SOURCES
    *.hpp
    *.cpp
	*.cu
	*.h
)

set(CMAKE_CUDA_ARCHITECTURES 52)

add_executable(RelativisticGpeBCC ${SOURCES})

target_include_directories(RelativisticGpeBCC PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(RelativisticGpeBCC Mesh)
target_link_libraries(RelativisticGpeBCC Output)
target_link_libraries(RelativisticGpeBCC Types)
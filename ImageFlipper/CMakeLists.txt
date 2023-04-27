cmake_minimum_required(VERSION 3.18)
project(ImageFlipper)

file(GLOB SOURCES
    *.hpp
    *.cpp
)

add_executable(ImageFlipper ${SOURCES})

target_include_directories(ImageFlipper PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(ImageFlipper Mesh)
target_link_libraries(ImageFlipper Output)
target_link_libraries(ImageFlipper Types)
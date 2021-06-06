cmake_minimum_required(VERSION 3.18)
project(GrossPitaevskii)

file(GLOB SOURCES
    GrossPitaevskii/*.hpp
    GrossPitaevskii/*.cpp
)

add_executable(GrossPitaevskii ${SOURCES})

target_include_directories(GrossPitaevskii PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/GrossPitaevskii/>
)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(Mesh)
add_subdirectory(Output)
add_subdirectory(Types)

target_link_libraries(GrossPitaevskii Mesh)
target_link_libraries(GrossPitaevskii Output)
target_link_libraries(GrossPitaevskii Types)
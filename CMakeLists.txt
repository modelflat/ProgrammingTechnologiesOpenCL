cmake_minimum_required(VERSION 3.9)
project(ProgrammingTechnologiesOpenCL)


set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-Werror -Wall -Wno-deprecated-declarations -Wno-ignored-attributes")

set(OCL_ROOT $ENV{OCL_ROOT})
include_directories( ${OCL_ROOT}/include )
link_directories( ${OCL_ROOT}/lib/x86_64 )

add_executable(${PROJECT_NAME}-trivial-c trivial.c)
target_link_libraries(${PROJECT_NAME}-trivial-c OpenCL)

add_executable(${PROJECT_NAME}-trivial-cpp trivial.cpp)
target_link_libraries(${PROJECT_NAME}-trivial-cpp OpenCL)



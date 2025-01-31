cmake_minimum_required(VERSION 3.24)
project(DP)

set(CMAKE_CXX_STANDARD 20)

add_executable(dp_experiment DP.cpp)
# Link compiled lleaves model
target_link_libraries(dp_experiment ${CMAKE_SOURCE_DIR}/../lleaves.o)

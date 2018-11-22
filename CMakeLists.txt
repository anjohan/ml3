cmake_minimum_required(VERSION 3.7)
project(3 Fortran)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP)
if(OpenMP_Fortran_FOUND)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenMP_Fortran_FLAGS}")
endif()

if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wall -Wextra")
    set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -fcheck=all")
    set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -march=native")
endif(CMAKE_Fortran_COMPILER_ID MATCHES GNU)

add_library(euler_diffusion ${CMAKE_CURRENT_SOURCE_DIR}/lib/euler_diffusion.f90)
add_executable(euler ${CMAKE_CURRENT_SOURCE_DIR}/programs/euler.f90)
target_link_libraries(euler euler_diffusion)

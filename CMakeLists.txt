cmake_minimum_required(VERSION 3.9)
project(master_job C)

set(CMAKE_C_STANDARD 99)
find_package(OpenMP)
find_package(GSL REQUIRED)

add_executable(master_job main.c)

OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
    FIND_PACKAGE(OpenMP)
    IF(OPENMP_FOUND)
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    ENDIF()
ENDIF()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_C_COMPILER /usr/local/bin/gcc-8)

target_link_libraries(master_job GSL::gsl GSL::gslcblas)

cmake_minimum_required(VERSION 3.1)

# This will be picked up by cmake from the conda env
#LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/elibs/tbb")

project(Texturing)
include(ExternalProject)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'RELWITHDEBINFO' as none was specified.")
    set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

#Search for TBB library (release build)
set(CMAKE_BUILD_TYPE_BACKUP ${CMAKE_BUILD_TYPE})
set(CMAKE_BUILD_TYPE RELEASE)
FIND_PACKAGE(TBB REQUIRED)
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE_BACKUP})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED yes)
add_definitions(-DEIGEN_MPL2_ONLY)

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -Wextra -Wundef -pedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpmath=sse -funroll-loops")
endif()

find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
find_package(TIFF REQUIRED)

# Eigen
find_package(Eigen3 3 REQUIRED NO_MODULE)
if (EIGEN3_FOUND)
   message("-- Found Eigen3: ${EIGEN3_INCLUDE_DIRS}")
else (EIGEN3_FOUND)
   message(FATAL_ERROR "Cannot find Eigen3.")
endif (EIGEN3_FOUND)

#add_subdirectory(elibs)
include_directories(SYSTEM
#    ${CMAKE_SOURCE_DIR}/elibs/rayint/libs
    ${CMAKE_SOURCE_DIR}/rayint/libs
    
#    ${CMAKE_SOURCE_DIR}/elibs/mve/libs
    ${CMAKE_SOURCE_DIR}/mve/libs
    
    ${EIGEN3_INCLUDE_DIRS}
    
#    ${CMAKE_SOURCE_DIR}/elibs/mapmap
#    ${CMAKE_SOURCE_DIR}/elibs/mapmap/mapmap
#    ${CMAKE_SOURCE_DIR}/elibs/mapmap/ext/dset
     ${CMAKE_SOURCE_DIR}/mapmap
     ${CMAKE_SOURCE_DIR}/mapmap/mapmap
     ${CMAKE_SOURCE_DIR}/mapmap/ext/dset
)

include_directories(
    libs
)

# link_directories(
#     ${CMAKE_SOURCE_DIR}/elibs/mve/libs/mve
#     ${CMAKE_SOURCE_DIR}/elibs/mve/libs/util
# )

add_subdirectory(libs)
add_subdirectory(apps)

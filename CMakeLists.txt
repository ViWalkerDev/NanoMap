cmake_minimum_required(VERSION 3.15)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE Release)
set (CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set (CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
project(nanomap_ros)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  # Allow the user to provide CMAKE_CUDA_ARCHITECTURES
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 53)
  endif()
endif()
enable_language(CUDA)

set(NANOVDB_CUDA_EXTENDED_LAMBDA "--expt-extended-lambda")
if(CUDA_VERSION_MAJOR GREATER_EQUAL 11)
  set(NANOVDB_CUDA_EXTENDED_LAMBDA "--extended-lambda")
endif()

set(CMAKE_CUDA_FLAGS "${NANOVDB_CUDA_EXTENDED_LAMBDA} -use_fast_math -lineinfo ${CMAKE_CUDA_FLAGS}")

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set (CUDA_INCLUDE_DIR "/usr/local/cuda/include")


include_directories(${catkin_INCLUDE_DIRS})
set(EIGEN3_INCLUDE_DIR "$ENV{HOME}/github/third-party/lib/eigen3/include/")
IF( NOT EIGEN3_INCLUDE_DIR )
    MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
ENDIF()
find_package(Eigen3 REQUIRED)
set(TBB_ROOT "$ENV{HOME}/github/third-party/lib/tbb/lib/cmake/tbb")
find_package(TBB REQUIRED)
find_package(CUDA REQUIRED)

set(Blosc_ROOT "$ENV{HOME}/github/third-party/lib/blosc")
set(OpenVDB_LIBRARYDIR "$ENV{HOME}/github/third-party/lib/openvdb/lib")
set(CMAKE_MODULE_PATH "$ENV{HOME}/github/third-party/lib/openvdb/lib/cmake/OpenVDB")
find_package(OpenVDB REQUIRED)
set(nanomap_DIR "$ENV{HOME}/github/first-party/lib/nanomap/lib/cmake/nanomap")
find_package(nanomap REQUIRED)
set(NANOMAP_INCLUDE_DIR "$ENV{HOME}/github/first-party/lib/nanomap/include")
set(dependencies "geometry_msgs" "nanomap_msgs" "roscpp" "rospy" "message_filters" "std_msgs")

add_library(nanomap_ros
    INTERFACE
    )


target_link_libraries(nanomap_ros
    INTERFACE
    ${OpenVDB_LIBRARIES}
    ${CUDA_LIBRARIES}
    -ltbb
    -lHalf
    $ENV{HOME}/github/first-party/lib/nanomap/lib/libnanomap.a
    )

target_include_directories(nanomap_ros
    INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      ${NANOMAP_INCLUDE_DIR}
      ${OpenVDB_INCLUDE_DIR}
      ${CUDA_INCLUDE_DIR}
      )

find_package(catkin REQUIRED COMPONENTS
          message_filters
          geometry_msgs
          sensor_msgs
          roscpp
          rospy
          std_msgs
          nanomap_msgs
      )

catkin_package()

add_executable(server src/server.cpp)

target_link_libraries(server PUBLIC nanomap_ros ${catkin_LIBRARIES})

set_property(TARGET server PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(server PUBLIC ${catkin_INCLUDE_DIRS})

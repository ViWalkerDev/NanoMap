cmake_minimum_required(VERSION 3.18)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set (CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set (CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
Set(CUDA_SEPARABLE_COMPILATION ON)
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()
project(nanomap_sim)
set(nanomap_VERSION 0.9)


if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  # Allow the user to provide CMAKE_CUDA_ARCHITECTURES
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 72)
  endif()
endif()
enable_language(CUDA)

set(NANOVDB_CUDA_EXTENDED_LAMBDA "--expt-extended-lambda")
if(CUDA_VERSION_MAJOR GREATER_EQUAL 11)
  set(NANOVDB_CUDA_EXTENDED_LAMBDA "--extended-lambda")
endif()

set(CMAKE_CUDA_FLAGS "${NANOVDB_CUDA_EXTENDED_LAMBDA} -use_fast_math -lineinfo -rdc=true ${CMAKE_CUDA_FLAGS}")

## DEPENDENCY SETUP
##BLOSC
set(Blosc_ROOT "$ENV{HOME}/github/third-party/lib/blosc")

##EIGEN
##list(APPEND CMAKE_MODULE_PATH "/usr/local/share/eigen3/cmake")
##list(APPEND CMAKE_MODULE_PATH "$ENV{HOME}/github/third-party/lib/eigen3/share/eigen3/cmake/")
set(Eigen3_DIR "$ENV{HOME}/github/third-party/src/eigen-3.3.9/build")
##set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
find_package(Eigen3 3.3.9 REQUIRED)

##TBB
find_package(TBB REQUIRED)

##OPENVDB
set(OpenVDB_LIBRARYDIR "$ENV{HOME}/github/third-party/lib/openvdb/lib")
list(APPEND CMAKE_MODULE_PATH "$ENV{HOME}/github/third-party/lib/openvdb/lib/cmake/OpenVDB")
find_package(OpenVDB REQUIRED)

##NANOVDB
set (NanoVDB_INCLUDE_DIR "$ENV{HOME}/github/third-party/lib/openvdb/include")

##CUDA
set (CUDA_INCLUDE_DIR "/usr/local/cuda/include")
find_package(CUDA REQUIRED)


###############################################################################
## NanoMap Library Definition

set(NANOMAP_KERNEL_FILES
    src/nanomap/kernels/generateCloud.cu
    src/nanomap/kernels/filterCloud.cu
    src/nanomap/kernels/frustumCastCloud.cu
  )

set(NANOMAP_SRC_FILES
    src/nanomap/mapgen/procedural/CaveGen.cpp
    src/nanomap/handler/BlockWorker.cpp
    src/nanomap/handler/Handler.cpp
    src/nanomap/handler/SimHandler.cpp
    src/nanomap/config/Config.cpp
)

add_library(nanomap
  STATIC
    ${NANOMAP_KERNEL_FILES}
    ${NANOMAP_SRC_FILES}
    )
target_link_libraries(nanomap
  PUBLIC
    ${OpenVDB_LIBRARIES}
    ${CUDA_LIBRARIES}
    -ltbb
    -lHalf
    )
target_include_directories(nanomap
  PUBLIC
      ${NanoVDB_INCLUDE_DIR}
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      ${EIGEN3_INCLUDE_DIR}
      ${OpenVDB_INCLUDE_DIR}
      ${CUDA_INCLUDE_DIR}
      )

set_property(TARGET nanomap PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

###############################################################################
##Map Generation Executables

add_executable(test3DCaveGen src/nanomap/test/test3DCaveGen.cpp)

target_link_libraries(test3DCaveGen PUBLIC nanomap)

add_executable(test2DCaveGen src/nanomap/test/test2DCaveGen.cpp)

target_link_libraries(test2DCaveGen PUBLIC nanomap)


###############################################################################
##Installation instructions

install(TARGETS nanomap EXPORT nanomaptargets
LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
INCLUDES DESTINATION ${CMAKE_INSTALL_PREFIX}/include
DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
set(NANOMAP_INCLUDE_DIR
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
)
install(
  DIRECTORY
    ${NANOMAP_INCLUDE_DIR}
  DESTINATION
    ${CMAKE_INSTALL_PREFIX}/include
  COMPONENT
    Devel
  FILES_MATCHING PATTERN "*.h*"

)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/nanomap/nanomap.cmake"
  VERSION ${nanomap_VERSION}
  COMPATIBILITY AnyNewerVersion
)

export(EXPORT nanomaptargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/nanomap/nanomaptargets.cmake"
  NAMESPACE nanomap::
)
configure_file(cmake/nanomapConfig.cmake
  "${CMAKE_CURRENT_BINARY_DIR}/nanomap/nanomapConfig.cmake"
  COPYONLY
)

set(ConfigPackageLocation lib/cmake/nanomap)
install(EXPORT nanomaptargets
  FILE
    nanomaptargets.cmake
  NAMESPACE
    nanomap::
  DESTINATION
    ${CMAKE_INSTALL_PREFIX}/${ConfigPackageLocation}
)
install(
  FILES
    cmake/nanomapConfig.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/nanomap/nanomapConfig.cmake"
  DESTINATION
    ${CMAKE_INSTALL_PREFIX}/${ConfigPackageLocation}
  COMPONENT
    Devel
)

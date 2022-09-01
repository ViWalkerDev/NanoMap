

# NanoMap Library

The aim of the NanoMap library is to provide accelerated occupancy mapping and simulation capabilities for robotic agents and systems equipped with CUDA capable GPUs. Only frustum style sensors such as RGB-D and Stereo Cameras have GPU accelerated support. LIDAR sensors can still be processed on the CPU, but due to the sparsity of the information they provide, do not benefit from the current methods used to accelerate frustum style sensors. Publication is pending acceptance, citation details will provided as soon as they are available. 

## Disclaimer

This library is currently in beta and under active development. 

The library is provided by the author “AS IS.” WITHOUT WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY.

This document is currently under construction and is subject to change at any time.

## Dependencies
These are the main dependencies, currently this is not guaranteed to be an exhaustive list. 

  * CUDA >= 10.0
  * CUB (This should be a part of the CUDA toolkit, otherwise install it separately)
  * Blosc == 1.5.0
  * Eigen3 == 3.3.9
  * Intel Threaded Building Blocks (apt-get install libtbb-dev)
  * OpenVDB built from source with NanoVDB support 

## Currently Supported and tested Platforms

  * Desktops and Laptops with CUDA capable GPUs running ubuntu 20.04.3 LTS
  * Jetson Nano running latest compatible JetPack. (The jetson nano specific code will be uploaded shortly)
  
## Building NanoMap
NanoMap is built using CMAKE >= 3.18.

Make sure you have built or installed the required dependencies listed above. Currently the CMakeLists.txt uses hardcoded paths to most dependencies, so edit the CMakeLists.txt file to point to the locations where they are built/installed.

Make a build directory in the NanoMap source folder and from inside run:

`cmake .. -DCMAKE_INSTALL_PREFIX="Your_Desired_Install_Location"` 

then

`make install`

## Supplemental Packages

Once installed, the library is ready to use. It should build two executables that can be used to randomly generate simulation environments for use with the simulation components. The library itself is designed to be used by other packages. See nanomap_ros and nanomap_benchmark for more information and examples of usage. 

[nanomap_ros](https://github.com/ViWalkerDev/nanomap_ros) is a ROS package targeted for ROS1 and ROS2 that acts as an interface to use NanoMap with robotic agents. 

[nanomap_benchmark](https://github.com/ViWalkerDev/nanomap_benchmark) is another package used for benchmarking the performance of NanoMap and its different modes of operation against Octomap.

[nanomap_msgs](https://github.com/ViWalkerDev/nanomap_msgs) is a msg package that is used by nanomap_ros for sending openvdb grid objects between nodes.

[nanomap_rviz2_plugins](https://github.com/ViWalkerDev/nanomap_rviz2_plugins) is an rviz2 plugin for rendering the openvdb grid messages defined by nanomap_msgs and sent by nanomap_ros.

## Performance
[This](https://youtu.be/UBrlLRqY_E4) is a video of the sensor simulation and map generation capabilities provided by NanoMap and the nanomap_ros package. Functionality is basic, but performance is good. The time to generate and then process the pointclouds took between 2-10ms for a frustum sensor configuration with 10m range and 20-40ms for a LIDAR with 20m range. The test was performed on a laptop with a Ryzen 4900HS and RTX 2060 MaxQ GPU. 

On the Jetson Nano, by taking advantage of the GPU, NanoMap can provide a 5x performance improvement over OctoMap at mapping resolutions of 0.1m, the performance difference only grows at higher resolutions. The Jetson Nano using NanoMap is capable of processing a kinect style depth camera sensor input capped at 5m in approximately 10ms. At 0.05m mapping resolution the same sensor can be processed in approximately 30ms. 

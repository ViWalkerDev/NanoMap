// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file SimHandler.h
///
/// @author Violet Walker
///
#ifndef NANOMAP_HANDLER_SIMHANDLER_H_INCLUDED
#define NANOMAP_HANDLER_SIMHANDLER_H_INCLUDED
#include <iomanip>
#include <chrono>
#include <iostream>
#include <string>
#include <mutex>

#include <tbb/parallel_for.h>

#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/math/Ray.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/math/DDA.h>

#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>

#include "nanomap/gpu/NodeWorker.h"
#include "nanomap/gpu/PointCloud.h"
#include "nanomap/gpu/Sensor.h"
#include "nanomap/gpu/GridData.h"
#include "nanomap/sensor/SensorData.h"
#include "nanomap/map/OccupancyMap.h"
#include "nanomap/nanomap.h"
#include "nanomap/gpu/SensorBucket.h"
#include "nanomap/allocator/SensorAllocator.h"
#include "nanomap/handler/BlockWorker.h"
#include "nanomap/config/Config.h"

// The following functions are called by the host and launch the gpu kernels.

//This kernel generates a point cloud using a simulated sensor view and an existing map
extern "C" void generateCloud(nanomap::gpu::SensorBucket& _sensorBucket, nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& _simGridHandle, cudaStream_t s0);

//This kernel filters the cloud. Filter functionality depends on user defined settings provided at runtime
extern "C" void filterCloud(nanomap::gpu::SensorBucket& _sensorBucket, cudaStream_t s0);

//This kernel raycasts the cloud over the area that is observable by the sensor in question
extern "C" void frustumCastCloud(nanomap::gpu::SensorBucket& _sensorBucket, cudaStream_t s0,  cudaStream_t s1);


namespace nanomap{
  namespace handler{
    using Map = nanomap::map::Map;
    using FloatGrid = openvdb::FloatGrid;

    using TreeT = openvdb::FloatGrid::TreeType;
    using LeafT = TreeT::LeafNodeType;
    using AccessorT = openvdb::tree::ValueAccessor<TreeT>;
    using VecTreeT = openvdb::Vec3DGrid::TreeType;
    using VecAccessorT = openvdb::tree::ValueAccessor<VecTreeT>;
    using BufferT = nanovdb::CudaDeviceBuffer;
    using ValueT  = float;
    using Vec3T   = nanovdb::Vec3<ValueT>;
    using RayT = nanovdb::Ray<ValueT>;
    using HDDA     = nanovdb::HDDA<RayT>;
    using SensorData = nanomap::sensor::SensorData;
    using IterType = TreeT::ValueOnIter;

    using EigenVec = Eigen::Matrix<ValueT, 3, 1>;
    using EigenMat = Eigen::Matrix<ValueT, 3, 3>;
    using Quaternion = Eigen::Quaternion<ValueT>;

    class SimHandler{
      public:
        SimHandler(std::shared_ptr<nanomap::config::Config> config, openvdb::FloatGrid::Ptr simGrid);
        void populateTempGrid(openvdb::FloatGrid::Accessor& tempAcc, int sensorIndex, std::shared_ptr<nanomap::map::Map> map);

        void integrateTempGrid(openvdb::FloatGrid::Ptr tempGrid, std::shared_ptr<nanomap::map::Map> map);
        void processPointCloudCPU(int index,  std::shared_ptr<nanomap::map::Map> map);
        void voxelUpdateFromFrustumBuffer(int nodeEdge);
        void blockUpdateFromFrustumBuffer();
        void integrateTempGrid(openvdb::FloatGrid::Ptr tempGrid,
                               openvdb::FloatGrid::Ptr Grid,
                               openvdb::FloatGrid::Accessor& acc,
                               float emptyClampThres,
                               float occClampThres,
                               float logodds_thres_min,
                               float logodds_thres_max);
        void processPointCloud(int sensorIndex,
                               std::shared_ptr<nanomap::map::Map> agentMap);
        void closeHandler();
        void printUpdateTime(int count);

      private:
        cudaStream_t                                                    _s0;
        cudaStream_t                                                    _s1;
        float _mapUpdateTime;
        float _gpuTime;
        std::shared_ptr<nanomap::map::Map>  _agentMap;
        std::shared_ptr<nanomap::config::Config> _config;
        openvdb::FloatGrid::Ptr _simGrid;
        nanomap::allocator::SensorAllocator _sensorAllocator;
        nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> _laserHandle;
        nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> _simGridHandle;


      };


  }
}
#endif

// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file BlockWorker.h
///
/// @author Violet Walker
///

#ifndef NANOMAP_HANDLER_BLOCKWORKER_H_INCLUDED
#define NANOMAP_HANDLER_BLOCKWORKER_H_INCLUDED
#include <iomanip>
#include <chrono>
#include <iostream>
#include <string>
#include <mutex>

#include <tbb/parallel_for.h>

#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include <openvdb/Types.h>
#include <openvdb/math/Coord.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/Grid.h>

#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/Ray.h> // for nanovdb::Ray
#include <nanovdb/util/HDDA.h>

#include "nanomap/gpu/NodeWorker.h"
#include "nanomap/gpu/PointCloud.h"
#include "nanomap/gpu/Sensor.h"
#include "nanomap/gpu/GridData.h"
#include "nanomap/sensor/SensorData.h"
#include "nanomap/map/OccupancyMap.h"
#include "nanomap/nanomap.h"

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

    class BlockWorker
    {

    public:
      struct Block {
          openvdb::CoordBBox      bbox;
          LeafT*                  leaf;
          std::pair<ValueT, bool> node;
          Block(const openvdb::CoordBBox& b) : bbox(b), leaf(nullptr) {}
      };
      std::vector<Block>* _blocks;
      std::vector<LeafT*>* _leaves;

      BlockWorker(int nodeEdge, float occClampThres, float emptyClampThres, float logOddsHitThres,
                  float logOddsMissThres, std::shared_ptr<AccessorT> accessor, int8_t* hostVoxelBuffer,
                  int* hostNodeBuffer, int hostCount);
      void destroyBlocks();
      void processBlocks(bool serial);
      void operator()(const tbb::blocked_range<size_t> &r) const;


    private:

      int _voxelVolume;
      int _nodeEdge;
      float _occupiedClampingThreshold;
      float _emptyClampingThreshold;
      float _logOddsHitThreshold;
      float _logOddsMissThreshold;

      int8_t* _hostVoxelBuffer;

      void fillOccupiedLeafFromBuffer(LeafT* leaf, int index) const;

      void combineOccupiedLeafFromBuffer(LeafT*                          leaf,
                                         LeafT*                        target,
                                         int                            index) const;


    };

  }
}
#endif

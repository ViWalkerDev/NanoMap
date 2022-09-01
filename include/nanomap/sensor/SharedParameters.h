// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file SharedParameters.h
///
/// @author Violet Walker
///
#ifndef NANOMAP_SENSOR_SHAREDPARAMETERS_H_HAS_BEEN_INCLUDED
#define NANOMAP_SENSOR_SHAREDPARAMETERS_H_HAS_BEEN_INCLUDED
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <string>
#include <nanovdb/NanoVDB.h>
#include "nanomap/nanomap.h"
#include <cfenv>
#include <cmath>
using ValueT = float;

namespace nanomap{
  namespace sensor{
    class SharedParameters{

      public:
        int _type;
        int _rate;
        float _frameTime;
        ValueT _gridRes;
        int _hRes;
        int _vRes;
        int _pclSize;
        ValueT _aspect;
        ValueT _vfov;
        ValueT _hfov;
        nanomap::Pose _pose;
        Eigen::Matrix<ValueT, 3, 1> _voxelPosition;
        Eigen::Matrix<int, 3, 1> _leafOffset;
        Eigen::Matrix<int, 3, 1> _voxelOffset;
        Eigen::Matrix<ValueT, 3, 1> _leafOriginOffset;
        Eigen::Matrix<ValueT, 3, 1> _voxelOriginOffset;
        Eigen::Matrix<ValueT, 3, 9> _sensorBounds;
        Eigen::Matrix<ValueT, 3, 9> _worldBounds;
        Eigen::Matrix<ValueT, 3, 9> _transformedBounds;
        Eigen::Matrix<ValueT, 3, 9> _voxelTransformedBounds;
        Eigen::Matrix<ValueT, 3, 9> _leafTransformedBounds;
        Eigen::Matrix<ValueT, 3, 5> _sensorNorms;
        Eigen::Matrix<ValueT, 3, 5> _transformedNorms;
        Eigen::Matrix<ValueT, 3, 3> _frameTransform;
        nanovdb::CoordBBox _leafBounds;
        nanovdb::CoordBBox _voxelBounds;
        Eigen::Vector3f _worldMax;
        Eigen::Vector3f _worldMin;
        ValueT _probHit;
        ValueT _probMiss;

        ValueT _maxRange;
        ValueT _minRange;
        ValueT _minVoxelRange;
        ValueT _maxVoxelRange;

        int _leafEdge;
        int _maxLeafBufferSize;
        int _maxVoxelBufferSize;
        int _leafBufferSize;

      };
  }
}
#endif

// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file SensorData.h
///
/// @author Violet Walker
///
#ifndef NANOMAP_SENSOR_SENSORDATA_H_HAS_BEEN_INCLUDED
#define NANOMAP_SENSOR_SENSORDATA_H_HAS_BEEN_INCLUDED
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <string>
#include <nanovdb/NanoVDB.h>
#include "nanomap/nanomap.h"
#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/openvdb.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/math/Coord.h>
#include <openvdb/math/Ray.h>
#include <openvdb/math/DDA.h>
#include "nanomap/sensor/SharedParameters.h"
#include <cfenv>
#include <cmath>
using ValueT = float;

namespace nanomap{
  namespace sensor{
    class SensorData{

      public:
        SensorData(){}

        virtual void calculateSensorBounds(){}
        virtual void calculateBufferSize(){}
        void updatePose(nanomap::Pose pose){
          #pragma STDC FENV_ACCESS ON
          std::fesetround(FE_DOWNWARD);
          _shared._pose = pose;
          if(_shared._pose.position(0) == 0.0){
            _shared._voxelPosition(0) = 0;
            _shared._leafOffset(0) = 0;
            _shared._leafOriginOffset(0) = 0;
            _shared._voxelOffset(0) = 0;
            _shared._voxelOriginOffset(0) = 0;
          }else{
            _shared._voxelPosition(0) = _shared._pose.position(0)/_shared._gridRes;
            _shared._leafOffset(0) = std::floor(std::nearbyint(_shared._voxelPosition(0))/_shared._leafEdge)*_shared._leafEdge;
            _shared._leafOriginOffset(0) = _shared._voxelPosition(0) - _shared._leafOffset(0);
            _shared._voxelOffset(0) = std::nearbyint(_shared._voxelPosition(0));
            _shared._voxelOriginOffset(0) = _shared._voxelPosition(0)-_shared._voxelOffset(0);
          }
          if(_shared._pose.position(1)==0.0){
            _shared._voxelPosition(1) = 0;
            _shared._leafOffset(1) = 0;
            _shared._leafOriginOffset(1) = 0;
            _shared._voxelOffset(1) = 0;
            _shared._voxelOriginOffset(1) = 0;
          }else{
            _shared._voxelPosition(1) = _shared._pose.position(1)/_shared._gridRes;
            _shared._leafOffset(1) = std::floor(std::nearbyint(_shared._voxelPosition(1))/_shared._leafEdge)*_shared._leafEdge;
            _shared._leafOriginOffset(1) = _shared._voxelPosition(1) - _shared._leafOffset(1);
            _shared._voxelOffset(1) = std::nearbyint(_shared._voxelPosition(1));
            _shared._voxelOriginOffset(1) = _shared._voxelPosition(1)-_shared._voxelOffset(1);
          }
          if(_shared._pose.position(2) == 0.0){
            _shared._voxelPosition(2) = 0;
            _shared._leafOffset(2) = 0;
            _shared._leafOriginOffset(2) = 0;
            _shared._voxelOffset(2) = 0;
            _shared._voxelOriginOffset(2) = 0;
          }else{
            _shared._voxelPosition(2) = _shared._pose.position(2)/_shared._gridRes;
            _shared._leafOffset(2) = std::floor(std::nearbyint(_shared._voxelPosition(2))/_shared._leafEdge)*_shared._leafEdge;
            _shared._leafOriginOffset(2) = _shared._voxelPosition(2) - _shared._leafOffset(2);
            _shared._voxelOffset(2) = std::nearbyint(_shared._voxelPosition(2));
            _shared._voxelOriginOffset(2) = _shared._voxelPosition(2)-_shared._voxelOffset(2);
          }
        }

        void rotateView(){
          Eigen::Matrix<float, 3, 3> rotation = _shared._pose.orientation.toRotationMatrix()*_shared._frameTransform;
          _shared._transformedBounds = rotation*_shared._sensorBounds;
          _shared._transformedNorms = rotation*_shared._sensorNorms;
          _shared._voxelTransformedBounds = _shared._transformedBounds.colwise()+_shared._voxelOriginOffset;
          _shared._leafTransformedBounds = _shared._transformedBounds.colwise() + _shared._leafOriginOffset;
          getBoundingBoxes();
        }

        void calculateSensorNorms(){
          _shared._sensorNorms.col(0) =
            ((_shared._sensorBounds.col(1)-_shared._sensorBounds.col(2)).cross(
              _shared._sensorBounds.col(0)-_shared._sensorBounds.col(1))).normalized();
          _shared._sensorNorms.col(1) =
            (_shared._sensorBounds.col(5).cross(
              _shared._sensorBounds.col(4)-_shared._sensorBounds.col(5))).normalized();
          _shared._sensorNorms.col(2) =
            (_shared._sensorBounds.col(6).cross(
              _shared._sensorBounds.col(5)-_shared._sensorBounds.col(6))).normalized();
          _shared._sensorNorms.col(3) =
            (_shared._sensorBounds.col(7).cross(
              _shared._sensorBounds.col(6)-_shared._sensorBounds.col(7))).normalized();
          _shared._sensorNorms.col(4) =
            (_shared._sensorBounds.col(4).cross(
              _shared._sensorBounds.col(7)-_shared._sensorBounds.col(4))).normalized();
        }
        void getBoundingBoxes(){
          #pragma STDC FENV_ACCESS ON
          std::fesetround(FE_DOWNWARD);
          Eigen::Vector3f leafMax(_shared._leafTransformedBounds.row(0).maxCoeff(),
                              _shared._leafTransformedBounds.row(1).maxCoeff(),
                              _shared._leafTransformedBounds.row(2).maxCoeff());
          Eigen::Vector3f leafMin(_shared._leafTransformedBounds.row(0).minCoeff(),
                              _shared._leafTransformedBounds.row(1).minCoeff(),
                              _shared._leafTransformedBounds.row(2).minCoeff());
          nanovdb::CoordBBox leafBound(nanovdb::Coord(
                                                          std::floor((std::nearbyint(leafMin(0)))/_shared._leafEdge),
                                                          std::floor((std::nearbyint(leafMin(1)))/_shared._leafEdge),
                                                          std::floor((std::nearbyint(leafMin(2)))/_shared._leafEdge)),
                                       nanovdb::Coord(
                                                          std::floor((std::nearbyint(leafMax(0)))/_shared._leafEdge),
                                                          std::floor((std::nearbyint(leafMax(1)))/_shared._leafEdge),
                                                          std::floor((std::nearbyint(leafMax(2)))/_shared._leafEdge)));

          Eigen::Vector3f voxelMax(_shared._voxelTransformedBounds.row(0).maxCoeff(),
                              _shared._voxelTransformedBounds.row(1).maxCoeff(),
                              _shared._voxelTransformedBounds.row(2).maxCoeff());
          Eigen::Vector3f voxelMin(_shared._voxelTransformedBounds.row(0).minCoeff(),
                              _shared._voxelTransformedBounds.row(1).minCoeff(),
                              _shared._voxelTransformedBounds.row(2).minCoeff());
          nanovdb::CoordBBox voxelBound(nanovdb::Coord(
                                                        std::nearbyint(voxelMin(0)),
                                                        std::nearbyint(voxelMin(1)),
                                                        std::nearbyint(voxelMin(2))),
                                       nanovdb::Coord(
                                                        std::nearbyint(voxelMax(0)),
                                                        std::nearbyint(voxelMax(1)),
                                                        std::nearbyint(voxelMax(2))));
          _shared._leafBounds = leafBound;
          _shared._voxelBounds = voxelBound;
          _shared._leafBufferSize = leafBound.dim().x()*leafBound.dim().y()*leafBound.dim().z();
        }

        std::string sensorName(){return _sensorName;}
        int getId(){return _sensorId;}
        std::string getName(){return _sensorName;}
        nanomap::sensor::SharedParameters sharedParameters(){return _shared;}
        openvdb::Int32Grid::Ptr laserGrid(){return _laserGrid;}
        nanovdb::Coord* laserIndex(){return _laserIndex;}
        int laserVoxelCount(){return  _laserVoxelCount;}
        Eigen::Matrix<ValueT, 3, Eigen::Dynamic>& pointCloud(){return _pointCloud;}
      protected:
        std::string _sensorName;
        int _sensorId;
        openvdb::Int32Grid::Ptr _laserGrid;
        nanovdb::Coord* _laserIndex;
        int _laserVoxelCount;

        nanomap::sensor::SharedParameters _shared;

        Eigen::Matrix<ValueT, 3, Eigen::Dynamic> _pointCloud;
      };
  }
}
#endif

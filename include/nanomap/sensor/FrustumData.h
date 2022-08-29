// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file FrustumData.h
///
/// @author Violet Walker
///
#ifndef NANOMAP_SENSOR_FRUSTUMDATA_H_HAS_BEEN_INCLUDED
#define NANOMAP_SENSOR_FRUSTUMDATA_H_HAS_BEEN_INCLUDED
#include "SensorData.h"


using ValueT = float;
using EigenVec = Eigen::Matrix<ValueT, 3, 1>;
namespace nanomap{
  namespace sensor{
    class FrustumData : public SensorData{

      public:
        FrustumData(std::string sensorName, int sensorId, Eigen::Matrix<float, 3, 3> frameTransform, ValueT gridRes, int leafEdge, int hRes, int vRes,
                    ValueT vFOV, int rate, ValueT maxRange, ValueT minRange, ValueT probHit, ValueT probMiss)
        {
          _shared._vfov = vFOV;
          _shared._aspect = ((float)hRes)/((float)vRes);
          _shared._type = 0;
          _shared._hRes = hRes;
          _shared._vRes = vRes;
          _shared._rate = rate;
          _shared._frameTime = 1/rate;
          _shared._frameTransform = frameTransform;
          _shared._pclSize = hRes*vRes;
          _shared._probHit = probHit;
          _shared._probMiss = probMiss;
          _shared._maxRange = maxRange;
          _shared._minRange = minRange;
          _shared._minVoxelRange = minRange/gridRes;
          _shared._maxVoxelRange = maxRange/gridRes;
          _sensorId = sensorId;
          _sensorName = sensorName;
          _shared._gridRes = gridRes;
          _shared._leafEdge = leafEdge;
          _pointCloud.resize(3,_shared._pclSize);
          calculateSensorBounds();
          calculateSensorNorms();
          calculateBufferSize();

        }


        void calculateSensorBounds(){
          ValueT halfHeight = _shared._maxVoxelRange*(tan(_shared._vfov*M_PI/360));
          ValueT halfWidth = _shared._aspect*halfHeight;
          ValueT closeHeight = _shared._minVoxelRange*(tan(_shared._vfov*M_PI/360));
          ValueT closeWidth = _shared._aspect*closeHeight;
          _shared._sensorBounds.col(0) = EigenVec( -closeWidth, -closeHeight,_shared._minVoxelRange);
          _shared._sensorBounds.col(1) = EigenVec( -closeWidth, closeHeight,_shared._minVoxelRange);
          _shared._sensorBounds.col(2) = EigenVec( closeWidth, closeHeight,_shared._minVoxelRange);
          _shared._sensorBounds.col(3) = EigenVec( closeWidth, -closeHeight,_shared._minVoxelRange);
          _shared._sensorBounds.col(4) = EigenVec( -halfWidth, -halfHeight,_shared._maxVoxelRange);
          _shared._sensorBounds.col(5) = EigenVec(-halfWidth, halfHeight,_shared._maxVoxelRange);
          _shared._sensorBounds.col(6) = EigenVec(halfWidth, halfHeight,_shared._maxVoxelRange);
          _shared._sensorBounds.col(7) = EigenVec( halfWidth, -halfHeight,_shared._maxVoxelRange);
          _shared._sensorBounds.col(8) = EigenVec(0.0f, 0.0f, 0.0f);
          halfHeight = _shared._maxRange*(tan(_shared._vfov*M_PI/360));
          halfWidth = _shared._aspect*halfHeight;
          closeHeight = _shared._minRange*(tan(_shared._vfov*M_PI/360));
          closeWidth = _shared._aspect*closeHeight;
          _shared._worldBounds.col(0) = EigenVec( -closeWidth, -closeHeight,_shared._minRange);
          _shared._worldBounds.col(1) = EigenVec( -closeWidth, closeHeight,_shared._minRange);
          _shared._worldBounds.col(2) = EigenVec( closeWidth, closeHeight,_shared._minRange);
          _shared._worldBounds.col(3) = EigenVec( closeWidth, -closeHeight,_shared._minRange);
          _shared._worldBounds.col(4) = EigenVec( -halfWidth, -halfHeight,_shared._maxRange);
          _shared._worldBounds.col(5) = EigenVec(-halfWidth, halfHeight,_shared._maxRange);
          _shared._worldBounds.col(6) = EigenVec(halfWidth, halfHeight,_shared._maxRange);
          _shared._worldBounds.col(7) = EigenVec( halfWidth, -halfHeight,_shared._maxRange);
          _shared._worldBounds.col(8) = EigenVec(0.0f, 0.0f, 0.0f);
          _shared._worldMax = Eigen::Vector3f(_shared._worldBounds.row(0).maxCoeff(),
                              _shared._worldBounds.row(1).maxCoeff(),
                              _shared._worldBounds.row(2).maxCoeff());
          _shared._worldMin = Eigen::Vector3f(_shared._worldBounds.row(0).minCoeff(),
                              _shared._worldBounds.row(1).minCoeff(),
                              _shared._worldBounds.row(2).minCoeff());
          }

        void calculateBufferSize(){
          #pragma STDC FENV_ACCESS ON
          std::fesetround(FE_DOWNWARD);
          int leafBufferSize;
          int roll, pitch, yaw;
          float leafEdgeFactor = ((float)(_shared._leafEdge)-0.5);

          _shared._maxLeafBufferSize = 0;


          Eigen::Matrix<float, 3, 9> rotatedBounds;
          Eigen::Matrix<float, 3, 9> offsetBounds;
          Eigen::Matrix<float, 3, 8> offset;
          Eigen::Quaternionf q = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX())
                                  * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY())
                                    * Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
          Eigen::Matrix<float, 3, 3> r = q.toRotationMatrix();
          for(roll = 0; roll < 90; roll ++){
            for(pitch = 0; pitch < 180; pitch ++){
              for(yaw = 0; yaw < 360; yaw ++ ){
                q = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX())
                                        * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY())
                                          * Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
                r = q.toRotationMatrix();
                rotatedBounds = r*_shared._sensorBounds;
                  Eigen::Vector3f max(rotatedBounds.row(0).maxCoeff(),
                                      rotatedBounds.row(1).maxCoeff(),
                                      rotatedBounds.row(2).maxCoeff());
                  Eigen::Vector3f min(rotatedBounds.row(0).minCoeff(),
                                      rotatedBounds.row(1).minCoeff(),
                                      rotatedBounds.row(2).minCoeff());
                  nanovdb::CoordBBox leafBound(nanovdb::Coord(
                                                  std::floor(std::nearbyint(min(0))/_shared._leafEdge),
                                                  std::floor(std::nearbyint(min(1))/_shared._leafEdge),
                                                  std::floor(std::nearbyint(min(2))/_shared._leafEdge)),
                                               nanovdb::Coord(
                                                  std::floor(std::nearbyint(max(0))/_shared._leafEdge),
                                                  std::floor(std::nearbyint(max(1))/_shared._leafEdge),
                                                  std::floor(std::nearbyint(max(2))/_shared._leafEdge)));
                  leafBufferSize = leafBound.dim().x()*leafBound.dim().y()*leafBound.dim().z();
                  if(_shared._maxLeafBufferSize  < leafBufferSize){
                    _shared._maxLeafBufferSize = leafBufferSize;
                  }
                }
              }
            }
          }

      protected:


      };
  }
}
#endif

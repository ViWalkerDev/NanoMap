// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file LaserData.h
///
/// @author Violet Walker
///
#ifndef NANOMAP_SENSOR_LASERDATA_H_HAS_BEEN_INCLUDED
#define NANOMAP_SENSOR_LASERDATA_H_HAS_BEEN_INCLUDED
#include "SensorData.h"
#include <nanovdb/util/CudaDeviceBuffer.h>
#include "nanomap/handler/handlerAssert.h"
using ValueT = float;
using EigenVec = Eigen::Matrix<ValueT, 3, 1>;
namespace nanomap{
  namespace sensor{
    class LaserData : public SensorData{

      public:
        LaserData(std::string sensorName, int sensorId, Eigen::Matrix<float, 3, 3> frameTransform, ValueT gridRes, int leafEdge, float aHRes, float aVRes,
                  ValueT hFOV, ValueT vFOV, int rate, ValueT maxRange, ValueT minRange, ValueT probHit, ValueT probMiss)
        {
          _shared._vfov = vFOV;
          _shared._hfov = hFOV;
          _shared._hRes = (hFOV/aHRes+1);
          _shared._vRes = (vFOV/aVRes+1);
          _scale[0] = hFOV/_shared._hRes;
          _scale[1] = vFOV/_shared._vRes;
          _shared._rate = rate;
          _shared._frameTime = 1/rate;
          _shared._type = 1;
          _shared._frameTransform = frameTransform;
          _shared._pclSize = _shared._hRes*_shared._vRes;
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
          //calculateBufferSize();

        }

      // void getLaserRay(int w, int h, openvdb::Vec3d& rayDir){
      //
      //         float u = -_shared._hfov/2+((float)w)*_scale[0]; // yaw angle in radians
      //         float v = -_shared._vfov/2+((float)h)*_scale[1]; // pitch angle in radians
      //         u *= 3.14159265358979323846 / 180;
      //         v *= 3.14159265358979323846 / 180;
      //         //Calculate unit direction of ray in xyz frame
      //         EigenVec matrix_dir(std::sin(u)*std::cos(v), std::sin(v),std::cos(u)*std::cos(v));
      //         //normalise ray
      //         matrix_dir.normalize();
      //         rayDir =  openvdb::Vec3d(matrix_dir(0),matrix_dir(1),matrix_dir(2));
      //
      //     }
      // void calculateBufferSize(){
      //   openvdb::Int32Grid::Ptr tempGrid = openvdb::Int32Grid::create(0);
      //   openvdb::Vec3d rayEyes[] = {
      //     openvdb::Vec3d{0.5,0.5,0.5},
      //     openvdb::Vec3d{0.0,0.0,0.0},
      //     openvdb::Vec3d{0.0,0.0,1.0},
      //     openvdb::Vec3d{0.0,1.0,0.0},
      //     openvdb::Vec3d{0.0,1.0,1.0},
      //     openvdb::Vec3d{1.0,0.0,0.0},
      //     openvdb::Vec3d{1.0,0.0,1.0},
      //     openvdb::Vec3d{1.0,1.0,0.0},
      //     openvdb::Vec3d{1.0,1.0,1.0}
      //   };
      //   auto acc = tempGrid->getAccessor();
      //   openvdb::math::Ray<double> ray;
      //   openvdb::math::DDA<openvdb::math::Ray<double>, 0> dda;
      //   for(int pos = 0; pos < 9; pos++){
      //     for(int w = 0; w < _shared._hRes; w++){
      //       for(int h = 0; h < _shared._vRes; h++){
      //         openvdb::Vec3d rayEye = rayEyes[pos];
      //         openvdb::Vec3d rayDir;
      //         getLaserRay(w, h, rayDir);
      //         ray.setEye(rayEye);
      //         ray.setDir(rayDir);
      //         dda.init(ray, _shared._minVoxelRange, _shared._maxVoxelRange);
      //         openvdb::Coord voxel = dda.voxel();
      //         while(dda.step()){
      //           acc.setValue(voxel, 1.0);
      //           voxel = dda.voxel();
      //         }
      //         acc.setValue(voxel, 1.0);
      //       }
      //     }
      //   }
      //   _laserVoxelCount = tempGrid->activeVoxelCount();
      //   std::cout << "laserVoxelCount = " << _laserVoxelCount << std::endl;
      //   cudaCheck(cudaMallocHost((void**)&_laserIndex, (_laserVoxelCount)*sizeof(nanovdb::Coord)));
      //
      //   int count = 0;
      //   _laserGrid = openvdb::Int32Grid::create(-1);
      //   auto laserAcc = _laserGrid->getAccessor();
      //   for (openvdb::Int32Grid::ValueOnCIter iter = tempGrid->cbeginValueOn(); iter; ++iter) {
      //     openvdb::Coord coord = iter.getCoord();
      //     _laserIndex[count] = nanovdb::Coord(coord.x(), coord.y(), coord.z());
      //     laserAcc.setValue(coord, count);
      //     count++;
      //     //std::cout << "Grid" << iter.getCoord() << " = " << *iter << std::endl;
      //   }
      // }
      //
      //
      void calculateSensorBounds(){
        ValueT ZMin, XMax, XMin, WorldZMin, WorldXMax, WorldXMin;
        ValueT YMax = _shared._maxVoxelRange*(sin(_shared._vfov*M_PI/360));
        ValueT WorldYMax = _shared._maxRange*(sin(_shared._vfov*M_PI/360));
        ValueT YMin = -YMax;
        ValueT WorldYMin = -WorldYMax;
        ValueT ZMax = _shared._maxVoxelRange;
        ValueT WorldZMax = _shared._maxRange;
        if(_shared._hfov < 180){
          ZMin = 0.0;
          WorldZMin = 0.0;
          XMax = _shared._maxVoxelRange*(sin(_shared._hfov*M_PI/360));
          XMin = - XMax;
          WorldXMax = _shared._maxRange*(sin(_shared._hfov*M_PI/360));
          WorldXMin = -WorldXMax;
        }else if(_shared._hfov>= 180 && _shared._hfov < 360){
          ZMin = _shared._maxVoxelRange*(cos(_shared._hfov*M_PI/360));
          WorldZMin = _shared._maxRange*(cos(_shared._hfov*M_PI/360));
          XMax = _shared._maxVoxelRange;
          XMin = -XMax;
          WorldXMax = _shared._maxRange;
          WorldXMin = -WorldXMax;
        }else if(_shared._hfov>=360){
          ZMin = -ZMax;
          WorldZMin = -WorldZMax;
          XMax = _shared._maxVoxelRange;
          XMin = -XMax;
          WorldXMax = _shared._maxRange;
          WorldXMin = -WorldXMax;
        }
        _shared._sensorBounds.col(0) = EigenVec( XMin, YMin, ZMin);
        _shared._sensorBounds.col(1) = EigenVec( XMin, YMax, ZMin);
        _shared._sensorBounds.col(2) = EigenVec( XMax, YMax, ZMin);
        _shared._sensorBounds.col(3) = EigenVec( XMax, YMin, ZMin);
        _shared._sensorBounds.col(4) = EigenVec( XMin, YMin, ZMax);
        _shared._sensorBounds.col(5) = EigenVec( XMin, YMax, ZMax);
        _shared._sensorBounds.col(6) = EigenVec( XMax, YMax, ZMax);
        _shared._sensorBounds.col(7) = EigenVec( XMax, YMin, ZMax);
        _shared._sensorBounds.col(8) = EigenVec(0.0f, 0.0f, 0.0f);
        _shared._worldBounds.col(0) = EigenVec( WorldXMin, WorldYMin, WorldZMin);
        _shared._worldBounds.col(1) = EigenVec( WorldXMin, WorldYMax, WorldZMin);
        _shared._worldBounds.col(2) = EigenVec( WorldXMax, WorldYMax, WorldZMin);
        _shared._worldBounds.col(3) = EigenVec( WorldXMax, WorldYMin, WorldZMin);
        _shared._worldBounds.col(4) = EigenVec( WorldXMin, WorldYMin, WorldZMax);
        _shared._worldBounds.col(5) = EigenVec( WorldXMin, WorldYMax, WorldZMax);
        _shared._worldBounds.col(6) = EigenVec( WorldXMax, WorldYMax, WorldZMax);
        _shared._worldBounds.col(7) = EigenVec( WorldXMax, WorldYMin, WorldZMax);
        _shared._worldBounds.col(8) = EigenVec(0.0f, 0.0f, 0.0f);
        _shared._worldMax = Eigen::Vector3f(_shared._worldBounds.row(0).maxCoeff(),
                            _shared._worldBounds.row(1).maxCoeff(),
                            _shared._worldBounds.row(2).maxCoeff());
        _shared._worldMin = Eigen::Vector3f(_shared._worldBounds.row(0).minCoeff(),
                            _shared._worldBounds.row(1).minCoeff(),
                            _shared._worldBounds.row(2).minCoeff());
      }

      protected:
        float _scale[2];

      };
  }
}
#endif

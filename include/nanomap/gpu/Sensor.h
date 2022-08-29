// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file Sensor.h
///
/// @author Violet Walker
///
/// @brief a sensor class for managing sensor information for CUDA kernels.

#ifndef NANOMAP_GPU_SENSOR_H_HAS_BEEN_INCLUDED
#define NANOMAP_GPU_SENSOR_H_HAS_BEEN_INCLUDED


#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Ray.h>
#include <nanomap/sensor/SharedParameters.h>
#include <cfenv>

namespace nanomap{
  namespace gpu{

//Contains all variables and functions to perform Frustum or Laser updates on GPU
//What is performed depends on sensorData type acquired during update().
//combined class instead of inheritance is used to avoid virtual issues
//when moving between CPU and GPU.
template<typename ValueT = ValueT, typename Vec3T = nanovdb::Vec3<ValueT>,typename RayT = nanovdb::Ray<ValueT>,typename EigenVec = Eigen::Matrix<ValueT, 3, 1>, typename EigenMat = Eigen::Matrix<ValueT, 3, 3>>
class Sensor
{

public:

    __hostdev__ Sensor()
      :  m_gridRes(0.0)
      ,  m_leafOriginOffset(0.0,0.0,0.0)
      ,  m_voxelOriginOffset(0.0,0.0,0.0)
      ,  m_leafOffset(0,0,0)
      ,  m_worldMin(0.0,0.0,0.0)
      ,  m_worldMax(0.0,0.0,0.0)
      ,  m_maxRange(0.0)
      ,  m_minRange(0.0)
      ,  m_minVoxelRange(0)
      ,  m_maxVoxelRange(0)
      ,  m_logOddsHit(0.0)
      ,  m_logOddsMiss(0.0)
      ,  m_vRes(0)
      ,  m_hRes(0)
      {
      }

    __host__ void updateLaser(nanomap::sensor::SharedParameters shared){
        updateShared(shared);
        m_hFOV = shared._hfov;
        m_vFOV = shared._vfov;
        m_scale[0] = m_hFOV/m_hRes;
        m_scale[1] = m_vFOV/m_vRes;
        m_transformedViewVecs =  m_rotation;
    }

    __host__ void updateFrustum(nanomap::sensor::SharedParameters shared){
      updateShared(shared);
      float aspect = shared._aspect;
      float vfov = shared._vfov;
      float halfHeight = ValueT(tan(vfov * 3.14159265358979323846 / 360));
      float halfWidth = aspect * halfHeight;
      m_scale[0] = (float)(1.0/m_hRes);
      m_scale[1] = (float)(1.0/m_vRes);
      m_transformedViewVecs =  m_rotation;
      m_transformedViewVecs.col(2) = halfWidth * m_transformedViewVecs.col(0) + halfHeight * m_transformedViewVecs.col(1) - m_transformedViewVecs.col(2);
      m_transformedViewVecs.col(0) *= 2 * halfWidth;
      m_transformedViewVecs.col(1) *= 2 * halfHeight;
      m_identityVecs.col(2) = halfWidth * Eigen::Vector3f(1.0,0.0,0.0) + halfHeight * Eigen::Vector3f(0.0,1.0,0.0)-Eigen::Vector3f(0.0,0.0,1.0);;
      m_identityVecs.col(0) = Eigen::Vector3f(1.0,0.0,0.0) * 2 * halfWidth;
      m_identityVecs.col(1) = Eigen::Vector3f(0.0,1.0,0.0) * 2 * halfHeight;
    }
    __host__ void updateShared(nanomap::sensor::SharedParameters& shared)
     {
       m_defaultViewVecs = shared._frameTransform;
       m_gridRes = shared._gridRes;
       m_position = shared._pose.position;
       m_voxelPosition = shared._voxelPosition;
       m_leafOffset = shared._leafOffset;
       m_leafOriginOffset = shared._leafOriginOffset;
       m_voxelOriginOffset = shared._voxelOriginOffset;
       Eigen::Quaternionf q = shared._pose.orientation.normalized();
       m_rotation = q.toRotationMatrix()*m_defaultViewVecs;
       m_logOddsHit = log(shared._probHit)-log(1-shared._probHit);
       m_logOddsMiss = log(shared._probMiss)-log(1-shared._probMiss);
       m_worldMin(0) = shared._worldMin(0);
       m_worldMin(1) = shared._worldMin(1);
       m_worldMin(2) = shared._worldMin(2);
       #pragma STDC FENV_ACCESS ON
       std:: fesetround(FE_DOWNWARD);
       m_voxelWorldMin(0) = std::nearbyint(shared._worldMin(0)/m_gridRes);
       m_voxelWorldMin(1) = std::nearbyint(shared._worldMin(1)/m_gridRes);
       m_voxelWorldMin(2) = std::nearbyint(shared._worldMin(2)/m_gridRes);
       m_worldMax(0) = shared._worldMax(0);
       m_worldMax(1) = shared._worldMax(1);
       m_worldMax(2) = shared._worldMax(2);
       m_voxelWorldMax(0) = std::nearbyint(shared._worldMax(0)/m_gridRes);
       m_voxelWorldMax(1) = std::nearbyint(shared._worldMax(1)/m_gridRes);
       m_voxelWorldMax(2) = std::nearbyint(shared._worldMax(2)/m_gridRes);
       m_voxelWorldDim(0) = m_voxelWorldMax(0)-m_voxelWorldMin(0);
       m_voxelWorldDim(1) = m_voxelWorldMax(1)-m_voxelWorldMin(1);
       m_voxelWorldDim(2) = m_voxelWorldMax(2)-m_voxelWorldMin(2);
       m_transformedNorms = shared._transformedNorms;
       m_vRes = shared._vRes;
       m_hRes = shared._hRes;
       m_minRange = shared._minRange;
       m_maxRange = shared._maxRange;
       m_minVoxelRange = shared._minVoxelRange;
       m_maxVoxelRange = shared._maxVoxelRange;
       m_type = shared._type;

     }

    __hostdev__ void getRay(int w, int h, Vec3T& rotatedRay, Vec3T& rayEye, Vec3T& rayDir, ValueT& minTime, ValueT& maxTime, const ValueT& gridRes) const
    {
      if(m_type == 0){
        float u = (float)w*m_scale[0];
        float v = (float)h*m_scale[1];
        EigenVec matrix_dir = u * m_transformedViewVecs.col(0) + v * m_transformedViewVecs.col(1) - m_transformedViewVecs.col(2);
        EigenVec default_dir = u * m_identityVecs.col(0) + v * m_identityVecs.col(1) - m_identityVecs.col(2);
        ValueT norm = matrix_dir.norm();
        default_dir.normalize();
        matrix_dir.normalize();
        minTime = (m_minRange/gridRes)*norm;
        maxTime = (m_maxRange/gridRes)*norm;
        rayEye = Vec3T(m_position(0)/gridRes,m_position(1)/gridRes,m_position(2)/gridRes);
        rayDir =  Vec3T(matrix_dir(0),matrix_dir(1),matrix_dir(2));
        rotatedRay = Vec3T(default_dir(0), default_dir(1), default_dir(2));
      }else if(m_type==1){
        float u = -m_hFOV/2+((float)w)*m_scale[0];
        float v = -m_vFOV/2+((float)h)*m_scale[1];
        u *= 3.14159265358979323846 / 180;
        v *= 3.14159265358979323846 / 180;
        //Calculate unit direction of ray in xyz frame
        EigenVec matrix_dir(std::sin(u)*std::cos(v), std::sin(v),std::cos(u)*std::cos(v));
        //normalise ray
        matrix_dir.normalize();
        EigenVec  default_dir = matrix_dir;
        //rotate dir matrix to match sensor
        matrix_dir = EigenVec(m_rotation(0,0)*matrix_dir(0)+m_rotation(0,1)*matrix_dir(1)+m_rotation(0,2)*matrix_dir(2),
                   m_rotation(1,0)*matrix_dir(0)+m_rotation(1,1)*matrix_dir(1)+m_rotation(1,2)*matrix_dir(2),
                   m_rotation(2,0)*matrix_dir(0)+m_rotation(2,1)*matrix_dir(1)+m_rotation(2,2)*matrix_dir(2));
        matrix_dir.normalize();
        default_dir.normalize();
        minTime = m_minRange/gridRes;
        maxTime = m_maxRange/gridRes;
        rayEye = Vec3T(m_position(0)/gridRes,m_position(1)/gridRes,m_position(2)/gridRes);
        rayDir =  Vec3T(matrix_dir(0),matrix_dir(1),matrix_dir(2));
        rotatedRay = Vec3T(default_dir(0), default_dir(1), default_dir(2));
      }
    }

    __hostdev__ const int type() const{return m_type;}
    __hostdev__ const EigenMat rotation() const {return m_rotation;}
    __hostdev__ const EigenVec leafOriginOffset() const {return m_leafOriginOffset;}
    __hostdev__ const EigenVec voxelOriginOffset() const {return m_voxelOriginOffset;}
    __hostdev__ const EigenVec position() const {return m_position;}
    __hostdev__ const EigenVec voxelPosition() const {return m_voxelPosition;}
    __hostdev__ const Eigen::Matrix<int, 3, 1> leafOffset() const {return m_leafOffset;}
    __hostdev__ const EigenVec worldMin() const {return m_worldMin;}
    __hostdev__ const EigenVec worldMax() const {return m_worldMax;}
    __hostdev__ const Eigen::Matrix<int, 3, 1> voxelWorldMin() const {return m_voxelWorldMin;}
    __hostdev__ const Eigen::Matrix<int, 3, 1> voxelWorldMax() const {return m_voxelWorldMax;}
    __hostdev__ const Eigen::Matrix<int, 3, 1> voxelWorldDim() const {return m_voxelWorldDim;}

    __hostdev__ const ValueT gridRes() const {return m_gridRes;}

    __hostdev__ const ValueT maxRange() const {return m_maxRange;}
    __hostdev__ const ValueT minRange() const {return m_minRange;}
    __hostdev__ const ValueT maxVoxelRange() const {return m_maxVoxelRange;}
    __hostdev__ const ValueT minVoxelRange() const {return m_minVoxelRange;}

    __hostdev__ const int vRes() const { return m_vRes;}
    __hostdev__ const int hRes() const { return m_hRes;}

    __hostdev__ const ValueT clogOddsMiss() const {return m_logOddsMiss;}
    __hostdev__ const ValueT clogOddsHit() const {return m_logOddsHit;}
    __hostdev__ ValueT logOddsMiss() {return m_logOddsMiss;}
    __hostdev__ ValueT logOddsHit() {return m_logOddsHit;}
  protected:

    ValueT m_gridRes;
    int m_type;
    EigenMat m_rotation;
    EigenVec m_worldMin;
    Eigen::Matrix<int, 3, 1> m_voxelWorldMin;
    EigenVec m_worldMax;
    Eigen::Matrix<int,3,1> m_voxelWorldMax;
    Eigen::Matrix<int,3,1> m_voxelWorldDim;
    EigenVec m_position;
    EigenVec m_voxelPosition;
    Eigen::Matrix<int, 3, 1> m_leafOffset;
    EigenVec m_voxelOriginOffset;
    EigenVec m_leafOriginOffset;
    ValueT m_maxRange;
    ValueT m_minRange;
    ValueT m_maxVoxelRange;
    ValueT m_minVoxelRange;
    ValueT m_logOddsHit;
    ValueT m_logOddsMiss;
    int m_vRes;
    int m_hRes;
    float m_vFOV;
    float m_hFOV;
    float m_scale[2];
    EigenMat m_defaultViewVecs;
    EigenMat m_transformedViewVecs;
    EigenMat m_identityVecs;
    Eigen::Matrix<ValueT, 3, 5> m_transformedNorms;
  }; // sensor
} //namespace gpu
} // namespace nanomap

#endif

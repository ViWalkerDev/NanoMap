// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file nanomap.h
///
/// @author Violet Walker
///
#ifndef NANOMAP_H_INCLUDED
#define NANOMAP_H_INCLUDED
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#define M_PI 3.14159265358979323846
#define MAX_INT 32766
#define VOXEL_SIZE 1

namespace nanomap{
//Basic pose structure for managing agent Poses
  struct Pose
  {
      Eigen::Vector3f position;
      Eigen::Quaternionf orientation;
      Pose(){
        position = Eigen::Vector3f(0.0f,0.0f,0.0f);
        orientation = Eigen::Quaternionf(0.0f,0.0f,0.0f,1.0f);
      }
  };

}




#endif

// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file Manager.h
///
/// @author Violet Walker
///
#ifndef NANOMAP_MANAGER_MANAGER_H_INCLUDED
#define NANOMAP_MANAGER_MANAGER_H_INCLUDED
#include <iomanip>
#include <chrono>
#include <iostream>
#include <string>

#include "nanomap/sensor/SensorData.h"
#include "nanomap/map/OccupancyMap.h"
#include "nanomap/handler/Handler.h"
#include "nanomap/nanomap.h"
#include "nanomap/config/Config.h"

/******************************************************************************

*******************************************************************************/

namespace nanomap{
  namespace manager{




  using Map = nanomap::map::Map;

  using Handler = nanomap::handler::Handler;
  using ValueT  = float;
  using Vec3T   = nanovdb::Vec3<ValueT>;

  using EigenVec = Eigen::Matrix<ValueT, 3, 1>;
  using EigenMat = Eigen::Matrix<ValueT, 3, 3>;
  using Quaternion = Eigen::Quaternion<ValueT>;
  using SensorData = nanomap::sensor::SensorData;


  class Manager{

    public:
      Manager(std::shared_ptr<nanomap::config::Config> config)
      :_config(config)
      {
        _handler = std::make_unique<nanomap::handler::Handler>(_config);
      }

      void insertPointCloud(std::string sensorName, int pcl_width, int pcl_height, int pcl_step, unsigned char* cloud,
                                        const nanomap::Pose& pose, std::shared_ptr<nanomap::map::Map> map){

        for(auto itr = _config->sensorData().begin(); itr != _config->sensorData().end(); itr++){
          if((*itr)->sensorName().compare(sensorName)==0){
            int index = std::distance(_config->sensorData().begin(), itr);
            (*itr)->updatePose(pose);
            (*itr)->rotateView();
            _handler->processPointCloud(index, pcl_width, pcl_height, pcl_step, cloud, map);
          }
        }
      }

      void printUpdateTime(int count){
        _handler->printUpdateTime(count);
      }

      void closeHandler(){
        _handler->closeHandler();
        _handler.reset();
      }
      std::shared_ptr<nanomap::config::Config> config(){return _config;}





    protected:

      std::unique_ptr<nanomap::handler::Handler> _handler;

      std::shared_ptr<nanomap::config::Config> _config;

    };
  }
}
#endif

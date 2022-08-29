// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file SimManager.h
///
/// @author Violet Walker
///
#ifndef NANOMAP_MANAGER_SIMMANAGER_H_INCLUDED
#define NANOMAP_MANAGER_SIMMANAGER_H_INCLUDED
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <string>
#include <nanovdb/util/OpenToNanoVDB.h> // converter from OpenVDB to NanoVDB (includes NanoVDB.h and GridManager.h)
#include <nanovdb/util/IO.h>
#include <nanovdb/util/CudaDeviceBuffer.h>

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/tools/RayIntersector.h>
#include <openvdb/tools/Dense.h>

#include "nanomap/handler/SimHandler.h"
#include "nanomap/agent/Agent.h"
#include "nanomap/sensor/FrustumData.h"
#include "nanomap/sensor/SensorData.h"
#include "nanomap/map/OccupancyMap.h"
#include "nanomap/nanomap.h"
#include "nanomap/config/Config.h"

namespace nanomap{
    namespace manager{

  using Map = nanomap::map::Map;
  using Agent = nanomap::agent::Agent;
  using Map = nanomap::map::Map;
  using FloatGrid = openvdb::FloatGrid;

  using GridType = openvdb::FloatGrid;
  using TreeT = GridType::TreeType;
  using RootType = TreeT::RootNodeType;   // level 3 RootNode
  using Int1Type = RootType::ChildNodeType;  // level 2 InternalNode
  using Int2Type = Int1Type::ChildNodeType;  // level 1 InternalNode
  using LeafType = TreeT::LeafNodeType;   // level 0 LeafNode
  using IterType = TreeT::ValueOnIter;
  using LeafT = TreeT::LeafNodeType;
  using AccessorT = openvdb::tree::ValueAccessor<TreeT>;
  using VecTreeT = openvdb::Vec3DGrid::TreeType;
  using VecAccessorT = openvdb::tree::ValueAccessor<VecTreeT>;
  using BufferT = nanovdb::CudaDeviceBuffer;
  using ValueT  = float;
  using Vec3T   = nanovdb::Vec3<ValueT>;
  using RayT = nanovdb::Ray<ValueT>;
  using HDDA     = nanovdb::HDDA<RayT>;
  using IterType = TreeT::ValueOnIter;
  using EigenVec = Eigen::Matrix<ValueT, 3, 1>;
  using EigenMat = Eigen::Matrix<ValueT, 3, 3>;
  using Quaternion = Eigen::Quaternion<ValueT>;
  using SensorData = nanomap::sensor::SensorData;

  class SimManager{

    public:
      SimManager(std::shared_ptr<nanomap::config::Config> config)
      :_config(config){

        loadSimGrid();
        _config->loadAgents(_simGrid);
        for(int i = 0; i<_config->agentData().size(); i++){
          if(_config->agentData(i)->spawnRandom()){
            _config->agentData(i)->updatePose(getRandomSpawn());
          }
        }
      _handler = std::make_unique<nanomap::handler::SimHandler>(_config, _simGrid);
      std::cout << "_simHandler instantiated" <<  std::endl;
    }

      //Given a global pose, set agent pose to new pose.
      void updateAgentPoses(std::vector<Pose> poses){
        int index = 0;
        for(auto itr = poses.begin(); itr != poses.end(); itr++){
          index = std::distance(poses.begin(), itr);
          _config->agentData(index)->updatePose(*itr);
        }
      }

      std::vector<Pose> agentPoses(){
          std::vector<Pose> poses;
          for(auto itr = _config->agentData().begin(); itr != _config->agentData().end(); itr++){
            poses.push_back((*itr)->pose());
          }
          return poses;
      }
      //For each agent, for each sensor, upload relevant info to GPU.
      //Then generate and process simulated point clouds into map.
      void updateAgentViews(){
        //For each agent;
        for(int i = 0; i<_config->agentData().size(); i++){
          for(int j = 0; j<(_config->agentData(i)->sensorNames().size()); j++){
            for(int k  = 0; k < _config->sensorData().size() ; k++){
              if(_config->sensorData(k)->sensorName().compare(_config->agentData(i)->sensorNames()[j])==0){
                  _config->sensorData(k)->updatePose(_config->agentData(i)->sensorPose(j));
                  _config->sensorData(k)->rotateView();
                  _handler->processPointCloud(k, _config->agentData(i)->map());
              }
            }
          }
        }
      }
      void updateAgentViews(std::shared_ptr<nanomap::map::Map> map_ptr){
        //For each agent;
        for(int i = 0; i<_config->agentData().size(); i++){
          for(int j = 0; j<(_config->agentData(i)->sensorNames().size()); j++){
            for(int k  = 0; k < _config->sensorData().size() ; k++){
              if(_config->sensorData(k)->sensorName().compare(_config->agentData(i)->sensorNames()[j])==0){
                  _config->sensorData(k)->updatePose(_config->agentData(i)->sensorPose(j));
                  _config->sensorData(k)->rotateView();
                  _handler->processPointCloud(k, map_ptr);
              }
            }
          }
        }
      }

      void closeHandler(){
        _handler->closeHandler();
      }

      std::shared_ptr<nanomap::map::Map> getMap(int index){
        return _config->agentData(index)->map();
      }

      void step(){};

      void updateObservations(){};

      bool getAgentSimCollision(int index, Eigen::Vector3f positionDelta){
          Eigen::Vector3f newEye = _config->agentData(index)->pose().position + positionDelta;
          openvdb::Vec3d vecEye(newEye(0)/_config->gridRes(), newEye(1)/_config->gridRes(), newEye(2)/_config->gridRes());
          openvdb::Vec3d vecDir;

          double t0=0;
          double t1=0;
          openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid> intersector(*_simGrid);
          for(int i=0; i<_config->agentData(index)->observationRays().cols(); i++){
            vecDir = openvdb::Vec3f(_config->agentData(index)->observationRays().col(i)(0),
                                    _config->agentData(index)->observationRays().col(i)(1),
                                    _config->agentData(index)->observationRays().col(i)(2));
            openvdb::math::Ray<double> simRay(vecEye, vecDir);
            t0 = 0;
            t1 = 0;
            if(intersector.setIndexRay(simRay)){
              if(intersector.march(t0,t1)){
                if(t1<_config->agentData(index)->observationNorms().col(i)(0)){
                  std::cout << "time: " << t1<< std::endl;
                  std::cout << _config->agentData(index)->observationNorms().col(i)(0) << std::endl;
                  return false;
                }
              }

            }
          }
          return true;
      }


      void loadSimGrid(){
                std::cout <<"reading in map config file: " << _config->config() << std::endl;
                std::ifstream *input = new std::ifstream(_config->config().c_str(), std::ios::in | std::ios::binary);
                bool end = false;
                std::string file, line;
                *input >> line;
                float res;
                if(line.compare("#config") != 0){
                  std::cout << "Error: first line reads [" << line << "] instead of [#config]" << std::endl;
                  delete input;
                  return;
                }
                while(input->good()) {
                    *input >> line;
                    if (line.compare("SimGrid:") == 0){
                      *input >> file;
                    _simGrid = loadGridFromFile(file);
                    }else if (line.compare("#endconfig")==0){
                      break;
                    }
                  }
                input->close();
                }


      openvdb::FloatGrid::Ptr loadGridFromFile(std::string fileIn){
        openvdb::io::File file(fileIn);
        // Open the file.  This reads the file header, but not any grids.
        file.open();
        openvdb::GridBase::Ptr baseGrid;
        for (openvdb::io::File::NameIterator nameIter = file.beginName();
        nameIter != file.endName(); ++nameIter)
        {
          // Read in only the grid we are interested in.
          if (nameIter.gridName() == "grid") {
            baseGrid = file.readGrid(nameIter.gridName());
          } else {
            std::cout << "skipping grid " << nameIter.gridName() << std::endl;
          }
        }
        file.close();
        return openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
      }
      Pose getRandomSpawn(){
        Pose pose;
        Eigen::Matrix<float,3,3> rotation;
        //Right
        rotation.col(0)=Eigen::Matrix<float, 3, 1>(0.0,-1.0,0.0);
        //Forward
        rotation.col(1)=Eigen::Matrix<float, 3, 1>(1.0,0.0,0.0);
        //Up
        rotation.col(2)=Eigen::Matrix<float, 3, 1>(0.0,0.0,1.0);
        pose.orientation = Eigen::Quaternionf(rotation);

        auto bbox = _simGrid->evalActiveVoxelBoundingBox();
        openvdb::Coord min = bbox.min();
        openvdb::Coord max = bbox.max();
        openvdb::Vec3f center = bbox.getCenter();
        int x_range = max.x()-min.x();
        int y_range = max.y()-min.y();
        int z_range = max.z()-min.z();
        srand(time(0));
        auto acc = _simGrid->getAccessor();
        int x = rand()%x_range+min.x();
        int y = rand()%y_range+min.y();
        int z = rand()%z_range+min.z();
        openvdb::Coord coord_spawn(x,y,z);
        openvdb::Coord coord_value;
        bool close_to_wall;
        float wall_distance;
        float length = 10;
        bool finished = false;
        while(!finished){
          x = (int)(std::floor(rand()%x_range+min.x()));
          x = (int)(std::floor(rand()%y_range+min.y()));
          x = (int)(std::floor(rand()%z_range+min.z()));
          coord_spawn = openvdb::Coord(x,y,z);

          TreeT::NodeIter iter = _simGrid->tree().beginNode();
          iter.setMaxDepth(TreeT::NodeIter::LEAF_DEPTH-1);
          for (iter; iter; ++iter){
            switch (iter.getDepth()){
              case 1: {
                Int1Type* node = nullptr;
                iter.getNode(node);
                if (node){
                }
                break;
              }
              case 2: {
                Int2Type* node = nullptr;
                iter.getNode(node);
                if (node && !finished){
                  if(acc.getValue(iter.getCoord())<=-0.3){
                    coord_value = iter.getCoord()+openvdb::Coord(4,4,4);
                    length = (coord_value-coord_spawn).asVec3s().length();
                    if(length<=(5/_config->gridRes())){
                      finished = true;
                      break;
                    }
                  }
                }
              break;
              }
              case 3: {
              LeafType* node = nullptr;
              iter.getNode(node);
              if (node){

              }
              break;
              }
            }
          }
        }
        pose.position = Eigen::Matrix<float, 3, 1>(coord_spawn.asVec3d().x()*_config->gridRes(),
                                                   coord_spawn.asVec3d().y()*_config->gridRes(),
                                                   coord_spawn.asVec3d().z()*_config->gridRes());
        return pose;
      }


    protected:

      std::unique_ptr<nanomap::handler::SimHandler> _handler;

      std::shared_ptr<nanomap::config::Config> _config;

      openvdb::FloatGrid::Ptr _simGrid;
    };
    }
}
#endif

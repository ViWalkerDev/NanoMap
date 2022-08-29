// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file Config.h
///
/// @author Violet Walker
///

#ifndef NANOMAP_CONFIG_CONFIG_H_INCLUDED
#define NANOMAP_CONFIG_CONFIG_H_INCLUDED
#include <iomanip>
#include <chrono>
#include <iostream>
#include <string>
#include <fstream>

#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "nanomap/nanomap.h"
#include "nanomap/sensor/SensorData.h"
#include "nanomap/sensor/FrustumData.h"
#include "nanomap/sensor/LaserData.h"
#include "nanomap/agent/Agent.h"



#define M_PI 3.14159265358979323846
#define MAX_INT 32766
#define VOXEL_SIZE 1

namespace nanomap{
  namespace config{
//Config class. This contains all relevant user defined information for operation.
//Config information is loaded using config text files.
    class Config
  {
      public:
        //Empty Constructor
        Config(){}
        //Constructor that takes main config file as input
        Config(std::string configFile);
        void init();




  void loadConfig();
  void loadAgents(openvdb::FloatGrid::Ptr& simGrid);
  void loadAgentData(openvdb::FloatGrid::Ptr& simGrid);
  void loadSensors();
  std::shared_ptr<nanomap::sensor::SensorData> loadSensorData(float gridRes, int nodeEdge, std::string config);


  /******************************************************************************/
  //Fetching Sensor and Agent Data;
        //return full sensor data vector
          std::vector<std::shared_ptr<nanomap::sensor::SensorData>>& sensorData();
          std::vector<std::shared_ptr<nanomap::agent::Agent>>& agentData();
          //return sensor data object by vector index or sensorId.
          std::shared_ptr<nanomap::sensor::SensorData> sensorData(int index, bool searchById=false);
          std::shared_ptr<nanomap::agent::Agent> agentData(int index, bool searchById=false);

          std::shared_ptr<nanomap::sensor::SensorData> sensorData(std::string name);
          std::shared_ptr<nanomap::agent::Agent> agentData(std::string name);
    /******************************************************************************/
    //Fetch Mode variables
          int& filterType(){return _filterType;}
          int& updateType(){return _updateType;}
          int& exploreType(){return _exploreType;}
          int& processType(){return _processType;}
          int& precisionType(){return _precisionType;}
          int& simType(){return _simType;}
          int& publishSensor(){return _publishSensor;}
          float& gridRes(){return _gridRes;}
          float& mappingRes(){return _mappingRes;}
          int& leafEdge(){return _leafEdge;}
          int& leafVolume(){return _leafVolume;}
          float& frustumAllocationFactor(){return _frustumAllocationFactor;}
          float& laserAllocationFactor(){return _laserAllocationFactor;}
          int& maxPCLSize(){return _maxPCLSize;}
          int& laserPCLSize(){return _laserPCLSize;}
          int& laserVoxelBufferSize(){return _laserVoxelBufferSize;}
          int& frustumPCLSize(){return _frustumPCLSize;}
          int& frustumLeafBufferSize(){return _frustumLeafBufferSize;}
          bool& serialUpdate(){return _serialUpdate;}
          float& probHitThres(){return _probHitThres;}
          float& probMissThres(){return _probMissThres;}
    /****************************************************************************/
    //Fetch _config
          std::string& config(){return _config;}

  /******************************************************************************/


  /****************************************************************************/

      private:
          bool _configCheck=false;
          bool _serialUpdate=false;
          std::string _config;
          //Variables that define mode of operation.
          float _gridRes;
          float _mappingRes;
          int _leafEdge;
          int _leafVolume;

          int _filterType;
          int _exploreType;
          int _updateType;
          int _processType;
          int _simType;
          int _precisionType;
          int _publishSensor;

          float _probHitThres;
          float _probMissThres;


          //
          int _maxPCLSize;
          int _frustumPCLSize;
          float _frustumAllocationFactor;
          int _frustumLeafBufferSize;
          int _laserPCLSize;
          float _laserAllocationFactor;
          int _laserVoxelBufferSize;



          //Contains the sensors definitions used by agents
          std::vector<std::shared_ptr<nanomap::sensor::SensorData>> _sensorData;
          //Contains the agent information
          std::vector<std::shared_ptr<nanomap::agent::Agent>> _agentData;
        };
  }
}




#endif

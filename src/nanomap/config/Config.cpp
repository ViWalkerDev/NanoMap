// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file Config.cpp
///
/// @author Violet Walker
///

#include "nanomap/config/Config.h"

#define M_PI 3.14159265358979323846
#define MAX_INT 32766
#define VOXEL_SIZE 1

namespace nanomap{
  namespace config{
    //Large Config Class Definitions
    Config::Config(std::string configFile)
    {
      _config = configFile;
      _configCheck = true;
      init();
    }
    void Config::init()
    {
      if(_configCheck){
        loadConfig();
        loadSensors();
        _leafVolume = _leafEdge * _leafEdge * _leafEdge;
        _frustumLeafBufferSize = 0;
        _frustumPCLSize = 0;
        _laserPCLSize = 0;
        //_laserVoxelBufferSize = 0;
        _maxPCLSize = 0;
        for(int i = 0; i < _sensorData.size(); i++){
          if(_sensorData[i]->sharedParameters()._pclSize>_maxPCLSize){
            _maxPCLSize = _sensorData[i]->sharedParameters()._pclSize;
          }
          if(_sensorData[i]->sharedParameters()._type==0){
            if(_sensorData[i]->sharedParameters()._pclSize > _frustumPCLSize){
              _frustumPCLSize = _sensorData[i]->sharedParameters()._pclSize;
            }
            if(_sensorData[i]->sharedParameters()._maxLeafBufferSize > _frustumLeafBufferSize){
              _frustumLeafBufferSize = _sensorData[i]->sharedParameters()._maxLeafBufferSize;
            }
          }else if(_sensorData[i]->sharedParameters()._type==1){
            if(_sensorData[i]->sharedParameters()._pclSize > _laserPCLSize){
              _laserPCLSize = _sensorData[i]->sharedParameters()._pclSize;

            }
          }


        }
      }
    }
    void Config::loadConfig(){
      std::cout <<"reading in config file: " << _config << std::endl;
      std::ifstream *input = new std::ifstream(_config.c_str(), std::ios::in | std::ios::binary);
      //bool end = false;
      std::string file, line;
      float gridRes, mappingRes, frustumAllocationFactor, laserAllocationFactor, probHitThres, probMissThres;
      int leafEdge, filterType, processType, updateType, exploreType, precisionType, simType, publishSensor;
      int serialUpdate;
      *input >> line;
      //std::cout << line << std::endl;
      float res;
      if(line.compare("#config") != 0){
        std::cout << "Error: first line reads [" << line << "] instead of [#config]" << std::endl;
        delete input;
        return;
      }
      while(input->good()) {
          *input >> line;
          if (line.compare("SerialUpdate:") == 0){
            *input >> serialUpdate;
            if(serialUpdate){
              _serialUpdate = true;
            }else{
              _serialUpdate = false;
            }
          }else if (line.compare("MappingRes:") == 0){
            *input >> mappingRes;
            _mappingRes = mappingRes;
          }else if (line.compare("ProbHitThres:") == 0){
            *input >> probHitThres;
            _probHitThres = probHitThres;
          }else if (line.compare("ProbMissThres:") == 0){
            *input >> probMissThres;
            _probMissThres = probMissThres;
          }else if (line.compare("GridRes:") == 0){
            *input >> gridRes;
            _gridRes = gridRes;
          }else if (line.compare("NodeEdge:") == 0){
            *input >> leafEdge;
            _leafEdge = leafEdge;
          }else if (line.compare("FrustumAllocationFactor:") == 0){
            *input >> frustumAllocationFactor;
            _frustumAllocationFactor = frustumAllocationFactor;
          }else if (line.compare("LaserAllocationFactor:") == 0){
            *input >> laserAllocationFactor;
            _laserAllocationFactor = laserAllocationFactor;
          }else if (line.compare("FilterType:") == 0){
            *input >> filterType;
            _filterType = filterType;
          }else if (line.compare("ProcessType:") == 0){
            *input >> processType;
            _processType = processType;
          }else if(line.compare("UpdateType:")==0){
            *input >> updateType;
            _updateType = updateType;
          }else if (line.compare("ExploreType:") == 0){
            *input >> exploreType;
            _exploreType = exploreType;
          }else if (line.compare("PrecisionType:") == 0){
            *input >> precisionType;
            _precisionType = precisionType;
          }else if (line.compare("SimType:") == 0){
            *input >> simType;
            _simType = simType;
          }else if (line.compare("SimType:") == 0){
            *input >> publishSensor;
            _publishSensor = publishSensor;
          }else if (line.compare("#endconfig")==0){
            break;
          }
        }
      input->close();
      }

    void Config::loadAgents(openvdb::FloatGrid::Ptr& simGrid){
      loadAgentData(simGrid);
    }
    /******************************************************************************/
    void Config::loadAgentData(openvdb::FloatGrid::Ptr& simGrid){
      std::string line;
      std::string agentFile;
      std::vector<std::string> agentFiles;
      std::ifstream *input = new std::ifstream(_config.c_str(), std::ios::in | std::ios::binary);
      *input >> line;
      //check if file is frustumconfig
      if(line.compare("#config") == 0){
        while(input->good()) {
             *input >> line;
             if (line.compare("Agent:") == 0){
               *input >> agentFile;
               agentFiles.push_back(agentFile);
             }else if(line.compare("#endconfig")==0){
               break;
             }
        }
        input->close();
        for(int i=0; i<agentFiles.size(); i++){
           _agentData.push_back(std::make_shared<nanomap::agent::Agent>(nanomap::agent::Agent(simGrid, agentFiles[i], _mappingRes, _gridRes, _probHitThres, _probMissThres)));
        }
      }
    }



      void Config::loadSensors(){
        std::vector<std::string> sensorConfigs;
        std::string configFile, line;
        std::ifstream *input = new std::ifstream(_config.c_str(), std::ios::in | std::ios::binary);
        *input >> line;
        if(line.compare("#config") != 0){
          std::cout << "Error: first line reads [" << line << "] instead of [#config]" << std::endl;
          delete input;
          return;
        }
        while(input->good()){
          *input>>line;
          if(line.compare("Sensor:")==0){
            *input>>configFile;
            sensorConfigs.push_back(configFile);
          }else if(line.compare("#endconfig")==0){
            break;
          }
        }
        input->close();
        for(int i=0; i<sensorConfigs.size(); i++){
          _sensorData.push_back(
              loadSensorData(_mappingRes, _leafEdge, sensorConfigs[i]));
            }
        }


      std::shared_ptr<nanomap::sensor::SensorData> Config::loadSensorData(float gridRes,
                                                                           int leafEdge,
                                                                     std::string config)
      {
          std::cout <<"reading in sensor config file: " << config << std::endl;
          std::string line;
          std::string name;
          int id, rate, hRes, vRes;
          float aHRes, aVRes, vfov, hfov, maxRange, minRange, probHit, probMiss;
          Eigen::Matrix<float, 3, 3> frameTransform;
          std::ifstream *input = new std::ifstream(config.c_str(), std::ios::in | std::ios::binary);
          *input >> line;
          //check if file is frustumconfig
           if(line.compare("#laserconfig") == 0){
             while(input->good()) {
                 *input >> line;
                 if (line.compare("Name:") == 0){
                   *input >> name;
                 }else if(line.compare("Id:") == 0){
                   *input >> id;
                 }else if(line.compare("AngularHRes:") == 0){
                   *input >> aHRes;
                 }else if(line.compare("AngularVRes:") == 0){
                   *input >> aVRes;
                 }else if(line.compare("HFOV:") == 0){
                   *input >> hfov;
                 }else if(line.compare("VFOV:") == 0){
                   *input >> vfov;
                 }else if(line.compare("Rate:") == 0){
                   *input >> rate;
                 }else if(line.compare("MaxRange:") == 0){
                   *input >> maxRange;
                 }else if(line.compare("MinRange:") == 0){
                   *input >> minRange;
                 }else if(line.compare("ProbHit:") == 0){
                   *input >> probHit;
                 }else if(line.compare("ProbMiss:") == 0){
                   *input >> probMiss;
                 }else if(line.compare("FrameTransform:") == 0){
                   float x;
                   for(int i = 0; i <3 ; i++){
                     for(int j = 0; j<3; j++){
                       *input >> x;
                       frameTransform.col(i)(j) = x ;
                     }
                   }
                 }else if (line.compare("#endconfig")==0){
                             break;
                 }
               }
             input->close();
             return std::make_shared<nanomap::sensor::SensorData>(nanomap::sensor::LaserData(name, id, frameTransform, gridRes, leafEdge, aHRes, aVRes, hfov,
                                                     vfov, rate, maxRange, minRange,
                                                   probHit, probMiss));
          }else if(line.compare("#frustumconfig")==0){
            while(input->good()) {
                *input >> line;
                if (line.compare("Name:") == 0){
                  *input >> name;
                }else if(line.compare("Id:") == 0){
                  *input >> id;
                }else if(line.compare("HRes:") == 0){
                  *input >> hRes;
                }else if(line.compare("VRes:") == 0){
                  *input >> vRes;
                }else if(line.compare("VFOV:") == 0){
                  *input >> vfov;
                }else if(line.compare("Rate:") == 0){
                  *input >> rate;
                }else if(line.compare("MaxRange:") == 0){
                  *input >> maxRange;
                }else if(line.compare("MinRange:") == 0){
                  *input >> minRange;
                }else if(line.compare("ProbHit:") == 0){
                  *input >> probHit;
                }else if(line.compare("ProbMiss:") == 0){
                  *input >> probMiss;
                }else if(line.compare("FrameTransform:") == 0){
                  float x;
                  for(int i = 0; i <3 ; i++){
                    for(int j = 0; j<3; j++){
                      *input >> x;
                      frameTransform.col(i)(j) = x ;
                    }
                  }
                }else if (line.compare("#endconfig")==0){
                            break;
                }
              }
            input->close();
            return std::make_shared<nanomap::sensor::SensorData>(nanomap::sensor::FrustumData(name, id, frameTransform, gridRes, leafEdge, hRes, vRes,
                                                    vfov, rate, maxRange, minRange,
                                                  probHit, probMiss));
          }
          return nullptr;
        }

/******************************************************************************/
//Fetching Sensor and Agent Data;
    //return full sensor data vector
    std::vector<std::shared_ptr<nanomap::sensor::SensorData>>& Config::sensorData(){return _sensorData;}
    //return full agent data vector
    std::vector<std::shared_ptr<nanomap::agent::Agent>>& Config::agentData(){return _agentData;}
    //return sensor data object by vector index or sensorId.
    std::shared_ptr<nanomap::sensor::SensorData> Config::sensorData(int index, bool searchById){
      if(searchById){
        //Searching objects by id and not by vector index;
        for (auto it = _sensorData.begin(); it != _sensorData.end(); ++it) {
            if((*it)->getId() == index){
              return *it;
            }
        }
        //Object wasn't found.
        //RETURN EXCEPTON HERE.

      }else{
        if(index >=0 && index < _sensorData.size()){
          //index refers to valid object
          return _sensorData[index];
        }//else{
          //RETURN AN EXCEPTION HERE
        //}
      }
    }

    //return agent data object by vector index or sensorId.
    std::shared_ptr<nanomap::agent::Agent> Config::agentData(int index, bool searchById){
      if(searchById){
        //Searching objects by id and not by vector index;
        for (auto it = _agentData.begin(); it != _agentData.end(); ++it) {
            if((*it)->getId() == index){
              return *it;
            }
        }
        //Object wasn't found.
        //RETURN EXCEPTON HERE.

      }else{
        if(index >=0 && index < _agentData.size()){
          //index refers to valid object
          return _agentData[index];
        }//else{
          //RETURN AN EXCEPTION HERE
        //}
      }
    }
    //return sensor data from string
    std::shared_ptr<nanomap::sensor::SensorData> Config::sensorData(std::string name){
      //Searching objects by name;
      for (auto it = _sensorData.begin(); it != _sensorData.end(); ++it) {
          if((*it)->getName() == name){
            return *it;
          }
      }
      //Object wasn't found.
      //RETURN EXCEPTON HERE.
    }
    //return agent data from string
    std::shared_ptr<nanomap::agent::Agent> Config::agentData(std::string name){
      //Searching objects by name;
      for (auto it = _agentData.begin(); it != _agentData.end(); ++it) {
          if((*it)->getName() == name){
            return *it;
          }
      }
      //Object wasn't found.
      //RETURN EXCEPTON HERE.
    }
  }
}

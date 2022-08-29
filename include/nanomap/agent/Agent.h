// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file Agent.h
///
/// @author Violet Walker
///
#ifndef NANOMAP_AGENT_AGENT_H_INCLUDED
#define NANOMAP_AGENT_AGENT_H_INCLUDED
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <string>

#include "nanomap/map/OccupancyMap.h"
#include "nanomap/nanomap.h"


/******************************************************************************
This class defines a basic agent. It is used for single body UAVs with one or
more FIXED sensors and doesn't currently contain more advanced robotics kinematics like
Joint and Link definitions. update the UAV pose, then use the class to update the
sensor poses and pass a map file to the sensors to update the map with the new
sensor views. This also uses a map file to generate observation distances for
deeprl. Rays for an agent are generally loaded from a file and are precomputed.
however, a spherical observation sphere of radius r is generated if file not provided.
*******************************************************************************/

namespace nanomap{
  namespace agent{

  using Map = nanomap::map::Map;
  using FloatGrid = openvdb::FloatGrid;


  using ValueT  = float;
  using Vec3T   = nanovdb::Vec3<ValueT>;

  using EigenVec = Eigen::Matrix<ValueT, 3, 1>;
  using EigenMat = Eigen::Matrix<ValueT, 3, 3>;
  using Quaternion = Eigen::Quaternion<ValueT>;

  class Agent{

    public:
      Agent(openvdb::FloatGrid::Ptr& simGrid, std::string agentFile,
          float mappingRes, float gridRes, float probHitThres, float probMissThres)
          :_mappingRes(mappingRes)
          ,_gridRes(gridRes)
          ,_probHitThres(probHitThres)
          ,_probMissThres(probMissThres)
        {
          _map = std::make_shared<Map>(Map(_mappingRes, _probHitThres, _probMissThres));
          std::cout <<"reading in agent config file: " << agentFile << std::endl;
          bool isRandom = false;
          ValueT x,y,z,qx,qy,qz,qw,lenObs, numObs, collisionDist;
          std::string name, line;
          openvdb::Vec3f VecSpawn;
          Pose pose;
          std::ifstream *input = new std::ifstream(agentFile.c_str(), std::ios::in | std::ios::binary);
          bool end = false;
          *input >> line;
          if(line.compare("#agentconfig") != 0){
            std::cout << "Error: first line reads [" << line << "] instead of [#agentconfig]" << std::endl;
            delete input;
            return;
          }
          while(input->good()) {
            *input >> line;
            if(line.compare("Spawn:") == 0){
              *input >> line;
              *input >> line;
              if(line.compare("Random") == 0){
                _spawnRandom = true;
                //VecSpawn = nanomap::util::getAgentSpawn(simGrid, _gridRes);
                _pose.position = Eigen::Matrix<float, 3, 1>(0.0,0.0,0.0);
                Eigen::Matrix<float,3,3> rotation;
                //Right
                rotation.col(0)=Eigen::Matrix<float, 3, 1>(0.0,-1.0,0.0);
                //Forward
                rotation.col(1)=Eigen::Matrix<float, 3, 1>(1.0,0.0,0.0);
                //Up
                rotation.col(2)=Eigen::Matrix<float, 3, 1>(0.0,0.0,1.0);
                _pose.orientation = Eigen::Quaternionf(rotation);
              }else{
                _spawnRandom = false;
                *input>>line;
                *input>>x>>y>>z;
                *input>>line;
                *input>>qx>>qy>>qz>>qw;
                _pose.position = Eigen::Vector3f(x,y,z);
                Eigen::Matrix<float,3,3> rotation;
                //Right
                rotation.col(0)=Eigen::Matrix<float, 3, 1>(0.0,-1.0,0.0);
                //Forward
                rotation.col(1)=Eigen::Matrix<float, 3, 1>(1.0,0.0,0.0);
                //Up
                rotation.col(2)=Eigen::Matrix<float, 3, 1>(0.0,0.0,1.0);
                _pose.orientation = Eigen::Quaternionf(rotation);
              }
          }else if(line.compare("Observations:")==0){
              *input>>line;
              *input >> numObs;
              *input >> line;
              *input >> lenObs;
              *input >> line;
              *input >> collisionDist;
          }else if(line.compare("Sensor:")==0){
              *input>>line;
              *input>>name;
              *input>>line;
              *input>>x>>y>>z;
              *input>>line;
              *input>>qw>>qx>>qy>>qz;
              pose.position = Eigen::Vector3f(x,y,z);
              pose.orientation = Eigen::Quaternionf(qw,qx,qy,qz);
              _sensorNames.push_back(name);
              _sensorOrigins.push_back(pose);
          }else if (line.compare("#endconfig")==0){
            break;
          }
        }
        input->close();
        _sensorPoses = _sensorOrigins;
        _lenObs = lenObs/_mappingRes;
        _numObs = numObs;
        _collisionDist = (float)(collisionDist/_gridRes);
        _observationRays(3,_numObs);
        _observationNorms(2,_numObs);
        generateObservationSphere(_numObs, _collisionDist, _lenObs,
                                                     _observationRays, _observationNorms);
      }
      void updatePose(Pose pose){
        _pose = pose;
        int index = 0;
        Eigen::Matrix<float, 3, 3> defaultFrameTransform;
        for(auto itr = _sensorOrigins.begin(); itr != _sensorOrigins.end(); itr++){
          index = std::distance(_sensorOrigins.begin(), itr);
          _sensorPoses[index].position = _pose.position;
          _sensorPoses[index].orientation = _pose.orientation*((*itr).orientation);
        }
      }

      Eigen::Vector3f sphericalCoordinate(float x, float y){
        Eigen::Vector3f point;
        point(0)= std::cos(x)*std::cos(y);
        point(1)= std::sin(x)*std::cos(y);
        point(2)= std::sin(y);
        return point;
      }

      Eigen::Matrix<float,3,Eigen::Dynamic> normalisedSphere(int n, float x){
        Eigen::Matrix<float, 3, Eigen::Dynamic> sphere_points(3,n);
        float start = (-1.0+1.0/(n-1.0));
        float increment = (2.0-2.0/(n-1.0))/(n-1.0);
        float s,j,k;
        for(int i = 0; i < n; i++){
          s = start+i*increment;
          j = s*x;
          k = (M_PI/2.0)*std::copysign(1.0,s)*(1.0-std::sqrt(1.0-std::abs(s)));
          sphere_points.col(i) = sphericalCoordinate(j, k);
          sphere_points.col(i).normalize();
        }
        return sphere_points;
      }





          Eigen::Matrix<float, 3, Eigen::Dynamic> generateSphere(int n){
            //std::cout << "1" << std::endl;

            return normalisedSphere(n, (0.1+1.2*n));
          }




          void generateObservationSphere(int numObs, float collisionDist, float lenObs,
                                                Eigen::Matrix<float, 3, Eigen::Dynamic>& observationRays,
                                                Eigen::Matrix<float, 2, Eigen::Dynamic>& observationNorms){
            observationRays = generateSphere(numObs);
            Eigen::Matrix<float, 2, Eigen::Dynamic> norms(2,numObs);
            for(int x = 0; x < numObs; x++){
              norms.col(x) = Eigen::Matrix<float, 2, 1>(collisionDist,lenObs);
            }
            observationNorms = norms;
          }

      Pose pose(){return _pose;}

      std::vector<Pose> sensorOrigins(){return _sensorOrigins;}
      std::vector<Pose> sensorPoses(){return _sensorPoses;}
      void clearSensors(){_sensorPoses.clear();
                          _sensorNames.clear();
                          _sensorOrigins.clear();}

      void clearObservationRays(){ _observationNorms.resize(0,0);
                                    _observationRays.resize(0,0);}
      Eigen::Matrix<float, 3, Eigen::Dynamic> observationRays(){return _observationRays;}
      Eigen::Matrix<float, 2, Eigen::Dynamic> observationNorms(){return _observationNorms;}
      void resetAgent(){clearSensors();
                        clearObservationRays();}
      Pose sensorPose(int index){return _sensorPoses[index];}

      std::shared_ptr<Map> map(){return _map;}
      std::vector<std::string> sensorNames() {return _sensorNames;}
      int getId(){return _agentId;}
      std::string getName(){return _agentName;}
      bool spawnRandom(){return _spawnRandom;}
    protected:
      std::string _agentName;
      int _agentId;
      float _gridRes, _mappingRes, _probHitThres, _probMissThres;
      Pose _pose;
      int _numObs;
      float _lenObs;
      float _collisionDist;
      bool _spawnRandom;
      Eigen::Matrix<float, 3, Eigen::Dynamic> _observationRays;
      Eigen::Matrix<float, 2, Eigen::Dynamic> _observationNorms;
      Eigen::ArrayXd _observations;
      std::vector<std::string> _sensorNames;
      std::vector<Pose> _sensorOrigins;
      std::vector<Pose> _sensorPoses;
      std::shared_ptr<Map> _map;

    };
  }
}
#endif

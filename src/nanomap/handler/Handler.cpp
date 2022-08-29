// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file Handler.cpp
///
/// @author Violet Walker
///

#include "nanomap/handler/Handler.h"

// The following functions are called by the host and launch the gpu kernels.

//This kernel generates a point cloud using a simulated sensor view and an existing map
//extern "C" void generateCloud(nanomap::handler::SensorBucket& _sensorBucket, cudaStream_t s0);
namespace nanomap{
  namespace handler{
        Handler::Handler(std::shared_ptr<nanomap::config::Config> config)
          :_config(config)
          ,_sensorAllocator(config)
        {
          _gpuTime = 0.0;
          _mapUpdateTime = 0.0;
          cudaCheck(cudaStreamCreate(&_s0));
          cudaCheck(cudaStreamCreate(&_s1));
        }


              void Handler::populateTempGrid(openvdb::FloatGrid::Accessor& tempAcc, int sensorIndex, int pcl_width, int pcl_height, int pcl_step,
                                unsigned char* cloudPtr, std::shared_ptr<nanomap::map::Map> map){
                openvdb::math::Ray<double> ray;
                openvdb::math::DDA<openvdb::math::Ray<double> ,0> dda;
                nanomap::Pose pose = _config->sensorData(sensorIndex)->sharedParameters()._pose;
                openvdb::Vec3d ray_origin_world(pose.position(0), pose.position(1), pose.position(2));
                openvdb::Vec3d ray_origin_index(map->occupiedGrid()->worldToIndex(ray_origin_world));
                openvdb::Vec3d ray_direction;
                bool max_range_ray;
                openvdb::Vec3d x;
                double ray_length;
                float max_time = _config->sensorData(sensorIndex)->sharedParameters()._maxRange/map->gridRes();
                float pointx, pointy, pointz;
                Eigen::Matrix<float,3,3> rotation = pose.orientation.normalized().toRotationMatrix()*_config->sensorData(sensorIndex)->sharedParameters()._frameTransform;
                float prob_miss = _config->sensorData(sensorIndex)->sharedParameters()._probMiss;
                float prob_hit = _config->sensorData(sensorIndex)->sharedParameters()._probMiss;
                float logodds_miss = log(prob_miss)-log(1-prob_miss);
                float logodds_hit = log(prob_hit)-log(1-prob_hit);
                // Probability update lambda for empty grid elements
                auto miss = [&prob_miss = logodds_miss](float& voxel_value, bool& active) {
                  voxel_value += prob_miss;
                  active = true;
                };

                // Probability update lambda for occupied grid elements
                auto hit = [&prob_hit = logodds_hit](float& voxel_value, bool& active) {
                  voxel_value += prob_hit;
                  active = true;
                };
                // Raycasting of every point in the input cloud
                for (int i = 0; i < pcl_width*pcl_height; i++)
                {
                  unsigned char* byte_ptr = cloudPtr + i*pcl_step;
                  pointx = *(reinterpret_cast<float*>(byte_ptr+0));
                  pointy = *(reinterpret_cast<float*>(byte_ptr+4));
                  pointz = *(reinterpret_cast<float*>(byte_ptr+8));
                  max_range_ray = false;
                  ray_direction = openvdb::Vec3d(
              		  rotation(0,0)*pointx+rotation(0,1)*pointy+rotation(0,2)*pointz,
                                rotation(1,0)*pointx+rotation(1,1)*pointy+rotation(1,2)*pointz,
                                rotation(2,0)*pointx+rotation(2,1)*pointy+rotation(2,2)*pointz);

                  ray_length = ray_direction.length()/map->gridRes();
                  if(ray_length  < max_time){
                    ray_direction.normalize();
                    ray.setEye(ray_origin_index);
                    ray.setDir(ray_direction);
                    dda.init(ray,0,ray_length);
                    openvdb::Coord voxel = dda.voxel();
                    while(dda.step()){
                	     tempAcc.modifyValueAndActiveState(voxel, miss);
                       voxel = dda.voxel();
                    }
                    if(dda.time()<max_time){
                	     tempAcc.modifyValueAndActiveState(voxel, hit);
                    }
                  }
                }
              }

              void Handler::integrateTempGrid(openvdb::FloatGrid::Ptr tempGrid, std::shared_ptr<nanomap::map::Map> map){
                float tempValue;
                // Probability update lambda for occupied grid elements
                float occClampThres = map->occupiedClampingThreshold();
                float emptyClampThres = map->emptyClampingThreshold();
                float logodds_thres_max = map->logOddsHitThreshold();
                float logodds_thres_min = map->logOddsMissThreshold();
                auto update = [&prob_thres_max = logodds_thres_max, &prob_thres_min = logodds_thres_min,
              		             &occ_clamp = occClampThres, &empty_clamp = emptyClampThres, &temp_value = tempValue]
                               (float& voxel_value, bool& active) {
                  voxel_value += temp_value;
                  if (voxel_value > occ_clamp)
                  {
                    voxel_value = occ_clamp;
                  }else if(voxel_value < empty_clamp){
                    voxel_value = empty_clamp;
                  }

                  if(voxel_value > prob_thres_max){
                    active = true;
                  }else if(voxel_value < prob_thres_min){
                    active = false;
                  }
                };
                // Integrating the data of the temporary grid into the map using the probability update functions
                for (openvdb::FloatGrid::ValueOnCIter iter = tempGrid->cbeginValueOn(); iter; ++iter)
                {
                  tempValue = iter.getValue();
                  if (tempValue!=0.0)
                  {
                    map->occAccessor()->modifyValueAndActiveState(iter.getCoord(), update);
                  }
                }
              return;
              }

                void Handler::processPointCloudCPU(int index, int pcl_width, int pcl_height, int pcl_step, unsigned char* cloud,  std::shared_ptr<nanomap::map::Map> map){
                  openvdb::FloatGrid::Ptr tempGrid = openvdb::FloatGrid::create(0.0);
                  //tempGrid->setTransform(openvdb::math::Transform::createLinearTransform(_config->mappingRes()));
                  auto tempAcc = tempGrid->getAccessor();
                  populateTempGrid(tempAcc, index, pcl_width, pcl_height, pcl_step,
                                    cloud, map);

                  integrateTempGrid(tempGrid, map);
                }

        void Handler::voxelUpdateFromFrustumBuffer(int leafEdge){
          int leafVolume = leafEdge*leafEdge*leafEdge;
          float logOddsHit = _agentMap->logOddsHitThreshold();
          float logOddsMiss = _agentMap->logOddsMissThreshold();
          float occClamp = _agentMap->occupiedClampingThreshold();
          float emptyClamp =  _agentMap->emptyClampingThreshold();
          float probeValue;
          float value;
          float targetActive = false;
          auto update = [&log_hit_thres = logOddsHit,
                         &log_miss_thres = logOddsMiss,
                         &occ_clamp = occClamp,
                         &empty_clamp = emptyClamp,
                         &probe_value = probeValue,
                         &temp_value = value]
                         (float& voxel_value, bool& active) {
             voxel_value += temp_value;
             if (voxel_value > occ_clamp){
               voxel_value = occ_clamp;
             }else if(voxel_value < empty_clamp){
               voxel_value = empty_clamp;
             }
             if(voxel_value > log_hit_thres){
               active = true;
             }else if(voxel_value < log_miss_thres){
               active = false;
             }
          };
          int nodeIndex, voxelIndex, voxelBufferIndex;
          for(int i = 0; i < *(_sensorAllocator.sensorBucket()->hostFrustumLeafCount()); i++){
            int nodeX = *(_sensorAllocator.sensorBucket()->hostFrustumLeafBuffer()+ i*3);
            int nodeY = *(_sensorAllocator.sensorBucket()->hostFrustumLeafBuffer()+ i*3+1);
            int nodeZ = *(_sensorAllocator.sensorBucket()->hostFrustumLeafBuffer()+ i*3+2);
            for(int x = 0; x < leafEdge; x++){
              for(int y = 0; y < leafEdge ; y++){
                for(int z = 0; z < leafEdge; z++){
                  nodeIndex = i*leafVolume;
                  voxelIndex = z+y*leafEdge + x*leafEdge*leafEdge;
                  voxelBufferIndex = nodeIndex+voxelIndex;
                  value = (float)(*(_sensorAllocator.sensorBucket()->hostFrustumVoxelBuffer()+voxelBufferIndex))/100;
                  if(value != 0.0){
                    _agentMap->occAccessor()->modifyValueAndActiveState(openvdb::Coord(nodeX+x,nodeY+y,nodeZ+z), update);
                  }
                }
              }
            }
          }
        }



         void Handler::blockUpdateFromFrustumBuffer(){
            BlockWorker occBlockWorker(_sensorAllocator.sensorBucket()->hostGridData()->leafEdge(),
                                      _agentMap->occupiedClampingThreshold(),
                                      _agentMap->emptyClampingThreshold(),
                                      _agentMap->logOddsHitThreshold(),
                                      _agentMap->logOddsMissThreshold(),
                                  _agentMap->occAccessor(),
                                  _sensorAllocator.sensorBucket()->hostFrustumVoxelBuffer(),
                                  _sensorAllocator.sensorBucket()->hostFrustumLeafBuffer(),
                                  *(_sensorAllocator.sensorBucket()->hostFrustumLeafCount()));

            occBlockWorker.processBlocks(_config->serialUpdate());
            for(int i = 0; i < *(_sensorAllocator.sensorBucket()->hostFrustumLeafCount()); i++){
              BlockWorker::Block& occBlock = (*(occBlockWorker._blocks))[i];
              if (occBlock.leaf) {
                _agentMap->occAccessor()->addLeaf(occBlock.leaf);
              }
            }
            occBlockWorker.destroyBlocks();
          }
          void Handler::integrateTempGrid(openvdb::FloatGrid::Ptr tempGrid, openvdb::FloatGrid::Ptr Grid, openvdb::FloatGrid::Accessor& acc, float emptyClampThres, float occClampThres, float logodds_thres_min, float logodds_thres_max){
            float tempValue;
            // Probability update lambda for occupied grid elements
            auto update = [&prob_thres_max = logodds_thres_max, &prob_thres_min = logodds_thres_min,
              &occ_clamp = occClampThres, &empty_clamp = emptyClampThres, &temp_value = tempValue](float& voxel_value,
                                                                                          bool& active) {
              voxel_value += temp_value;
              if (voxel_value > occ_clamp)
              {
                voxel_value = occ_clamp;
              }else if(voxel_value < empty_clamp){
                voxel_value = empty_clamp;
              }

              if(voxel_value > prob_thres_max){
                active = true;
              }else if(voxel_value < prob_thres_min){
                active = false;
              }
            };
            // Integrating the data of the temporary grid into the map using the probability update functions
            for (openvdb::FloatGrid::ValueOnCIter iter = tempGrid->cbeginValueOn(); iter; ++iter)
            {
              tempValue = iter.getValue();
              if (tempValue!=0.0)
              {
                acc.modifyValueAndActiveState(iter.getCoord(), update);
              }
            }
            return;
          }
        void Handler::processPointCloud(int sensorIndex, int pcl_width, int pcl_height,
                                  int pcl_step, unsigned char* cloud,
                                    std::shared_ptr<nanomap::map::Map> agentMap){

          std::chrono::duration<double, std::milli> delay;

          auto gpu_start = std::chrono::high_resolution_clock::now();

          _agentMap = agentMap;
          auto start = std::chrono::high_resolution_clock::now();
          if(_config->sensorData(sensorIndex)->sharedParameters()._type==1){
            start = std::chrono::high_resolution_clock::now();
            processPointCloudCPU(sensorIndex, pcl_width, pcl_height, pcl_step, cloud, agentMap);
          }else if(_config->sensorData(sensorIndex)->sharedParameters()._type==0){
            _sensorAllocator.update(pcl_width, pcl_height, pcl_step, cloud, _config->sensorData(sensorIndex), _s0);

            cudaDeviceSynchronize();

            filterCloud(*(_sensorAllocator.sensorBucket()), _s0);

            cudaDeviceSynchronize();
            if(_sensorAllocator.sensorBucket()->hostSensor()->type()==0){
              frustumCastCloud(*(_sensorAllocator.sensorBucket()), _s0, _s1);
              cudaCheck(cudaMemcpyAsync(_sensorAllocator.sensorBucket()->hostFrustumLeafBuffer(), _sensorAllocator.sensorBucket()->devFrustumLeafBuffer(), (*_sensorAllocator.sensorBucket()->hostFrustumLeafCount())*3*sizeof(int), cudaMemcpyDeviceToHost, _s0));
              cudaCheck(cudaMemcpyAsync(_sensorAllocator.sensorBucket()->hostFrustumVoxelBuffer(), _sensorAllocator.sensorBucket()->devFrustumVoxelBuffer(), (*_sensorAllocator.sensorBucket()->hostFrustumLeafCount())*512*sizeof(int8_t), cudaMemcpyDeviceToHost,_s1));
            }
            cudaStreamSynchronize(_s0);
            cudaStreamSynchronize(_s1);
            auto gpu_end = std::chrono::high_resolution_clock::now();
            delay = gpu_end-gpu_start;
            _gpuTime+=delay.count();
            start = std::chrono::high_resolution_clock::now();

            if(_sensorAllocator.sensorBucket()->hostSensor()->type()==0){
              if(_config->updateType() == 0){
                voxelUpdateFromFrustumBuffer(_config->sensorData(sensorIndex)->sharedParameters()._leafEdge);
              }else if(_config->updateType() == 1){
                blockUpdateFromFrustumBuffer();
              }
            }
          }
           auto end = std::chrono::high_resolution_clock::now();
           delay = end-start;
           _mapUpdateTime+=delay.count();

        }
        void Handler::closeHandler(){
          cudaCheck(cudaStreamDestroy(_s0));
          cudaCheck(cudaStreamDestroy(_s1));
          }

        void Handler::printUpdateTime(int count){
            std::cout << "gpu update time per loop:" << _gpuTime / count << std::endl;
            std::cout << "map update time per loop:" << _mapUpdateTime / count << std::endl;
          }
  }
}

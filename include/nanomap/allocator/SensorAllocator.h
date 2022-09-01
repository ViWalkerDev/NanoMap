// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file SensorAllocator.h
///
/// @author Violet Walker
///
#ifndef NANOMAP_HANDLER_SENSORALLOCATOR_H_INCLUDED
#define NANOMAP_HANDLER_SENSORALLOCATOR_H_INCLUDED
#include "nanomap/handler/handlerAssert.h"
#include "cuda_fp16.h"
#include <nanovdb/NanoVDB.h>
#include "nanomap/gpu/NodeWorker.h"
#include "nanomap/gpu/PointCloud.h"
#include "nanomap/gpu/Sensor.h"
#include "nanomap/gpu/GridData.h"
#include "nanomap/gpu/SensorBucket.h"
#include "nanomap/sensor/SensorData.h"
#include "nanomap/config/Config.h"
#include "nanomap/nanomap.h"



namespace nanomap{
  namespace allocator{
    //using BufferT = nanovdb::CudaDeviceBuffer;
    using ValueT  = float;
    using EigenVec = Eigen::Matrix<ValueT, 3, 1>;
    class SensorAllocator{

      public:

      SensorAllocator(std::shared_ptr<nanomap::config::Config> config)
      :_sb(std::make_shared<nanomap::gpu::SensorBucket>(config->maxPCLSize(), config->frustumLeafBufferSize())),
      _cf(config)
      {
        _sb->setGridRes(_cf->gridRes());
        _sb->setMappingRes(_cf->mappingRes());
        _sb->setFilterType(_cf->filterType());
        _sb->setProcessType(_cf->processType());
        //If using node optimisation for frustum cameras
        if(_cf->processType()==1){ //|| _cf->processType() == 2){
          //Arrays used for sorting the active node information.
          cudaCheck(cudaMalloc((void**)&((*_sb)._activeLeafNodes), _cf->frustumLeafBufferSize()*sizeof(int)));
          //std::cout << "1" << std::endl;
          cudaCheck(cudaMalloc((void**)&((*_sb)._activeFlags), _cf->frustumLeafBufferSize()*sizeof(int)));
          //std::cout << "1" << std::endl;
          cudaCheck(cudaMalloc((void**)&((*_sb)._activeIndices), _cf->frustumLeafBufferSize()*sizeof(int)));
          //Set Voxel allocation. This can be changed as needed, and will resize as necessary,
          //but resizing is costly, so it is a good number to reduce memory consumption
          //While also preventing a lot of resizing when using a frustum style sensor.
          //Default factor of 0.4 means that container size for voxel representation set to 2/5th of max observable area for a given sensor.
          _frustumVoxelAllocation = (_cf->frustumLeafBufferSize()*(_cf->leafVolume())*_cf->frustumAllocationFactor());
          //Frustum voxel worker contains the probablistic occupancy result of raycasting on GPU
          cudaCheck(cudaMalloc((void**)&((*_sb)._devFrustumVoxelWorker), _frustumVoxelAllocation*sizeof(float)));
          //Frustum leaf buffer contains the coordinates of active leafNodes.
          cudaCheck(cudaMalloc((void**)&((*_sb)._devFrustumLeafBuffer), _cf->frustumLeafBufferSize()*3*sizeof(int)));
          cudaCheck(cudaMallocHost((void**)&((*_sb)._hostFrustumLeafBuffer), _cf->frustumLeafBufferSize()*3*sizeof(int)));
          //Frustum Voxel buffer is used to transfer contents of the frustum voxel worker to CPU, uses int8_t type to reduce the size of the container,
          //And improve memcpy times as this step happens each loop.
          cudaCheck(cudaMalloc((void**)&((*_sb)._devFrustumVoxelBuffer), _frustumVoxelAllocation*sizeof(int8_t)));
          cudaCheck(cudaMallocHost((void**)&((*_sb)._hostFrustumVoxelBuffer), _frustumVoxelAllocation*sizeof(int8_t)));
        }
        if(_cf->filterType() == 0){
          //Do Nothing
        }else if(_cf->filterType() == 1 && (_cf->processType() == 0 || _cf->processType() == 2)){
            //VOXEL FILTER SETTING
            //ONLY USEFUL FOR FRUSTUM CAMERA
            //Filtering increases runtime but also increases memory usage.
            //This is usually only a problem for sensors with long range, and/or sensors with high grid resolution.
            //If you have the memory, the speed ups are
            //Generally worth the cost.
            //Once the grid resolution becomes too fine to meaningfully recude the input sensor cloud
            //the performance benefit is lost, and the overhead causes slowdown.
            // Four precision modes are provided

            if(_cf->precisionType() == 0){
              //Precision == 0, provides full xyz precision for discretising rays to a voxel but costs the most memory
              _sb->setPrecisionType(_cf->precisionType());
              cudaCheck(cudaMalloc((void**)&((*_sb)._voxelFilterBuffer), _cf->leafVolume()*4*(_cf->frustumLeafBufferSize())*sizeof(float)));
            }else if(_cf->precisionType() == 1){
            //Precision == 1, provides (half2 type) xyz offset precision for discretising rays to a voxel. It costs half as much memory as precision 0
            //This is not supported on the Jetson Nano as the nano only supports Compute capability 5.4. This does work on the Xavier NX.
            //The jetson nano version of this code uses 16 bit integers to store an offset value that has been multiplied by 100.
            //This uses half the memory, which is good for a memory constrained platform, but loses some precision.
            //Uses 1/2 the memory of Precision  = 0;
              _sb->setPrecisionType(_cf->precisionType());
              cudaCheck(cudaMalloc((void**)&((*_sb)._voxelFilterBufferHalf2), _cf->leafVolume()*2*(_cf->frustumLeafBufferSize())*sizeof(__half2)));
            }else if(_cf->precisionType() == 2){
              //Precision 2 doesn't even bother tracking ray offsets within a voxel, if a ray ends in a voxel, a counter is incremented for that voxel,
              //The resultant ray is always directed at the center of the voxel in question.
              //This is slightly less accurate, but uses significantly less memory. Great if you don't mind the loss of precision.
              //For smaller voxel sizes the loss of precision becomes smaller.
              //Uses 1/8th of the memory of Precision  = 0;
              //Uses 1/4th the memory of Precision = 1;
              _sb->setPrecisionType(_cf->precisionType());
              cudaCheck(cudaMalloc((void**)&((*_sb)._voxelFilterBufferSimple), _cf->leafVolume()*(_cf->frustumLeafBufferSize())*sizeof(int)));
            }else if(_cf->precisionType() == 3){
              //Precision 3 tracks voxel count only, the same as Precision 2.
              //But it uses 1/4th the memory as Precision 2 by condensing 4 counts into a single 32 bit int,
              //for some resolutions this can cause bottlenecks for atomicAdd operations due to sharing single ints
              _sb->setPrecisionType(_cf->precisionType());
              cudaCheck(cudaMalloc((void**)&((*_sb)._voxelFilterBufferSimpleQuad), _cf->leafVolume()*(std::ceil(_cf->frustumLeafBufferSize()/4))*sizeof(int)));
            }
        }
        cudaDeviceSynchronize();
        _sb->hostGridData()->init(_cf->mappingRes(), _cf->leafEdge(), _cf->frustumLeafBufferSize());
      }

    void update(int pcl_width, int pcl_height, int pcl_step, unsigned char* cloud,
                std::shared_ptr<nanomap::sensor::SensorData> sensorData, cudaStream_t s0){
        if(sensorData->sharedParameters()._type == 0){
          _sb->hostSensor()->updateFrustum(sensorData->sharedParameters());
        }else if(sensorData->sharedParameters()._type == 1){
          _sb->hostSensor()->updateLaser(sensorData->sharedParameters());
        }
        _sb->hostGridData()->update(sensorData->sharedParameters()._voxelBounds,
                              sensorData->sharedParameters()._leafBounds,
                              sensorData->sharedParameters()._leafBufferSize);
        cudaCheck(cudaMemcpy(_sb->devSensor(), _sb->hostSensor(), sizeof(nanomap::gpu::Sensor<float>), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(_sb->devGridData(), _sb->hostGridData(), sizeof(nanomap::gpu::GridData), cudaMemcpyHostToDevice));
        *(_sb->hostFrustumLeafCount()) = 0;
        cudaCheck(cudaMemcpy(_sb->devFrustumLeafCount(), _sb->hostFrustumLeafCount(), sizeof(int), cudaMemcpyHostToDevice));
        *(_sb->hostRayCount()) = 0;
        cudaCheck(cudaMemcpy(_sb->devRayCount(), _sb->hostRayCount(), sizeof(int), cudaMemcpyHostToDevice));
        _sb->pclHandle().updatePointCloudHandle(pcl_height*pcl_width, pcl_step, cloud);
        _sb->pclHandle().deviceUpload(s0);
        cudaDeviceSynchronize();

    }

    void update(std::shared_ptr<nanomap::sensor::SensorData> sensorData, cudaStream_t s0){
      if(sensorData->sharedParameters()._type == 0){
        _sb->hostSensor()->updateFrustum(sensorData->sharedParameters());
      }else if(sensorData->sharedParameters()._type == 1){
        _sb->hostSensor()->updateLaser(sensorData->sharedParameters());
      }

      _sb->hostGridData()->update(sensorData->sharedParameters()._voxelBounds,
                            sensorData->sharedParameters()._leafBounds,
                            sensorData->sharedParameters()._leafBufferSize);
      cudaCheck(cudaMemcpy(_sb->devSensor(), _sb->hostSensor(), sizeof(nanomap::gpu::Sensor<float>), cudaMemcpyHostToDevice));
      cudaCheck(cudaMemcpy(_sb->devGridData(), _sb->hostGridData(), sizeof(nanomap::gpu::GridData), cudaMemcpyHostToDevice));
      *(_sb->hostFrustumLeafCount()) = 0;
      cudaCheck(cudaMemcpy(_sb->devFrustumLeafCount(), _sb->hostFrustumLeafCount(), sizeof(int), cudaMemcpyHostToDevice));
      *(_sb->hostRayCount()) = 0;
      cudaCheck(cudaMemcpy(_sb->devRayCount(), _sb->hostRayCount(), sizeof(int), cudaMemcpyHostToDevice));
    }

    void downloadCloudToSensor(std::shared_ptr<nanomap::sensor::SensorData> sensorData, cudaStream_t s0){
        _sb->pclHandle().deviceDownload(s0);
        cudaDeviceSynchronize();
        int cloudSize = sensorData->sharedParameters()._hRes*sensorData->sharedParameters()._vRes;
        nanomap::gpu::PointCloud* pcl = _sb->pclHandle().pointCloud();
        for(int i = 0; i < cloudSize; i++){
          if((*(pcl))(i).norm >= 0.0){
            (sensorData->pointCloud()).col(i)(0) = ((*(pcl))(i).x);
            (sensorData->pointCloud()).col(i)(1) = ((*(pcl))(i).y);
            (sensorData->pointCloud()).col(i)(2) = ((*(pcl))(i).z);
          }else{
              (sensorData->pointCloud()).col(i)(0) = ((*(pcl))(i).x);
              (sensorData->pointCloud()).col(i)(1) = ((*(pcl))(i).y);
              (sensorData->pointCloud()).col(i)(2) = nanf("");
          }

        }
    }
        std::shared_ptr<nanomap::gpu::SensorBucket> sensorBucket(){return _sb;}
      private:
        /******************************************************************************/
        int _frustumVoxelAllocation = 0;
        int _laserVoxelAllocation = 0;
        std::shared_ptr<nanomap::config::Config> _cf;
        std::shared_ptr<nanomap::gpu::SensorBucket> _sb;
    //  };


    };
  }
}
#endif

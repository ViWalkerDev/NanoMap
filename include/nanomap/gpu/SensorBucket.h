// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file SenorBucket.h
///
/// @author Violet Walker
///


#ifndef NANOMAP_HANDLER_SENSORBUCKET_H_INCLUDED
#define NANOMAP_HANDLER_SENSORBUCKET_H_INCLUDED
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/NanoVDB.h>
#include "nanomap/gpu/NodeWorker.h"
#include "nanomap/gpu/PointCloud.h"
#include "nanomap/gpu/Sensor.h"
#include "nanomap/gpu/GridData.h"
#include "nanomap/nanomap.h"
#include "nanomap/handler/handlerAssert.h"
#include "cuda_fp16.h"

namespace nanomap{
  namespace gpu{
    using BufferT = nanovdb::CudaDeviceBuffer;
    using ValueT  = float;
    using EigenVec = Eigen::Matrix<ValueT, 3, 1>;
    class SensorBucket
    {

    public:
      ~SensorBucket(){
        freeMemory();
      }

      SensorBucket(int maxPCLSize, int frustumLeafBufferSize)
      :_pclHandle(maxPCLSize),
      _leafHandle(frustumLeafBufferSize)
      {
              cudaCheck(cudaMalloc((void**)&_devSensor, sizeof(nanomap::gpu::Sensor<float>)));
              cudaCheck(cudaMallocHost((void**)&_hostSensor, sizeof(nanomap::gpu::Sensor<float>)));
              cudaCheck(cudaMalloc((void**)&_devGridData, sizeof(nanomap::gpu::GridData)));
              cudaCheck(cudaMallocHost((void**)&_hostGridData, sizeof(nanomap::gpu::GridData)));
              cudaCheck(cudaMalloc((void**)&_devFrustumLeafCount, sizeof(int)));
              cudaCheck(cudaMallocHost((void**)&_hostFrustumLeafCount, sizeof(int)));
              cudaCheck(cudaMalloc((void**)&_devRayCount, sizeof(int)));
              cudaCheck(cudaMallocHost((void**)&_hostRayCount, sizeof(int)));
              _leafHandle.deviceUpload();
              _pclHandle.deviceUpload();
      }

      void freeMemory(){
        cudaFreeHost(_hostSensor);
        cudaFree(_devSensor);
        cudaFreeHost(_hostGridData);
        cudaFree(_devGridData);
        cudaFreeHost(_hostRayCount);
        cudaFree(_devRayCount);
        if(_processType == 0 || _processType == 2){
          cudaFree(_activeLeafNodes);
          cudaFree(_activeFlags);
          cudaFree(_activeIndices);
          cudaFree(_devFrustumLeafCount);
          cudaFreeHost(_hostFrustumLeafCount);
          cudaFree(_devFrustumVoxelWorker);
          cudaFree(_devFrustumVoxelBuffer);
          cudaFreeHost(_hostFrustumVoxelBuffer);
          cudaFree(_devFrustumLeafBuffer);
          cudaFreeHost(_hostFrustumLeafBuffer);
        }

        if(_filterType == 1){
          if(_precisionType == 0){
            cudaFree(_voxelFilterBuffer);
          }
          if(_precisionType == 1){
            cudaFree(_voxelFilterBufferHalf2);
          }
          if(_precisionType == 2){
            cudaFree(_voxelFilterBufferSimple);
          }
          if(_precisionType == 3){
            cudaFree(_voxelFilterBufferSimpleQuad);
          }
        }
      }
      int getProcessType(){return _processType;}
      int getFilterType(){return _filterType;}
      int getPrecisionType(){return _precisionType;}
      int getFrustumVoxelAllocation(){return _frustumVoxelAllocation;}
      float getGridRes(){return _gridRes;}
      float getMappingRes(){return _mappingRes;}
      void setProcessType(int processType){_processType = processType;}
      void setFilterType(int filterType){_filterType = filterType;}
      void setPrecisionType(int precisionType){_precisionType = precisionType;}
      void setFrustumVoxelAllocation(int frustumVoxelAllocation){_frustumVoxelAllocation = frustumVoxelAllocation;}
      void setGridRes(float gridRes){_gridRes = gridRes;}
      void setMappingRes(float mappingRes){_mappingRes = mappingRes;}

      nanomap::gpu::PointCloudHandle<BufferT>& pclHandle(){return _pclHandle;}
      nanomap::gpu::NodeWorkerHandle<BufferT>& leafHandle(){return _leafHandle;}
      nanomap::gpu::Sensor<float>* devSensor(){return _devSensor;}
      nanomap::gpu::Sensor<float>* hostSensor(){return _hostSensor;}
      nanomap::gpu::GridData* devGridData(){return _devGridData;}
      nanomap::gpu::GridData* hostGridData(){return _hostGridData;}

      int* devFrustumLeafCount(){return _devFrustumLeafCount;}
      int* hostFrustumLeafCount(){return _hostFrustumLeafCount;}
      int* devFrustumLeafBuffer(){return _devFrustumLeafBuffer;}
      int* hostFrustumLeafBuffer(){return _hostFrustumLeafBuffer;}
      float* devFrustumVoxelWorker(){return _devFrustumVoxelWorker;}
      int8_t* devFrustumVoxelBuffer(){return _devFrustumVoxelBuffer;}
      int8_t* hostFrustumVoxelBuffer(){return _hostFrustumVoxelBuffer;}

      int* activeLeafNodes(){return _activeLeafNodes;}
      int* activeIndices(){return _activeIndices;}
      int* activeFlags(){return _activeFlags;}

      //Ray Count Variable for Filtering
      int* devRayCount(){return _devRayCount;}
      int* hostRayCount(){return _hostRayCount;}
      //Voxelized Filter Specific Buffers
      float* voxelFilterBuffer(){return _voxelFilterBuffer;}
      __hostdev__ __half2* voxelFilterBufferHalf2(){return _voxelFilterBufferHalf2;}
      int* voxelFilterBufferSimple(){return _voxelFilterBufferSimple;}
      int* voxelFilterBufferSimpleQuad(){return _voxelFilterBufferSimpleQuad;}
      //private:
        /******************************************************************************/
        //DEFAULT ALLOCATIONS
        int _filterType;
        int _processType;
        int _precisionType;
        float _mappingRes;
        float _gridRes;
        //Tracks current frustum and laser allocations.
        int _frustumVoxelAllocation;
        int _laserVoxelAllocation;

        //This is the container for the point cloud. This is populated either via a an external point cloud or a simulated one.
        //Can be used to represent a point in world space, or a unit direction and magnitude. This allows the same container to be reused when converting
        //From the world space point cloud to grid space rays.
        nanomap::gpu::PointCloudHandle<BufferT> _pclHandle;
        //The node handle is used when performing the node raycasting step of the GPU calculations. It tracks a dense representation of the observable node space.
        nanomap::gpu::NodeWorkerHandle<BufferT> _leafHandle;
        //Pointers for device and host sensor object. One object is used per manager,
        //and is updated as necessary prior to use with different senserData objects
        nanomap::gpu::Sensor<float>* _devSensor;
        nanomap::gpu::Sensor<float>* _hostSensor;
        //Pointer to GridData Object. This tracks important data necessary for GPU calculations.
        nanomap::gpu::GridData* _devGridData;
        nanomap::gpu::GridData* _hostGridData;
        //This variable tracks the active leafNode count.
        int* _devFrustumLeafCount;
        int* _hostFrustumLeafCount;
        //contains the index coordinates of all active leaf nodess with Z being the fastest moving axis.
        int* _devFrustumLeafBuffer;
        int* _hostFrustumLeafBuffer;
        float* _devFrustumVoxelWorker;

        //Used to transfer probability info from gpu to cpu in compact fashion
        int8_t* _devFrustumVoxelBuffer;
        int8_t* _hostFrustumVoxelBuffer;

        //Used for sorting active node information to improve CPU side grid access.
        int* _activeLeafNodes;
        int* _activeIndices;
        int* _activeFlags;

        /******************************************************************************/
        //FILTER SPECIFIC ALLOCATIONS
        //A dense representation of the observable voxel level environment
        //Used for bucketing points for voxel filtering.
        float* _voxelFilterBuffer;
        __half2* _voxelFilterBufferHalf2;
        int* _voxelFilterBufferSimple;
        int* _voxelFilterBufferSimpleQuad;
        //This variable tracks the number of pcl rays after filtering.
        int* _devRayCount;
        int* _hostRayCount;
      };


  }
}
#endif

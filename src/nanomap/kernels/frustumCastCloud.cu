// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file frustumCastCloud.cu
///
/// @author Violet Walker
///
/// @brief A CUDA kernel that performs frustum based ray casting.

#include <stdio.h>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>
#include <cub/device/device_partition.cuh>
#include "nanomap/gpu/NodeWorker.h"
#include "nanomap/gpu/GridData.h"
#include "nanomap/gpu/PointCloud.h"
#include "nanomap/gpu/SensorBucket.h"
#include "nanomap/gpu/Sensor.h"


/*****************************************************************************/
__global__ void clearNodeWorkerKernel(nanomap::gpu::NodeWorker& worker, int* activeIndices, int* activeFlags, int maxBufferSize)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= maxBufferSize){
        return;
    }
    //printf("clear %d \n", x);
      worker(x) = nanomap::gpu::NodeWorker::Node(0,0);
      *(activeIndices+x)=0;
      *(activeFlags+x)=0;
}
/*****************************************************************************/
__global__ void clearVoxelWorkerKernel(float* worker, int size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= size){
        return;
    }
      *(worker+x) = 0.0f;
}
/*****************************************************************************/
__global__ void activeCount(
            nanomap::gpu::NodeWorker& Array, int* activeIndices, int* activeFlags, int maxBufferSize)
{
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
              if(x >= maxBufferSize){
                  return;
              }
              if(Array(x).active == 1){
                *(activeIndices+x) = x;
                *(activeFlags+x) = 1;
              }

}
/*****************************************************************************/
__host__ void activePartition(int* activeIndices, int* activeFlags, int* activeLeafNodes, int maxBufferSize, int* devCount, cudaStream_t cStream)
{
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, activeIndices, activeFlags, activeLeafNodes, devCount, maxBufferSize, cStream);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run selection
  cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, activeIndices, activeFlags, activeLeafNodes, devCount, maxBufferSize, cStream);
  cudaStreamSynchronize(cStream);
  cudaFree(d_temp_storage);
}
/*****************************************************************************/
__global__ void activeAssignment(nanomap::gpu::NodeWorker& array, int* activeLeafNodes, int* count)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >=*count){
        return;
    }
    array(*(activeLeafNodes+x)).index = x;
}
/*****************************************************************************/
__global__ void setCount(nanomap::gpu::NodeWorker&     Array, int* count){
            Array.nodeCount() = *count;
}
/*****************************************************************************/
__global__ void nodePassHDDA(
  nanomap::gpu::PointCloud&                                         pclArray,
  int*                                                             devRayCount,
  nanomap::gpu::NodeWorker&                                        nodeArray,
  const nanomap::gpu::GridData&                                     gridData,
  const nanomap::gpu::Sensor<float>&                                  sensor)
  {
    using ValueT = float;
    using EigenVec = Eigen::Matrix<ValueT, 3, 1>;
    using Vec3T = nanovdb::Vec3<ValueT>;
    using RayT = nanovdb::Ray<ValueT>;
    using HDDA     = nanovdb::HDDA<RayT>;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("rayDir: %f | %f | %f \n", pclArray(index).x, pclArray(index).y, pclArray(index).z);
    if (index >= *devRayCount){
        return;
    }else if(pclArray(index).norm <= 0.0 || pclArray(index).norm >= sensor.maxVoxelRange()){
        return;
    }else{
        Vec3T rayEye(sensor.leafOriginOffset()(0),
                    sensor.leafOriginOffset()(1),
                    sensor.leafOriginOffset()(2));

        Vec3T rayDir(pclArray(index).x,
                      pclArray(index).y,
                        pclArray(index).z);


        ValueT maxTime = pclArray(index).norm;

        ValueT minTime = 0.0f;
        RayT pointRay(rayEye, rayDir);
        HDDA hdda;
        hdda.init(pointRay, minTime, maxTime, gridData.leafEdge());
        nanovdb::Coord voxel = hdda.voxel();
        minTime = hdda.time();
        int nodeX, nodeY, nodeZ;
        int count = 0;
        while(hdda.step()){
          nodeX = __float2int_rd((float)(voxel[0])/(float)gridData.leafEdge())-gridData.nodeMin()[0];
          nodeY = __float2int_rd((float)(voxel[1])/(float)gridData.leafEdge())-gridData.nodeMin()[1];
          nodeZ = __float2int_rd((float)(voxel[2])/(float)gridData.leafEdge())-gridData.nodeMin()[2];
          if(!(nodeX < 0 || nodeX >= gridData.nodeDim()[0] ||
               nodeY < 0 || nodeY >= gridData.nodeDim()[1] ||
               nodeZ < 0 || nodeZ >= gridData.nodeDim()[2])){
          (nodeArray)(nodeX, nodeY, nodeZ,
                                gridData.nodeDim()[0],
                                gridData.nodeDim()[1],
                                gridData.nodeDim()[2]).active = 1;
          }
          voxel = hdda.voxel();
        }
        nodeX = __float2int_rd((float)(voxel[0])/(float)gridData.leafEdge())-gridData.nodeMin()[0];
        nodeY = __float2int_rd((float)(voxel[1])/(float)gridData.leafEdge())-gridData.nodeMin()[1];
        nodeZ = __float2int_rd((float)(voxel[2])/(float)gridData.leafEdge())-gridData.nodeMin()[2];
        if(!(nodeX < 0 || nodeX >= gridData.nodeDim()[0] ||
             nodeY < 0 || nodeY >= gridData.nodeDim()[1] ||
             nodeZ < 0 || nodeZ >= gridData.nodeDim()[2])){
        (nodeArray)(nodeX, nodeY, nodeZ,
                              gridData.nodeDim()[0],
                              gridData.nodeDim()[1],
                              gridData.nodeDim()[2]).active = 1;
        //printf("%d | %d | %d \n", nodeX, nodeY, nodeZ);
        }
    }
  }
  /*****************************************************************************/
    __global__ void voxelPassHDDA(
      nanomap::gpu::PointCloud&                                         pclArray,
      int*                                                             devRayCount,
      nanomap::gpu::NodeWorker&                                        nodeArray,
      float*                                                               voxelWorker,
      const nanomap::gpu::GridData&                                     gridData,
      const nanomap::gpu::Sensor<float>&                                  sensor)
      {
        using ValueT = float;
        using EigenVec = Eigen::Matrix<ValueT, 3, 1>;
        using Vec3T = nanovdb::Vec3<ValueT>;
        using RayT = nanovdb::Ray<ValueT>;
        using HDDA     = nanovdb::HDDA<RayT>;
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= *devRayCount){
            return;
        }else if(pclArray(i).norm <= 0.0 ){
            return;
        }else{
            Vec3T rayEye(sensor.leafOriginOffset()(0),
                          sensor.leafOriginOffset()(1),
                            sensor.leafOriginOffset()(2));

            Vec3T rayDir(pclArray(i).x,
                          pclArray(i).y,
                            pclArray(i).z);
            int voxelX = 0;
            int voxelY = 0;
            int voxelZ = 0;
            int nodeVolume = gridData.leafEdge()*gridData.leafEdge()*gridData.leafEdge();

            ValueT time = 0;
            ValueT maxTime = pclArray(i).norm;
            RayT pointRay(rayEye, rayDir);
            HDDA hddaVoxel;


            hddaVoxel.init(pointRay,time,maxTime, 1);
            nanovdb::Coord voxel = hddaVoxel.voxel();
            int index = 0;

            nanomap::gpu::NodeWorker::Node nodeNode(0,0);
            int nodeX, nodeZ, nodeY;
            voxel = hddaVoxel.voxel();
            time = hddaVoxel.time();
            while(hddaVoxel.step()){
                nodeX = __float2int_rd((float)(voxel[0])/(float)gridData.leafEdge())-gridData.nodeMin()[0];
                nodeY = __float2int_rd((float)(voxel[1])/(float)gridData.leafEdge())-gridData.nodeMin()[1];
                nodeZ = __float2int_rd((float)(voxel[2])/(float)gridData.leafEdge())-gridData.nodeMin()[2];
                if(!(nodeX < 0 || nodeX >= gridData.nodeDim()[0] ||
                     nodeY < 0 || nodeY >= gridData.nodeDim()[1] ||
                     nodeZ < 0 || nodeZ >= gridData.nodeDim()[2])){
                nodeNode = (nodeArray)(
                                  nodeX,
                                  nodeY,
                                  nodeZ,
                                  gridData.nodeDim()[0],
                                  gridData.nodeDim()[1],
                                  gridData.nodeDim()[2]);

                voxelX = voxel[0]-__float2int_rd((float)(voxel[0])/(float)gridData.leafEdge())*gridData.leafEdge();
                voxelY = voxel[1]-__float2int_rd((float)(voxel[1])/(float)gridData.leafEdge())*gridData.leafEdge();
                voxelZ = voxel[2]-__float2int_rd((float)(voxel[2])/(float)gridData.leafEdge())*gridData.leafEdge();
                if(nodeNode.index>=0 && nodeNode.index<nodeArray.nodeCount()){
                  index = nodeNode.index*nodeVolume+voxelZ+voxelY*gridData.leafEdge()+voxelX*gridData.leafEdge()*gridData.leafEdge();
                  atomicAdd((voxelWorker+index), sensor.clogOddsMiss());//*pclArray(i).count);
                }
              }
              voxel = hddaVoxel.voxel();
              time = hddaVoxel.time();
            }

            nodeX = __float2int_rd((float)(voxel[0])/(float)gridData.leafEdge())-gridData.nodeMin()[0];
            nodeY = __float2int_rd((float)(voxel[1])/(float)gridData.leafEdge())-gridData.nodeMin()[1];
            nodeZ = __float2int_rd((float)(voxel[2])/(float)gridData.leafEdge())-gridData.nodeMin()[2];
            if(!(nodeX < 0 || nodeX >= gridData.nodeDim()[0] ||
                 nodeY < 0 || nodeY >= gridData.nodeDim()[1] ||
                 nodeZ < 0 || nodeZ >= gridData.nodeDim()[2])){
            nodeNode = (nodeArray)(
                              nodeX,
                              nodeY,
                              nodeZ,
                              gridData.nodeDim()[0],
                              gridData.nodeDim()[1],
                              gridData.nodeDim()[2]);
            voxelX = voxel[0]-__float2int_rd((float)(voxel[0])/(float)gridData.leafEdge())*gridData.leafEdge();
            voxelY = voxel[1]-__float2int_rd((float)(voxel[1])/(float)gridData.leafEdge())*gridData.leafEdge();
            voxelZ = voxel[2]-__float2int_rd((float)(voxel[2])/(float)gridData.leafEdge())*gridData.leafEdge();
              if(nodeNode.index>=0 && nodeNode.index<nodeArray.nodeCount()){
                int indexOffset = voxelZ+voxelY*gridData.leafEdge()+voxelX*gridData.leafEdge()*gridData.leafEdge();
                index = nodeNode.index*nodeVolume+voxelZ+voxelY*gridData.leafEdge()+voxelX*gridData.leafEdge()*gridData.leafEdge();
                atomicAdd((voxelWorker+index), sensor.clogOddsHit());
              }
            }
          }
        }

/*****************************************************************************/
__global__ void   populateNodeBuffer(
      nanomap::gpu::NodeWorker&               nodeArray,
                           int*           devNodeBuffer,
  const nanomap::gpu::GridData&                gridData,
  const nanomap::gpu::Sensor<float>&             sensor
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if(x >= gridData.nodeDim()[0] || y >= gridData.nodeDim()[1] || z >= gridData.nodeDim()[2]) {
        return;
    }
    int index = z + y*gridData.nodeDim()[2] + x*gridData.nodeDim()[2]*gridData.nodeDim()[1];
    if((nodeArray)(index).active == 1){
      *(devNodeBuffer+3*((nodeArray)(index).index)) = (x+gridData.nodeMin()[0])*gridData.leafEdge()+sensor.leafOffset()[0];
      *(devNodeBuffer+3*((nodeArray)(index).index)+1) = (y+gridData.nodeMin()[1])*gridData.leafEdge()+sensor.leafOffset()[1];
      *(devNodeBuffer+3*((nodeArray)(index).index)+2) = (z+gridData.nodeMin()[2])*gridData.leafEdge()+sensor.leafOffset()[2];
    }
}
/*****************************************************************************/
__global__ void   populateVoxelBuffer(
                              int*           devNodeBuffer,
                              int8_t*        devVoxelBuffer,
                              float*         devVoxelWorker,
  const nanomap::gpu::GridData&                gridData,
                              int*                devCount)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= *devCount){
        return;
    }
    int index;
    int value = 0;
    int voxelVolume = gridData.leafEdge()* gridData.leafEdge()* gridData.leafEdge();
    for(int i = 0; i<voxelVolume; i++){
      index = x*voxelVolume+i;
      value = (int)((*(devVoxelWorker+index))*100);
      if(value > 127){
        value = 127;
      }else if(value < -128){
        value = -128;
      }
      *(devVoxelBuffer+index) = value;
    }
}
/*****************************************************************************/
__global__ void clearNodeBuffer(
                                int*         devNodeBuffer,
                                int*          devCount
){
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(x >= *devCount){
      return;
  }
  *(devNodeBuffer+x*3) = 0;
  *(devNodeBuffer+x*3+1) = 0;
  *(devNodeBuffer+x*3+2) = 0;
}

/*****************************************************************************/
// This is called by the host
extern "C" void frustumCastCloud(  nanomap::gpu::SensorBucket&                                          sensorBucket,
                               cudaStream_t                                                                 s0,
                               cudaStream_t                                                                 s1)
{
    using ValueT = float;
    using EigenVec = Eigen::Matrix<ValueT, 3, 1>;
    using Vec3T = nanovdb::Vec3<ValueT>;
    auto        round = [](int a, int b) { return (a + b - 1) / b; };

    //Clear All Node Worker Related Arrays using parallel clear
    const dim3 rayThreads(512), rayNumBlocks(round(*(sensorBucket.hostRayCount()), rayThreads.x));
    int nodeSize = sensorBucket.hostGridData()->nodeBufferSize();
    //printf("nodeSize = %d", nodeSize);
    const dim3 nodeThreads(512), nodeNumBlocks(round(nodeSize, nodeThreads.x));
    clearNodeWorkerKernel<<<nodeNumBlocks, nodeThreads, 0, s0>>>(*(sensorBucket.leafHandle().deviceNodeWorker()),
                                                                    sensorBucket.activeIndices(),
                                                                    sensorBucket.activeFlags(),
                                                                    nodeSize);
    //Determine the active nodes using ray casting
    nodePassHDDA<<<rayNumBlocks, rayThreads, 0, s0>>>(*(sensorBucket.pclHandle().devicePointCloud()),
                                                        sensorBucket.devRayCount(),
                                                       *(sensorBucket.leafHandle().deviceNodeWorker()),
                                                          *(sensorBucket.devGridData()),
                                                          *(sensorBucket.devSensor()));
    //Count the active number of nodes and assign each node worker an index
    activeCount<<<nodeNumBlocks, nodeThreads, 0, s0>>>(*(sensorBucket.leafHandle().deviceNodeWorker()),
                                                          sensorBucket.activeIndices(),
                                                          sensorBucket.activeFlags(),
                                                          nodeSize);
    //Partition the worker arrays to remove non active nodes from the resultant node array
    activePartition(sensorBucket.activeIndices(),
                    sensorBucket.activeFlags(),
                    sensorBucket.activeLeafNodes(),
                    nodeSize,
                    sensorBucket.devFrustumLeafCount(), s0);
    //Copy the leaf node count to the host.
    cudaMemcpy(sensorBucket.hostFrustumLeafCount(), sensorBucket.devFrustumLeafCount(), sizeof(int), cudaMemcpyDeviceToHost);
    //Set the leaf node count in the node worker object.
    setCount<<<1,1,0, s0>>>(*(sensorBucket.leafHandle().deviceNodeWorker()), sensorBucket.hostFrustumLeafCount());
    //If there are active leaf nodes, then we need to do voxel level raycasting.
    if(*(sensorBucket.hostFrustumLeafCount()) > 0){
      const dim3 assignmentThreads(512), assignmentBlocks(round(*(sensorBucket.hostFrustumLeafCount()), assignmentThreads.x));
      activeAssignment<<<assignmentBlocks,assignmentThreads,0, s0>>>(*(sensorBucket.leafHandle().deviceNodeWorker()),
                                                                        sensorBucket.activeLeafNodes(), sensorBucket.devFrustumLeafCount());

      cudaStreamSynchronize(s0);
      int voxelCount = (*(sensorBucket.hostFrustumLeafCount()))*sensorBucket.hostGridData()->leafVolume();;
      if(voxelCount > 0){
        if(voxelCount > sensorBucket.getFrustumVoxelAllocation()){
          //NEED TO INCREASE ALLOCATION SIZE OF VOXEL ARRAYS
          //TO DO
        }
        //Clear voxel workers
        const dim3 voxelThreads(512), voxelNumBlocks(round(voxelCount, voxelThreads.x));
        clearVoxelWorkerKernel<<<voxelNumBlocks, voxelThreads, 0, s1>>>(sensorBucket.devFrustumVoxelWorker(), voxelCount);
        //Define node threads and blocks
        const dim3 node3DThreads(8,8,8), node3DBlocks(round(sensorBucket.hostGridData()->nodeDim()[0], node3DThreads.x),
                                                    round(sensorBucket.hostGridData()->nodeDim()[1], node3DThreads.y),
                                                    round(sensorBucket.hostGridData()->nodeDim()[2], node3DThreads.z));

        populateNodeBuffer<<<node3DBlocks, node3DThreads, 0, s0>>>(*(sensorBucket.leafHandle().deviceNodeWorker()),
                                                                      sensorBucket.devFrustumLeafBuffer(),
                                                                      *(sensorBucket.devGridData()),
                                                                      *(sensorBucket.devSensor()));
        //Depending on the filter type,
        voxelPassHDDA<<<rayNumBlocks, rayThreads, 0, s1>>>(*(sensorBucket.pclHandle().devicePointCloud()),
                                                            sensorBucket.devRayCount(),
                                                           *(sensorBucket.leafHandle().deviceNodeWorker()),
                                                             sensorBucket.devFrustumVoxelWorker(),
                                                             *(sensorBucket.devGridData()),
                                                             *(sensorBucket.devSensor()));

        //Populate the Voxel Level Buffer with the results of the raycast.
        const dim3 voxelBufferThreads(512), voxelBufferBlocks(round(*(sensorBucket.hostFrustumLeafCount()), voxelBufferThreads.x));
        populateVoxelBuffer<<<voxelBufferBlocks, voxelBufferThreads, 0, s1>>>(
                                                                sensorBucket.devFrustumLeafBuffer(),
                                                                sensorBucket.devFrustumVoxelBuffer(),
                                                                sensorBucket.devFrustumVoxelWorker(),
                                                                *(sensorBucket.devGridData()),
                                                                sensorBucket.devFrustumLeafCount());
      }
    }
}

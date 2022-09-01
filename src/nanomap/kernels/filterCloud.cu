
// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file filterCloud.cu
///
/// @author Violet Walker
///
/// @brief A CUDA kernel that performs filtering/pre-processing of a sensor input.

#include <stdio.h>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include "nanomap/gpu/Sensor.h"
#include "nanomap/gpu/PointCloud.h"
#include "nanomap/gpu/GridData.h"
#include "nanomap/gpu/SensorBucket.h"
#include "cuda_fp16.h"

__device__ static inline uint8_t atomicCharIncrement(uint8_t* address, uint8_t val) {
    // offset, in bytes, of the char* address within the 32-bit address of the space that overlaps it
    size_t long_address_modulo = (size_t) address & 3;
    // the 32-bit address that overlaps the same memory
    auto* base_address = (unsigned int*) ((uint8_t*) address - long_address_modulo);
    // A 0x3210 selector in __byte_perm will simply select all four bytes in the first argument in the same order.
    // The "4" signifies the position where the first byte of the second argument will end up in the output.
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    // for selecting bytes within a 32-bit chunk that correspond to the char* address (relative to base_address)
    unsigned int selector = selectors[long_address_modulo];
    unsigned int long_old, long_assumed, long_val, replacement;

    long_old = *base_address;

    do {
        long_assumed = long_old;
        // replace bits in long_old that pertain to the char address with those from val
        long_val = __byte_perm(long_old, 0, long_address_modulo) + val;
        replacement = __byte_perm(long_old, long_val, selector);
        long_old = atomicCAS(base_address, long_assumed, replacement);
    } while (long_old != long_assumed);
    return __byte_perm(long_old, 0, long_address_modulo);
}

__global__ void voxelBufferClear(float* buffer, int size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= size){
        return;
    }
      *(buffer+x*4) = 0.0;
      *(buffer+x*4+1) = 0.0;
      *(buffer+x*4+2) = 0.0;
      *(buffer+x*4+3) = 0.0;
}

__global__ void voxelBufferClearHalf2(__half2* buffer, int size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= size){
        return;
    }

      *(buffer+x*2) = __half2half2((__half)0.0);
      *(buffer+x*2+1) = __half2half2((__half)0.0);
}

__global__ void voxelBufferClearSimple(int* buffer, int size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= size){
        return;
    }
      *(buffer+x) = 0;
}
__global__ void voxelBufferClearSimpleQuad(int* buffer, int size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= size){
        return;
    }
      *(buffer+x) = 0;
}


__global__ void voxelFilter(
            nanomap::gpu::PointCloud&                               pclArray,
            float*                                                     voxelFilterBuffer,
            int                                                          size,
            const nanomap::gpu::Sensor<float>&                        sensor,
            const nanomap::gpu::GridData&                             gridData)
{
    using ValueT = float;
    using EigenVec = Eigen::Matrix<ValueT, 3, 1>;
    using Vec3T = nanovdb::Vec3<ValueT>;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int index = w+h*sensor.hRes();
    //Filter if point is valid
    if (w >= sensor.hRes() || h >= sensor.vRes()){
        return;
    }else if (isnan(pclArray(index).x) || isnan(pclArray(index).y) || isnan(pclArray(index).z) || pclArray(index).norm < 0){
          pclArray(index).norm = -1;
          return;
    }else if (pclArray(index).x < sensor.worldMin()(0) || pclArray(index).x > sensor.worldMax()(0)){
          pclArray(index).norm = -1;
          return;
    }else if (pclArray(index).y < sensor.worldMin()(1) || pclArray(index).y > sensor.worldMax()(1)){
          pclArray(index).norm = -1;
          return;
    }else if (pclArray(index).z < sensor.worldMin()(2) || pclArray(index).z > sensor.worldMax()(2)){
          pclArray(index).norm = -1;
          return;
    }
    EigenVec point(pclArray(index).x,pclArray(index).y, pclArray(index).z);
    if(point.norm()>0.0 && point.norm()< sensor.maxRange()){

      Vec3T rayDir(sensor.rotation()(0,0)*point(0)+sensor.rotation()(0,1)*point(1)+sensor.rotation()(0,2)*point(2),
                 sensor.rotation()(1,0)*point(0)+sensor.rotation()(1,1)*point(1)+sensor.rotation()(1,2)*point(2),
                 sensor.rotation()(2,0)*point(0)+sensor.rotation()(2,1)*point(1)+sensor.rotation()(2,2)*point(2));
      int x = __float2int_rd((float)rayDir[0]/(float)sensor.gridRes()+sensor.voxelOriginOffset()(0))-gridData.voxelMin()[0];
      int y = __float2int_rd((float)rayDir[1]/(float)sensor.gridRes()+sensor.voxelOriginOffset()(1))-gridData.voxelMin()[1];
      int z = __float2int_rd((float)rayDir[2]/(float)sensor.gridRes()+sensor.voxelOriginOffset()(2))-gridData.voxelMin()[2];

      if(!(x>=gridData.voxelDim()[0] || y>=gridData.voxelDim()[1] || z>=gridData.voxelDim()[2]) && (!(x<0 || y<0 || z<0))){
        int i = z + y*gridData.voxelDim()[2] + x*gridData.voxelDim()[2]*gridData.voxelDim()[1];
        if(i < size){
          atomicAdd(voxelFilterBuffer+i*4,   (rayDir[0]/sensor.gridRes()-(__float2int_rd(rayDir[0]/sensor.gridRes()))));
          atomicAdd(voxelFilterBuffer+i*4+1, (rayDir[1]/sensor.gridRes()-(__float2int_rd(rayDir[1]/sensor.gridRes()))));
          atomicAdd(voxelFilterBuffer+i*4+2, (rayDir[2]/sensor.gridRes()-(__float2int_rd(rayDir[2]/sensor.gridRes()))));
          atomicAdd(voxelFilterBuffer+i*4+3, 1.0);
        }
      }
    }
  }
 
    __global__ void voxelFilterSimple(
                nanomap::gpu::PointCloud&                               pclArray,
                int*                                       voxelFilterBuffer,
                int                                                          size,
                const nanomap::gpu::Sensor<float>&                        sensor,
                const nanomap::gpu::GridData&                             gridData)
    {
        using ValueT = float;
        using EigenVec = Eigen::Matrix<ValueT, 3, 1>;
        using Vec3T = nanovdb::Vec3<ValueT>;
        const int w = blockIdx.x * blockDim.x + threadIdx.x;
        const int h = blockIdx.y * blockDim.y + threadIdx.y;
        const int index = w+h*sensor.hRes();
        //Filter if point is valid
        if (w >= sensor.hRes() || h >= sensor.vRes()){
            return;
        }else if (isnan(pclArray(index).x) || isnan(pclArray(index).y) || isnan(pclArray(index).z) || pclArray(index).norm < 0){
              pclArray(index).norm = -1;
              return;
        }else if (pclArray(index).x < sensor.worldMin()(0) || pclArray(index).x > sensor.worldMax()(0)){
              pclArray(index).norm = -1;
              return;
        }else if (pclArray(index).y < sensor.worldMin()(1) || pclArray(index).y > sensor.worldMax()(1)){
              pclArray(index).norm = -1;
              return;
        }else if (pclArray(index).z < sensor.worldMin()(2) || pclArray(index).z > sensor.worldMax()(2)){
              pclArray(index).norm = -1;
              return;
        }
        EigenVec point(pclArray(index).x,pclArray(index).y, pclArray(index).z);
        if(point.norm()>0.0 && point.norm()< sensor.maxRange()){

          Vec3T rayDir(sensor.rotation()(0,0)*point(0)+sensor.rotation()(0,1)*point(1)+sensor.rotation()(0,2)*point(2),
                     sensor.rotation()(1,0)*point(0)+sensor.rotation()(1,1)*point(1)+sensor.rotation()(1,2)*point(2),
                     sensor.rotation()(2,0)*point(0)+sensor.rotation()(2,1)*point(1)+sensor.rotation()(2,2)*point(2));
          int x = __float2int_rd((float)rayDir[0]/(float)sensor.gridRes()+sensor.voxelOriginOffset()(0))-gridData.voxelMin()[0];
          int y = __float2int_rd((float)rayDir[1]/(float)sensor.gridRes()+sensor.voxelOriginOffset()(1))-gridData.voxelMin()[1];
          int z = __float2int_rd((float)rayDir[2]/(float)sensor.gridRes()+sensor.voxelOriginOffset()(2))-gridData.voxelMin()[2];
          if(!(x>=gridData.voxelDim()[0] || y>=gridData.voxelDim()[1] || z>=gridData.voxelDim()[2]) && (!(x<0 || y<0 || z<0))){
            int i = z + y*gridData.voxelDim()[2] + x*gridData.voxelDim()[2]*gridData.voxelDim()[1];
            if(i < size){
            atomicAdd(voxelFilterBuffer+i, 1);
            }
          }
        }
      }
      __global__ void voxelFilterSimpleQuad(
                  nanomap::gpu::PointCloud&                               pclArray,
                  int*                                       voxelFilterBuffer,
                  int                                                          size,
                  const nanomap::gpu::Sensor<float>&                        sensor,
                  const nanomap::gpu::GridData&                             gridData)
      {

          using ValueT = float;
          using EigenVec = Eigen::Matrix<ValueT, 3, 1>;
          using Vec3T = nanovdb::Vec3<ValueT>;
          const int w = blockIdx.x * blockDim.x + threadIdx.x;
          const int h = blockIdx.y * blockDim.y + threadIdx.y;
          const int index = w+h*sensor.hRes();
          //Filter if point is valid
          if (w >= sensor.hRes() || h >= sensor.vRes()){
              return;
          }else if (isnan(pclArray(index).x) || isnan(pclArray(index).y) || isnan(pclArray(index).z) || pclArray(index).norm < 0){
                pclArray(index).norm = -1;
                return;
          }else if (pclArray(index).x < sensor.worldMin()(0) || pclArray(index).x > sensor.worldMax()(0)){
                pclArray(index).norm = -1;
                return;
          }else if (pclArray(index).y < sensor.worldMin()(1) || pclArray(index).y > sensor.worldMax()(1)){
                pclArray(index).norm = -1;
                return;
          }else if (pclArray(index).z < sensor.worldMin()(2) || pclArray(index).z > sensor.worldMax()(2)){
                pclArray(index).norm = -1;
                return;
          }
          EigenVec point(pclArray(index).x,pclArray(index).y, pclArray(index).z);
          if(point.norm()>0.0 && point.norm()< sensor.maxRange()){

            Vec3T rayDir(sensor.rotation()(0,0)*point(0)+sensor.rotation()(0,1)*point(1)+sensor.rotation()(0,2)*point(2),
                       sensor.rotation()(1,0)*point(0)+sensor.rotation()(1,1)*point(1)+sensor.rotation()(1,2)*point(2),
                       sensor.rotation()(2,0)*point(0)+sensor.rotation()(2,1)*point(1)+sensor.rotation()(2,2)*point(2));
            int x = __float2int_rd((float)rayDir[0]/(float)sensor.gridRes()+sensor.voxelOriginOffset()(0))-gridData.voxelMin()[0];
            int y = __float2int_rd((float)rayDir[1]/(float)sensor.gridRes()+sensor.voxelOriginOffset()(1))-gridData.voxelMin()[1];
            int z = __float2int_rd((float)rayDir[2]/(float)sensor.gridRes()+sensor.voxelOriginOffset()(2))-gridData.voxelMin()[2];
            if(!(x>=gridData.voxelDim()[0] || y>=gridData.voxelDim()[1] || z>=gridData.voxelDim()[2]) && (!(x<0 || y<0 || z<0))){
              int i = z + y*gridData.voxelDim()[2] + x*gridData.voxelDim()[2]*gridData.voxelDim()[1];
              if(i < size){
                atomicCharIncrement(((uint8_t*)(voxelFilterBuffer)+i), (uint8_t)1);
              }
            }
          }
        }

  __global__ void rayFromFilter(
              nanomap::gpu::PointCloud&                               pclArray,
              float*                                                     voxelFilterBuffer,
              int*                                                        devRayCount,
              const nanomap::gpu::Sensor<float>&                        sensor,
              const nanomap::gpu::GridData&                             gridData)
  {
    using ValueT = float;
    using EigenVec = Eigen::Matrix<ValueT, 3, 1>;
    using Vec3T = nanovdb::Vec3<ValueT>;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if(x >= gridData.voxelDim()[0] || y >= gridData.voxelDim()[1] || z >= gridData.voxelDim()[2]){
      return;
    }
    int index = z+y*gridData.voxelDim()[2] + x*gridData.voxelDim()[1]*gridData.voxelDim()[2];
    float count = *(voxelFilterBuffer+index*4+3);
    if(count>=1.0){
      float px = (*(voxelFilterBuffer+index*4));
      float py = (*(voxelFilterBuffer+index*4+1));
      float pz = (*(voxelFilterBuffer+index*4+2));

      EigenVec point(px/count+(float)x+(float)(gridData.voxelMin()[0]),
                      py/count+(float)y+(float)(gridData.voxelMin()[1]),
                        pz/count+(float)z+(float)(gridData.voxelMin()[2]));
      ValueT maxTime = point.norm();
      point.normalize();
      if(maxTime > 0.0 && maxTime < sensor.maxVoxelRange()){

          int i = atomicAdd(devRayCount, 1);

          pclArray(i) = nanomap::gpu::PointCloud::Point(point(0),
                                                            point(1),
                                                            point(2),
                                                            maxTime,
                                                            count);
        }
    }
}

__global__ void rayFromFilterSimple(
            nanomap::gpu::PointCloud&                               pclArray,
            int*                                                     voxelFilterBuffer,
            int*                                                        devRayCount,
            const nanomap::gpu::Sensor<float>&                        sensor,
            const nanomap::gpu::GridData&                             gridData)
{
  using ValueT = float;
  using EigenVec = Eigen::Matrix<ValueT, 3, 1>;
  using Vec3T = nanovdb::Vec3<ValueT>;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  if(x >= gridData.voxelDim()[0] || y >= gridData.voxelDim()[1] || z >= gridData.voxelDim()[2]){
    return;
  }
  int index = z+y*gridData.voxelDim()[2] + x*gridData.voxelDim()[1]*gridData.voxelDim()[2];
  int count = *(voxelFilterBuffer+index);
  if(count>=1){
    EigenVec point((float)(x+0.5+gridData.voxelMin()[0]),
                    (float)(y+0.5+gridData.voxelMin()[1]),
                    (float)(z+0.5+gridData.voxelMin()[2]));
    ValueT maxTime = point.norm();
    point.normalize();
    if(maxTime > 0.0 && maxTime < sensor.maxVoxelRange()){

        int i = atomicAdd(devRayCount, 1);

        pclArray(i) = nanomap::gpu::PointCloud::Point(point(0),
                                                          point(1),
                                                          point(2),
                                                          maxTime,
                                                          count);
      }
  }
}
__global__ void rayFromFilterSimpleQuad(
            nanomap::gpu::PointCloud&                               pclArray,
            int*                                                     voxelFilterBuffer,
            int                                                     indexSize,
            int*                                                        devRayCount,
            const nanomap::gpu::Sensor<float>&                        sensor,
            const nanomap::gpu::GridData&                             gridData)
{
  using ValueT = float;
  using EigenVec = Eigen::Matrix<ValueT, 3, 1>;
  using Vec3T = nanovdb::Vec3<ValueT>;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  if(x >= gridData.voxelDim()[0] || y >= gridData.voxelDim()[1] || z >= gridData.voxelDim()[2]){
    return;
  }

  int index = z+y*gridData.voxelDim()[2] + x*gridData.voxelDim()[1]*gridData.voxelDim()[2];
  if(index >= indexSize){

    return;
  }
  uint8_t count = *((uint8_t*)voxelFilterBuffer+index);
  if(count>=1){
    EigenVec point((float)(x+0.5+gridData.voxelMin()[0]),
                    (float)(y+0.5+gridData.voxelMin()[1]),
                    (float)(z+0.5+gridData.voxelMin()[2]));
    ValueT maxTime = point.norm();
    point.normalize();
    if(maxTime > 0.0 && maxTime < sensor.maxVoxelRange()){

        int i = atomicAdd(devRayCount, 1);
        pclArray(i) = nanomap::gpu::PointCloud::Point(point(0),
                                                          point(1),
                                                          point(2),
                                                          maxTime,
                                                          count);
      }
   }
}
__global__ void passthroughFilter(
            nanomap::gpu::PointCloud&                               pclArray,
            const nanomap::gpu::Sensor<float>&                        sensor)
{
    using ValueT = float;
    using EigenVec = Eigen::Matrix<ValueT, 3, 1>;
    using Vec3T = nanovdb::Vec3<ValueT>;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int index = w+h*sensor.hRes();
    //Filter if point is valid
    if (w >= sensor.hRes() || h >= sensor.vRes()){
        return;
    }else if (isnan(pclArray(index).x) || pclArray(index).norm < 0){
        return;
    }else if (pclArray(index).x < sensor.worldMin()(0) || pclArray(index).x > sensor.worldMax()(0)){
        pclArray(index).norm = -1.0;
        return;
    }else if (pclArray(index).y < sensor.worldMin()(1) || pclArray(index).y > sensor.worldMax()(1)){
        pclArray(index).norm = -1.0;
        return;
    }else if (pclArray(index).z < sensor.worldMin()(2) || pclArray(index).z > sensor.worldMax()(2)){
        pclArray(index).norm = -1.0;
        return;
    }

    //If point is valid, rotate to sensor pose and calculate length in terms of voxels
    EigenVec point(pclArray(index).x, pclArray(index).y, pclArray(index).z);

    ValueT maxTime = point.norm()/sensor.gridRes();
    if(maxTime < sensor.maxVoxelRange()){
    Vec3T rayDir(sensor.rotation()(0,0)*point(0)+sensor.rotation()(0,1)*point(1)+sensor.rotation()(0,2)*point(2),
                  sensor.rotation()(1,0)*point(0)+sensor.rotation()(1,1)*point(1)+sensor.rotation()(1,2)*point(2),
                  sensor.rotation()(2,0)*point(0)+sensor.rotation()(2,1)*point(1)+sensor.rotation()(2,2)*point(2));

    //printf("%f, %f, %f \n", rayDir[0],rayDir[1],rayDir[2]);
    rayDir.normalize();
    pclArray(index) = nanomap::gpu::PointCloud::Point(rayDir[0], rayDir[1], rayDir[2], maxTime, 1.0);
    }
}

extern "C" void filterCloud(nanomap::gpu::SensorBucket& sensorBucket, cudaStream_t s0)
{
  using ValueT = float;
  using EigenVec = Eigen::Matrix<ValueT, 3, 1>;
  using Vec3T = nanovdb::Vec3<ValueT>;
  auto        round = [](int a, int b) { return (a + b - 1) / b; };
  const dim3 rayThreads(8,8), rayNumBlocks(round(sensorBucket.hostSensor()->hRes(),rayThreads.x),
                                           round(sensorBucket.hostSensor()->vRes(), rayThreads.y));

  cudaDeviceSynchronize();
  if(sensorBucket.hostSensor()->type() == 0 && sensorBucket.getFilterType() == 0){
    passthroughFilter<<<rayNumBlocks, rayThreads, 0, s0>>>(*(sensorBucket.pclHandle().devicePointCloud()), *(sensorBucket.devSensor()));
    *(sensorBucket.hostRayCount()) = (sensorBucket.hostSensor()->hRes())*(sensorBucket.hostSensor()->vRes());
    cudaMemcpy(sensorBucket.devRayCount(), sensorBucket.hostRayCount(), sizeof(int), cudaMemcpyHostToDevice);
    cudaStreamSynchronize(s0);
  }else if(sensorBucket.getFilterType()==1){
    int voxelSize = sensorBucket.hostGridData()->voxelDim()[0]
                    *sensorBucket.hostGridData()->voxelDim()[1]
                    *sensorBucket.hostGridData()->voxelDim()[2];
    const dim3 voxelThreads(512), voxelNumBlocks(round(voxelSize,voxelThreads.x));
    const dim3 voxel3DThreads(8,8,8), voxel3DBlocks(round(sensorBucket.hostGridData()->voxelDim()[0], voxel3DThreads.x),
                                                round(sensorBucket.hostGridData()->voxelDim()[1], voxel3DThreads.y),
                                                round(sensorBucket.hostGridData()->voxelDim()[2], voxel3DThreads.z));
    if(sensorBucket.getPrecisionType()==0){
      voxelBufferClear<<<voxelNumBlocks, voxelThreads, 0, s0>>>(sensorBucket.voxelFilterBuffer(), voxelSize);
      voxelFilter<<<rayNumBlocks, rayThreads, 0, s0>>>(*(sensorBucket.pclHandle().devicePointCloud()),
                                                        sensorBucket.voxelFilterBuffer(),
                                                        voxelSize,
                                                        *(sensorBucket.devSensor()),
                                                        *(sensorBucket.devGridData()));
      rayFromFilter<<<voxel3DBlocks, voxel3DThreads, 0, s0>>>(*(sensorBucket.pclHandle().devicePointCloud()),
                                                        sensorBucket.voxelFilterBuffer(),
                                                        sensorBucket.devRayCount(),
                                                        *(sensorBucket.devSensor()),
                                                        *(sensorBucket.devGridData()));
    }else if(sensorBucket.getPrecisionType()==2){
      voxelBufferClearSimple<<<voxelNumBlocks, voxelThreads, 0, s0>>>(sensorBucket.voxelFilterBufferSimple(), voxelSize);
      voxelFilterSimple<<<rayNumBlocks, rayThreads, 0, s0>>>(*(sensorBucket.pclHandle().devicePointCloud()),
                                                        sensorBucket.voxelFilterBufferSimple(),
                                                        voxelSize,
                                                        *(sensorBucket.devSensor()),
                                                        *(sensorBucket.devGridData()));
      rayFromFilterSimple<<<voxel3DBlocks, voxel3DThreads, 0, s0>>>(*(sensorBucket.pclHandle().devicePointCloud()),
                                                        sensorBucket.voxelFilterBufferSimple(),
                                                        sensorBucket.devRayCount(),
                                                        *(sensorBucket.devSensor()),
                                                        *(sensorBucket.devGridData()));
    }else if(sensorBucket.getPrecisionType()==3){
      const dim3 voxelQuadThreads(512), voxelQuadNumBlocks(round(ceil((float)voxelSize/4.0),voxelThreads.x));
      voxelBufferClearSimpleQuad<<<voxelQuadNumBlocks, voxelQuadThreads, 0, s0>>>(sensorBucket.voxelFilterBufferSimpleQuad(), ceil((float)voxelSize/4.0));
      voxelFilterSimpleQuad<<<rayNumBlocks, rayThreads, 0, s0>>>(*(sensorBucket.pclHandle().devicePointCloud()),
                                                        sensorBucket.voxelFilterBufferSimpleQuad(),
                                                        voxelSize,
                                                        *(sensorBucket.devSensor()),
                                                        *(sensorBucket.devGridData()));
      rayFromFilterSimpleQuad<<<voxel3DBlocks, voxel3DThreads, 0, s0>>>(*(sensorBucket.pclHandle().devicePointCloud()),
                                                        sensorBucket.voxelFilterBufferSimpleQuad(),
                                                        voxelSize,
                                                        sensorBucket.devRayCount(),
                                                        *(sensorBucket.devSensor()),
                                                        *(sensorBucket.devGridData()));
    }
    cudaMemcpyAsync(sensorBucket.hostRayCount(), sensorBucket.devRayCount(), sizeof(int), cudaMemcpyDeviceToHost,s0);
    cudaStreamSynchronize(s0);
  }
}

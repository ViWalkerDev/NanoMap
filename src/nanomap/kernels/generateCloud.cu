// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file generateCloud.cu
///
/// @author Violet Walker
///
/// @brief A CUDA kernel that generates a sensor output from sensor info and a nanoVDB Grid.

#include <stdio.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/CudaDeviceBuffer.h> // for CUDA memory management
#include <nanovdb/util/Ray.h> // for nanovdb::Ray
#include <nanovdb/util/HDDA.h>
#include "nanomap/gpu/PointCloud.h"
#include "nanomap/gpu/Sensor.h"
#include "nanomap/gpu/SensorBucket.h"


template<typename RayT, typename AccT, typename CoordT>
__hostdev__ bool getActiveCrossing(RayT& ray, AccT& acc, CoordT& ijk, typename AccT::ValueType& v, float& t)
{
    if (!ray.clip(acc.root().bbox()) || ray.t1() > 1e20)
        return false; // clip ray to bbox
    ijk = nanovdb::RoundDown<CoordT>(ray.start());
    nanovdb::HDDA<RayT, CoordT> hdda(ray, 1);

    while(hdda.step()){
      if(hdda.dim() != acc.getDim(hdda.voxel(),ray)){
        hdda.update(ray, acc.getDim(hdda.voxel(), ray));
      }
      if(hdda.dim()>1){
        continue;
      }
      hdda.update(ray, 1);
      if(acc.isActive(hdda.voxel())){
              t = hdda.time();
              return true;
      }
    }
    return false;
}
template <typename T>
__global__ void viewSimKernel(    nanovdb::NanoGrid<T>&                grid,
                                  nanomap::gpu::PointCloud&         pclArray,
                                  const nanomap::gpu::Sensor<float>&           devSensor,
                                  const float gridRes)
{
    using ValueT = float;
    using Vec3T = nanovdb::Vec3<ValueT>;
    using CoordT = nanovdb::Coord;
    using RayT = nanovdb::Ray<ValueT>;
    using Point = nanomap::gpu::PointCloud::Point;

    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    if (w >= devSensor.hRes() || h >= devSensor.vRes()){
        return;
    }
    float minTime;
    float maxTime;
    Vec3T rayEye;
    Vec3T rayDir;
    Vec3T defaultRay;
    devSensor.getRay(w, h, defaultRay, rayEye, rayDir, minTime, maxTime, gridRes);
    RayT  ray = RayT(rayEye, rayDir);
    auto acc = grid.getAccessor();
    CoordT ijk;
    float  t;
    float  v0;
    pclArray(w+h*devSensor.hRes()) = Point(0.0,0.0,0.0,-1.0,0.0);

    if (getActiveCrossing(ray, acc, ijk, v0, t)) {
      if(t>=minTime && t<=maxTime){
        pclArray(w+h*devSensor.hRes()) = Point(defaultRay[0]*(t*gridRes), defaultRay[1]*(t*gridRes), defaultRay[2]*(t*gridRes), 0.0, 0.0);
      }else{
        pclArray(w+h*devSensor.hRes()) = Point(defaultRay[0],defaultRay[1],0.0,-1.0,0.0);
      }
    }else{
      pclArray(w+h*devSensor.hRes()) = Point(defaultRay[0],defaultRay[1],0.0,-1.0,0.0);
    }
  }
// This is called by the host
extern "C" void generateCloud(nanomap::gpu::SensorBucket& sensorBucket, nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>&  gridHandle, cudaStream_t s0)
{
    auto        round = [](int a, int b) { return (a + b - 1) / b; };
    const dim3  threadsPerBlock(8, 8), numBlocks(round(sensorBucket.hostSensor()->hRes(), threadsPerBlock.x),
                                                  round(sensorBucket.hostSensor()->vRes(), threadsPerBlock.y));
    auto*       deviceGrid = gridHandle.deviceGrid<float>(); // note this cannot be de-referenced since it rays to a memory address on the GPU!
    auto*       devicePointCloud = sensorBucket.pclHandle().devicePointCloud(); // note this cannot be de-referenced since it rays to a memory address on the GPU!
    assert(deviceGrid && devicePointCloud);
    viewSimKernel<<<numBlocks, threadsPerBlock, 0, s0>>>(*deviceGrid, *devicePointCloud, *(sensorBucket.devSensor()), sensorBucket.getGridRes());
    return;
}

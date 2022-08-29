// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file PointCloud.h
///
/// @author Violet Walker
///
/// @brief a class for managing pointcloud information for CUDA kernels.

#ifndef NANOMAP_GPU_POINTCLOUD_H_HAS_BEEN_INCLUDED
#define NANOMAP_GPU_POINTCLOUD_H_HAS_BEEN_INCLUDED

#include <stdint.h>
#include <string>
#include <fstream>
#include <cassert>

#include <nanovdb/util/HostBuffer.h>

namespace nanomap{
  namespace gpu{

struct PointCloudData
{
    int m_Size, m_maxSize;
    PointCloudData(int maxSize)
    :m_Size(maxSize)
    ,m_maxSize(maxSize)
    {}
};

class PointCloud : private PointCloudData
{
    using DataT = PointCloudData;

public:
    struct Point
    {
        float x,y,z, norm, count;
        __hostdev__ Point(float _x, float _y, float _z, float _norm, float _count)
            : x(_x)
            , y(_y)
            , z(_z)
            , norm(_norm)
            , count(_count)
        {
        }
    };

    __hostdev__ void                         clear();
    __hostdev__ void                         updateData(int size, int point_step, unsigned char* data_pointer);
    __hostdev__ void                         updateSize(int size);
    __hostdev__ int                          maxSize() const {return DataT::m_maxSize;}
    __hostdev__ int                          size() const { return DataT::m_Size; }
    __hostdev__ inline Point& operator()(int index);
}; // PointCloud

template<typename BufferT = nanovdb::HostBuffer>
class PointCloudHandle
{
    BufferT m_Buffer;

public:
    PointCloudHandle(int maxSize);

    void updatePointCloudHandle(int size, int point_step, unsigned char* data_pointer);
    void updatePointCloudSize(int size);

    const PointCloud* pointCloud() const { return reinterpret_cast<const PointCloud*>(m_Buffer.data()); }

    PointCloud* pointCloud() { return reinterpret_cast<PointCloud*>(m_Buffer.data()); }

    template<typename U = BufferT>
    typename std::enable_if<nanovdb::BufferTraits<U>::hasDeviceDual, const PointCloud*>::type
    devicePointCloud() const { return reinterpret_cast<const PointCloud*>(m_Buffer.deviceData()); }

    template<typename U = BufferT>
    typename std::enable_if<nanovdb::BufferTraits<U>::hasDeviceDual, PointCloud*>::type
    devicePointCloud() { return reinterpret_cast<PointCloud*>(m_Buffer.deviceData()); }

    template<typename U = BufferT>
    typename std::enable_if<nanovdb::BufferTraits<U>::hasDeviceDual, void>::type
    deviceUpload(void* stream = nullptr, bool sync = true) { m_Buffer.deviceUpload(stream, sync); }

    template<typename U = BufferT>
    typename std::enable_if<nanovdb::BufferTraits<U>::hasDeviceDual, void>::type
    deviceDownload(void* stream = nullptr, bool sync = true) { m_Buffer.deviceDownload(stream, sync); }
};

template<typename BufferT>
PointCloudHandle<BufferT>::PointCloudHandle(int maxSize)
    : m_Buffer(sizeof(PointCloudData) + maxSize*sizeof(PointCloud::Point))
{
    PointCloudData data(maxSize);
    *reinterpret_cast<PointCloudData*>(m_Buffer.data()) = data;
    this->pointCloud()->clear();
}

template<typename BufferT>
void PointCloudHandle<BufferT>::updatePointCloudHandle(int size, int point_step, unsigned char* data_pointer)
{
    this->pointCloud()->updateData(size, point_step, data_pointer);

}

template<typename BufferT>
void PointCloudHandle<BufferT>::updatePointCloudSize(int size)
{
    this->pointCloud()->updateSize(size);
}

__hostdev__ inline void PointCloud::clear()
{
    Point* ptr = &(*this)(0);
    for (auto* end = ptr + PointCloudData::m_Size; ptr != end;){
      *ptr++ = Point(0.0, 0.0, 0.0, 0.0, 0.0);

    }
}

__hostdev__ inline void PointCloud::updateData(int size, int point_step, unsigned char* data_pointer)
{
    PointCloudData::m_Size = size;
    int index = 0;
    Point* ptr = &(*this)(0);
    for(auto* end = ptr + PointCloudData::m_Size; ptr != end;){
      unsigned char* byte_ptr = data_pointer + index*point_step;
      *ptr++ = Point(*(reinterpret_cast<float*>(byte_ptr+0)),
                      *(reinterpret_cast<float*>(byte_ptr+4)),
                      *(reinterpret_cast<float*>(byte_ptr+8)), 0.0, 0.0);
      index += 1;
    }
}

__hostdev__ inline void PointCloud::updateSize(int size)
{
    PointCloudData::m_Size = size;
}

inline PointCloud::Point& PointCloud::operator()(int index)
{
    assert(index < PointCloudData::m_Size);
    return *(reinterpret_cast<Point*>((uint8_t*)this + sizeof(PointCloudData)) + index);
}

} // namespace gpu
} // namespace nanovdb

#endif

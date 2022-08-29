// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file NodeWorker.h
///
/// @author Violet Walker
///
/// @brief a class for managing node information during CUDA kernel calculations.


#ifndef NANOMAP_GPU_NODEWORKER_H_HAS_BEEN_INCLUDED
#define NANOMAP_GPU_NODEWORKER_H_HAS_BEEN_INCLUDED

#include <stdint.h> // for uint8_t
#include <string> //   for std::string
#include <fstream> //  for std::ofstream
#include <cassert>
#include <nanovdb/util/HostBuffer.h>

namespace nanomap{
  namespace gpu{

struct NodeWorkerData
{

    int m_maxSize;
    int m_nodeCount;
    NodeWorkerData(int maxSize)
        : m_maxSize(maxSize)
        , m_nodeCount(0)
    {
    }
};

class NodeWorker : private NodeWorkerData
{
    using DataT = NodeWorkerData;
public:

  struct Node
  {
    int active;
    int index;
    __hostdev__ Node(int _active, int _index)
        : active(_active)
        , index(_index)
    {
    }
  };

    __hostdev__ void             clear();
    __hostdev__ int              maxSize() const { return DataT::m_maxSize; }
    __hostdev__ int&              nodeCount() {return DataT::m_nodeCount; }
    __hostdev__ inline Node& operator()(int index);
    __hostdev__ inline Node& operator()(int x, int y, int z, int xDim, int yDim, int zDim);
};

template<typename BufferT = nanovdb::HostBuffer>
class NodeWorkerHandle
{
    BufferT m_Buffer;

public:
    NodeWorkerHandle(int maxSize);

    const NodeWorker* nodeWorker() const { return reinterpret_cast<const NodeWorker*>(m_Buffer.data()); }

    NodeWorker* nodeWorker() { return reinterpret_cast<NodeWorker*>(m_Buffer.data()); }

    template<typename U = BufferT>
    typename std::enable_if<nanovdb::BufferTraits<U>::hasDeviceDual, const NodeWorker*>::type
    deviceNodeWorker() const { return reinterpret_cast<const NodeWorker*>(m_Buffer.deviceData()); }

    template<typename U = BufferT>
    typename std::enable_if<nanovdb::BufferTraits<U>::hasDeviceDual, NodeWorker*>::type
    deviceNodeWorker() { return reinterpret_cast<NodeWorker*>(m_Buffer.deviceData()); }

    template<typename U = BufferT>
    typename std::enable_if<nanovdb::BufferTraits<U>::hasDeviceDual, NodeWorker*>::type
    deviceSize() { return reinterpret_cast<NodeWorker*>(m_Buffer.size()); }

    template<typename U = BufferT>
    typename std::enable_if<nanovdb::BufferTraits<U>::hasDeviceDual, void>::type
    deviceUpload(void* stream = nullptr, bool sync = true) { m_Buffer.deviceUpload(stream, sync); }

    template<typename U = BufferT>
    typename std::enable_if<nanovdb::BufferTraits<U>::hasDeviceDual, void>::type
    deviceDownload(void* stream = nullptr, bool sync = true) { m_Buffer.deviceDownload(stream, sync); }
};

template<typename BufferT>
NodeWorkerHandle<BufferT>::NodeWorkerHandle(int maxSize)
    : m_Buffer(sizeof(NodeWorkerData) + (maxSize) * sizeof(NodeWorker::Node))
{
    NodeWorkerData data(maxSize);
    *reinterpret_cast<NodeWorkerData*>(m_Buffer.data()) = data;
    this->nodeWorker()->clear(); // clear pixels or set background
}

inline void NodeWorker::clear()
{
    NodeWorker::Node* ptr = &(*this)(0);
    int clearSize = NodeWorkerData::m_maxSize;
    for (auto* end = ptr + clearSize; ptr != end;){
      *ptr++ = NodeWorker::Node(0,0);
    }
}

inline NodeWorker::Node& NodeWorker::operator()(int index)
{
  assert(index >= 0 && index < NodeWorkerData::m_maxSize);
  return *(reinterpret_cast<NodeWorker::Node*>((uint8_t*)this + sizeof(NodeWorkerData)) + index);
}

inline NodeWorker::Node& NodeWorker::operator()(int x, int y, int z, int xDim, int yDim, int zDim)
{
    assert(x >= 0 && x < xDim);
    assert(y >= 0 && y < yDim);
    assert(z >= 0 && z < zDim);
    return *(reinterpret_cast<NodeWorker::Node*>((uint8_t*)this + sizeof(NodeWorkerData)) + z + y * zDim + x * yDim * zDim);
}

} // namespace gpu
} // namespace nanovdb

#endif

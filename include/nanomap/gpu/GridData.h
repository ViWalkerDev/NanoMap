// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file GridData.h
///
/// @author Violet Walker
///
/// @brief a griddata class for managing griddata information for CUDA kernels.



#ifndef NANOMAP_GPU_GRIDDATA_H_HAS_BEEN_INCLUDED
#define NANOMAP_GPU_GRIDDATA_H_HAS_BEEN_INCLUDED

#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include <iomanip>
#include <chrono>
#include <iostream>
#include <string>
#include <nanovdb/NanoVDB.h>
#include "nanomap/nanomap.h"
namespace nanomap{
  namespace gpu{



class GridData
{

  using ValueT = float;
public:
    __hostdev__ GridData()
    {}

    __hostdev__ void init(ValueT gridRes, int leafEdge, int maxNodeBufferSize){
      m_gridRes = gridRes;
      m_leafEdge = leafEdge;
      m_leafFace = leafEdge*leafEdge;
      m_leafVolume = leafEdge*leafEdge*leafEdge;
      m_maxNodeBufferSize = maxNodeBufferSize;
    }

    __hostdev__ void update(nanovdb::CoordBBox voxelBounds,
                            nanovdb::CoordBBox nodeBounds,
                            int nodeBufferSize){
        m_voxelBounds = voxelBounds;

        m_voxelDim = voxelBounds.dim();

        m_voxelMin = voxelBounds.min();

        m_nodeBounds = nodeBounds;

        m_nodeDim = nodeBounds.dim();

        m_nodeMin = nodeBounds.min();

        m_nodeBufferSize = nodeBufferSize;
    }

    __hostdev__ const int maxNodeBufferSize() const {return m_maxNodeBufferSize;}
    __hostdev__ const int nodeBufferSize() const {return m_nodeBufferSize;}
    __hostdev__ const nanovdb::CoordBBox nodeBounds() const {return m_nodeBounds;}
    __hostdev__ const nanovdb::Coord nodeDim() const {return m_nodeDim;}
    __hostdev__ const nanovdb::Coord nodeMin() const {return m_nodeMin;}
    __hostdev__ const nanovdb::Coord voxelDim() const {return m_voxelDim;}
    __hostdev__ const nanovdb::Coord voxelMin() const {return m_voxelMin;}
    __hostdev__ const int leafEdge() const {return m_leafEdge;}
    __hostdev__ const int leafFace() const {return m_leafFace;}
    __hostdev__ const int leafVolume()  const {return m_leafVolume;}

  protected:

    ValueT m_gridRes;
    int m_maxNodeBufferSize, m_nodeBufferSize;
    nanovdb::CoordBBox m_nodeBounds, m_voxelBounds;
    nanovdb::Coord m_nodeDim, m_nodeMin, m_voxelDim, m_voxelMin;
    int m_leafEdge, m_leafFace, m_leafVolume;

  }; // griddata
} //namespace gpu
} // namespace nanomap

#endif

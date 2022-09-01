// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file BlockWorker.cpp
///
/// @author Violet Walker
///

#include "nanomap/handler/BlockWorker.h"

namespace nanomap{
  namespace handler{
      BlockWorker::BlockWorker(int nodeEdge,
                               float occClampThres,
                               float emptyClampThres,
                               float logOddsHitThres,
                               float logOddsMissThres,
                               std::shared_ptr<AccessorT> accessor,
                               int8_t* hostVoxelBuffer,
                               int* hostNodeBuffer,
                               int hostCount)
      : _nodeEdge(nodeEdge),
        _occupiedClampingThreshold(occClampThres),
        _emptyClampingThreshold(emptyClampThres),
        _logOddsHitThreshold(logOddsHitThres),
        _hostVoxelBuffer(hostVoxelBuffer)
      {
        _voxelVolume = nodeEdge*nodeEdge*nodeEdge;
        _blocks = new std::vector<Block>();
        _leaves = new std::vector<LeafT*>();
        for(int i = 0; i < hostCount ; i++){
          _blocks->push_back(Block(openvdb::CoordBBox(openvdb::Coord(
                                          *(hostNodeBuffer+i*3),
                                            *(hostNodeBuffer+i*3+1),
                                              *(hostNodeBuffer+i*3+2)),
                                        openvdb::Coord(
                                          *(hostNodeBuffer+i*3)+nodeEdge-1,
                                            *(hostNodeBuffer+i*3+1)+nodeEdge-1,
                                              *(hostNodeBuffer+i*3+2)+nodeEdge-1)
                                          )));
          if(LeafT* target = accessor->probeLeaf((*_blocks)[i].bbox.min())){
            _leaves->push_back(target);
          }else{
            _leaves->push_back(nullptr);
          }
        }
      }

      void BlockWorker::destroyBlocks()
      {
        delete _blocks;
        _blocks = nullptr;
        delete _leaves;
        _leaves = nullptr;
      }

      void BlockWorker::processBlocks(bool serial)
      {
        if(serial){
            (*this)(tbb::blocked_range<size_t>(0,_blocks->size()));
        }else{
          tbb::parallel_for(tbb::blocked_range<size_t>(0, _blocks->size()), *this);
        }
      }


      void BlockWorker::operator()(const tbb::blocked_range<size_t> &r)
      const{
        assert(_blocks);
        assert(_leaves);
        LeafT* leaf = new LeafT();
        for(size_t m=r.begin(), end=r.end(); m!=end; ++m){
          Block& block = (*_blocks)[m];
          const openvdb::CoordBBox &bbox = block.bbox;
          if((*_leaves)[m] != nullptr){
            combineOccupiedLeafFromBuffer(leaf, (*_leaves)[m], m*_voxelVolume);
          }else{
            fillOccupiedLeafFromBuffer(leaf, m*_voxelVolume);
          }
          leaf->setOrigin(bbox.min());
          block.leaf = leaf;
          leaf = new LeafT();
        }
        delete leaf;
      }

      void BlockWorker::fillOccupiedLeafFromBuffer(LeafT* leaf, int index)
      const{
	      float value;
        auto update = [&log_hit_thres = _logOddsHitThreshold,
                       &log_miss_thres = _logOddsMissThreshold,
		                   &occ_clamp = _occupiedClampingThreshold,
                       &empty_clamp = _emptyClampingThreshold,
                       &temp_value = value]
                       (float& voxel_value, bool& active)
        {
    	    voxel_value = temp_value;
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
	      for(int x = 0; x < _nodeEdge ; x++){
          for(int y = 0; y < _nodeEdge ; y++){
            for(int z = 0; z < _nodeEdge ; z++){
              value = (float)((*(_hostVoxelBuffer + index + z + y*_nodeEdge + x*_nodeEdge*_nodeEdge)))/10;
		          if(value != 0.0){
		            leaf->modifyValueAndActiveState(openvdb::Coord(x,y,z), update);
		          }else{
		            leaf->setValueOff(openvdb::Coord(x,y,z), 0.0);
		          }
            }
          }
        }
      }

      void BlockWorker::combineOccupiedLeafFromBuffer(LeafT*           leaf,
                                                      LeafT*         target,
                                                      int             index)
      const{
        float probeValue;
        float value;
        int linearOffset;
        bool targetActive = false;
	      auto update = [&log_hit_thres = _logOddsHitThreshold,
                       &log_miss_thres = _logOddsMissThreshold,
		                   &occ_clamp = _occupiedClampingThreshold,
                       &empty_clamp = _emptyClampingThreshold,
		                   &probe_value = probeValue,
                       &temp_value = value,
		                   &target_activity = targetActive]
                       (float& voxel_value, bool& active)
        {
    	     voxel_value = probe_value + temp_value;
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

	    auto probe = [&probe_value = probeValue,
                    &target_activity = targetActive]
                    (float& voxel_value, bool& active){
	  	  voxel_value = probe_value;
		    active = target_activity;
	    };

      for(int x = 0; x < _nodeEdge ; x++){
        for(int y = 0; y < _nodeEdge ; y++){
          for(int z = 0; z < _nodeEdge ; z++){
            linearOffset =  z + y*_nodeEdge + x*_nodeEdge*_nodeEdge;
            value = (float)(*(_hostVoxelBuffer + index + linearOffset))/10;
            targetActive = target->probeValue(openvdb::Coord(x,y,z), probeValue);
            if(value != 0.0){
		          leaf->modifyValueAndActiveState(openvdb::Coord(x,y,z), update);
		        }else{
		          leaf->modifyValueAndActiveState(openvdb::Coord(x,y,z), probe);
		        }
          }
        }
      }
    }
  }
}

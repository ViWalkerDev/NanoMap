// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file OccupancyMap.h
///
/// @author Violet Walker
///
#ifndef NANOMAP_MAP_MAP_H_INCLUDED
#define NANOMAP_MAP_MAP_H_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/openvdb.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/tools/ValueTransformer.h>
#include <nanovdb/util/GridHandle.h>
namespace nanomap{
  namespace map{


    //Base Map Class, contains a single grid.
    class Map{
      using FloatGrid = openvdb::FloatGrid;
      using FloatTreeT = openvdb::FloatGrid::TreeType;
      using FloatLeafT = FloatTreeT::LeafNodeType;
      using FloatAccessorT = openvdb::tree::ValueAccessor<FloatTreeT>;

      using IntGrid = openvdb::Int32Grid;
      using IntTreeT = IntGrid::TreeType;
      using IntLeafT = IntTreeT::LeafNodeType;
      using IntAccessorT = openvdb::tree::ValueAccessor<IntTreeT>;
      public:
        Map(){}

        Map(float gridRes, float probThresHit, float probThresMiss,float mappingRes=0, float plannerRes=0, int nodeEdge=0){
            //Define GridRes
            _gridRes = gridRes;
            //Define Mapping Variables
            _logOddsMissThreshold = log(probThresMiss)-log(1-probThresMiss);
            _logOddsHitThreshold = log(probThresHit)-log(1-probThresHit);
            _occupiedClampingThreshold = log(0.97)-log(1-0.97);
            _emptyClampingThreshold = (log(0.12)-log(1-0.12));
            //Create Grids
            _occupiedGrid = openvdb::FloatGrid::create(0.0);
            // Create Linear Transforms for World to GridRes transform.
            _occupiedGrid->setTransform(openvdb::math::Transform::createLinearTransform(_gridRes));
            // Identify the grids as level set
            _occupiedGrid->setGridClass(openvdb::GRID_LEVEL_SET);
            _occupiedGrid->setName("OccupiedGrid");
            //create accessor ptr
            _occAccessor = std::make_shared<FloatAccessorT>(_occupiedGrid->getAccessor());
          }
        virtual void populateGridIndex(){}
        virtual openvdb::FloatGrid::Ptr occupiedGrid(){return _occupiedGrid;}

        virtual float occupiedClampingThreshold() {return _occupiedClampingThreshold;}
        virtual float emptyClampingThreshold() {return _emptyClampingThreshold;}
        virtual float logOddsHitThreshold() {return _logOddsHitThreshold;}
        virtual float logOddsMissThreshold() {return _logOddsMissThreshold;}
        virtual std::shared_ptr<FloatAccessorT> occAccessor(){return _occAccessor;}
        float gridRes(){return _gridRes;}

      private:
        float _gridRes;
        float _logOddsMissThreshold;
        float _logOddsHitThreshold;
        float _occupiedClampingThreshold;
        float _emptyClampingThreshold;
        FloatGrid::Ptr _occupiedGrid;
        std::shared_ptr<FloatAccessorT> _occAccessor;


    };
  }//namespace map
}//namespace nanomap
#endif

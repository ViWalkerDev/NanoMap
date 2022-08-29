// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file CaveGen.h
///
/// @author Violet Walker
///
#ifndef NANOMAP_MAPGEN_CAVEGEN_H_INCLUDED
#define NANOMAP_MAPGEN_CAVEGEN_H_INCLUDED

#include <string.h>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include <openvdb/Types.h>
#include <openvdb/math/DDA.h>
#include <openvdb/math/Ray.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/TopologyToLevelSet.h>
#include <openvdb/math/Coord.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/Grid.h>
#include <openvdb/tools/RayIntersector.h>
#include <thread>
#include <fstream>
#include <iostream>
#include <chrono>

#define LEAF_SIZE 8
namespace nanomap{
  namespace mapgen{
    class CaveGen{
      public:
        CaveGen(std::string config){
          loadConfig(config);
          srand(_seed);
        }

        void gen();
        void populateMap();
        void populateMap2D();
        int getActiveNeighbourCount(int x, int y, int z);
        int get2DNeighbourCount(int x, int y);
        void smoothMap();
        void smoothMapXY();
        void setFloorAndCeil();
        void getRegions();
        void keepLargestRegion();
        void addNoise();
        void invertAndSave();
        void Save();
        void loadConfig(std::string file_in);

      private:
        bool _use_rand_seed;
        std::string _file_out;
        int _seed, _active_max, _active_min;
        std::array<double,3> _map_size;
        double _grid_res;
        std::array<int,3> _map_dimensions;
        int _smoothing_count, _fill_percentage;
        openvdb::FloatGrid::Ptr _cave_grid = openvdb::FloatGrid::create(0.0);
        std::vector<std::set<openvdb::Coord>> _regions;
        int _region_index;
    };
  }//namespace mapgen
}//namespace gymvdb
#endif

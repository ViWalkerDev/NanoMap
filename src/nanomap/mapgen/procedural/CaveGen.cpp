// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file CaveGen.cpp
///
/// @author Violet Walker
///

#include "nanomap/mapgen/procedural/CaveGen.h"
namespace nanomap{
  namespace mapgen{

    void CaveGen::populateMap(){
      _cave_grid->setTransform(openvdb::math::Transform::createLinearTransform(_grid_res));
      _cave_grid->setGridClass(openvdb::GRID_LEVEL_SET);
      auto _cave_acc = _cave_grid->getAccessor();
      int x,y,z;
      _map_dimensions[0] = std::ceil(_map_size[0]/_grid_res);
      _map_dimensions[1] = std::ceil(_map_size[1]/_grid_res);
      _map_dimensions[2] = std::ceil(_map_size[2]/_grid_res);
      std::cout << "Map_dimensions: " << _map_dimensions[0] << ","
                                      << _map_dimensions[1] << ","
                                      << _map_dimensions[2] << std::endl;
      for(x = 0; x < 3; x++){
        if((_map_dimensions[x]%2)!= 0){
          _map_dimensions[x] = (_map_dimensions[x]+1)/2;
        }else{
          _map_dimensions[x] = _map_dimensions[x]/2;
        }
      }
      int fill;
      std::cout << "Performing Fill, Fill Percentage: "<<_fill_percentage << std::endl;
      for(x=-_map_dimensions[0]+LEAF_SIZE;x<_map_dimensions[0]-LEAF_SIZE;x+=LEAF_SIZE){
        //std::cout << "Percentage Complete: " << ((double(x)+double(_map_dimensions[0]))/(double(_map_dimensions[0]*2)))*100 << std::endl;
        for(y=-_map_dimensions[1]+LEAF_SIZE;y<_map_dimensions[1]-LEAF_SIZE;y+=LEAF_SIZE){
          for(z=-_map_dimensions[2]+LEAF_SIZE;z<_map_dimensions[2]-LEAF_SIZE;z+=LEAF_SIZE){
            fill = rand()%100;
            if(fill<_fill_percentage){
              _cave_acc.setValueOn(openvdb::Coord(x,y,z), 1.0);
            }
          }
        }
      }
          //for(z=-_map_dimensions[2];z<_map_dimensions[2]-LEAF_SIZE;z+=LEAF_SIZE){
      //openvdb::io::File("caveTestGrid.vdb").write({_cave_grid});
    }

    void CaveGen::populateMap2D(){
      _cave_grid->setTransform(openvdb::math::Transform::createLinearTransform(_grid_res));
      _cave_grid->setGridClass(openvdb::GRID_LEVEL_SET);
      auto _cave_acc = _cave_grid->getAccessor();
      int x,y,z;
      _map_dimensions[0] = std::ceil(_map_size[0]/_grid_res);
      _map_dimensions[1] = std::ceil(_map_size[1]/_grid_res);
      _map_dimensions[2] = std::ceil(_map_size[2]/_grid_res);
      std::cout << "Map_dimensions: " << _map_dimensions[0] << ","
                                      << _map_dimensions[1] << ","
                                      << _map_dimensions[2] << std::endl;
      for(x = 0; x < 3; x++){
        if((_map_dimensions[x]%2)!= 0){
          _map_dimensions[x] = (_map_dimensions[x]+1)/2;
        }else{
          _map_dimensions[x] = _map_dimensions[x]/2;
        }
      }
      int fill;
      std::cout << "Performing Fill, Fill Percentage: "<<_fill_percentage << std::endl;
      for(x=-_map_dimensions[0]+LEAF_SIZE;x<_map_dimensions[0]-LEAF_SIZE;x+=LEAF_SIZE){
        //std::cout << "Percentage Complete: " << ((double(x)+double(_map_dimensions[0]))/(double(_map_dimensions[0]*2)))*100 << std::endl;
        for(y=-_map_dimensions[1]+LEAF_SIZE;y<_map_dimensions[1]-LEAF_SIZE;y+=LEAF_SIZE){
          //
            fill = rand()%100;

            if(fill<_fill_percentage){
              for(z=-_map_dimensions[2];z<_map_dimensions[2];z+=LEAF_SIZE){
                _cave_acc.setValueOn(openvdb::Coord(x,y,z), 1.0);
              }
            }
          //}
        }
      }
          //for(z=-_map_dimensions[2];z<_map_dimensions[2]-LEAF_SIZE;z+=LEAF_SIZE){
      //openvdb::io::File("caveTestGrid.vdb").write({_cave_grid});
    }

    void CaveGen::setFloorAndCeil(){
      auto _cave_acc = _cave_grid->getAccessor();
      int x,y,z;
      for(x=-_map_dimensions[0]+LEAF_SIZE;x<_map_dimensions[0]-LEAF_SIZE;x+=LEAF_SIZE){
        //std::cout << "Percentage Complete: " << ((double(x)+double(_map_dimensions[0]))/(double(_map_dimensions[0]*2)))*100 << std::endl;
        for(y=-_map_dimensions[1]+LEAF_SIZE;y<_map_dimensions[1]-LEAF_SIZE;y+=LEAF_SIZE){
          _cave_acc.setValueOn(openvdb::Coord(x,y,-_map_dimensions[2]), 1.0);
          _cave_acc.setValueOn(openvdb::Coord(x,y,_map_dimensions[2]), 1.0);
          }
        }
    }
    void CaveGen::smoothMap(){
      openvdb::FloatGrid::Ptr _temp_grid = openvdb::FloatGrid::create(0.0);
      openvdb::FloatGrid::Accessor _temp_acc = _temp_grid->getAccessor();
      auto _cave_acc = _cave_grid->getAccessor();
      std::cout << "Starting Map Smooth" << std::endl;
      int i,x,y,z;
      int active_count;
      for(i=0;i<_smoothing_count;i++){
        std::cout << "Smooth Iteration: " << i << std::endl;
        for(x=-_map_dimensions[0]+LEAF_SIZE;x<_map_dimensions[0]-LEAF_SIZE;x+=LEAF_SIZE){
          //std::cout << "Percentage Complete: " << ((double(x)+double(_map_dimensions[0]))/(double(_map_dimensions[0]*2)))*100 << std::endl;
          for(y=-_map_dimensions[1]+LEAF_SIZE;y<_map_dimensions[1]-LEAF_SIZE;y+=LEAF_SIZE){
            for(z=-_map_dimensions[2];z<_map_dimensions[2];z+=LEAF_SIZE){
              //std::cout << active_count;
              active_count = getActiveNeighbourCount(x,y,z);
              //std::cout << active_count<<std::endl;
              if(active_count>_active_max){
                _temp_acc.setValueOn(openvdb::Coord(x,y,z), 1.0);
              }else if(active_count<_active_min){
                _temp_acc.setValueOff(openvdb::Coord(x,y,z),0.0);
              }else{
                if(_cave_acc.isValueOn(openvdb::Coord(x, y,z))){
                  _temp_acc.setValueOn(openvdb::Coord(x,y,z), 1.0);
                }else{
                  _temp_acc.setValueOff(openvdb::Coord(x,y,z),0.0);
                }
              }
            }
          }
        }
        _cave_grid = _temp_grid->deepCopy();
      }
    }
    void CaveGen::smoothMapXY(){
      openvdb::FloatGrid::Ptr _temp_grid = openvdb::FloatGrid::create(0.0);
      openvdb::FloatGrid::Accessor _temp_acc = _temp_grid->getAccessor();
      openvdb::FloatGrid::Accessor _cave_acc = _cave_grid->getAccessor();;
      std::cout << "Starting Map Smooth" << std::endl;
      int i,x,y,z;
      int active_count=0;
      for(i=0;i<_smoothing_count;i++){
        _cave_acc = _cave_grid->getAccessor();
        std::cout << "Smooth Iteration: " << i << std::endl;
        for(x=-_map_dimensions[0]+LEAF_SIZE;x<_map_dimensions[0]-LEAF_SIZE;x+=LEAF_SIZE){
          //std::cout << "Percentage Complete: " << ((double(x)+double(_map_dimensions[0]))/(double(_map_dimensions[0]*2)))*100 << std::endl;
          for(y=-_map_dimensions[1]+LEAF_SIZE;y<_map_dimensions[1]-LEAF_SIZE;y+=LEAF_SIZE){
            //for(z=-_map_dimensions[2]+LEAF_SIZE;z<_map_dimensions[2]-LEAF_SIZE;z+=LEAF_SIZE){
              //std::cout << active_count;

              active_count = get2DNeighbourCount(x,y);
              std::cout << active_count<<std::endl;
              if(active_count>_active_max){
                //std::cout << "c" << std::endl;
                for(z=-_map_dimensions[2];z<_map_dimensions[2];z+=LEAF_SIZE){
                  _temp_acc.setValueOn(openvdb::Coord(x,y,z), 1.0);
                }
              }else if(active_count<_active_min){
                //std::cout << "f" << std::endl;
                for(z=-_map_dimensions[2];z<_map_dimensions[2];z+=LEAF_SIZE){
                  _temp_acc.setValueOff(openvdb::Coord(x,y,z), 1.0);
                }
              }else{
                //std::cout << "l" << std::endl;
                if(_cave_acc.isValueOn(openvdb::Coord(x, y,-_map_dimensions[2]))){
                  //std::cout << "c" << std::endl;
                  for(z=-_map_dimensions[2];z<_map_dimensions[2];z+=LEAF_SIZE){
                    _temp_acc.setValueOn(openvdb::Coord(x,y,z), 1.0);
                  }
                }else{
                  //std::cout << "l" << std::endl;
                  for(z=-_map_dimensions[2];z<_map_dimensions[2];z+=LEAF_SIZE){
                    _temp_acc.setValueOff(openvdb::Coord(x,y,z), 1.0);
                  }
                }
              }
            //}
          }
        }
        _cave_grid = _temp_grid->deepCopy();
      }
    }

    void CaveGen::getRegions(){
      std::cout << "Identifying Regions" << std::endl;
      std::vector<openvdb::Coord> current_region_queue;
      std::set<openvdb::Coord> current_region_coords;
      std::set<openvdb::Coord> all_region_coords;
      openvdb::Coord coord;
      openvdb::Coord current_coord;
      std::set<openvdb::Coord>::iterator it;
      auto _cave_acc = _cave_grid->getAccessor();
      int x,y,z;
      for (openvdb::FloatGrid::ValueOnCIter iter = _cave_grid->cbeginValueOn(); iter.test(); ++iter) {
        coord = iter.getCoord();
        it = all_region_coords.find(coord);
        if(it == all_region_coords.end()){
          current_region_coords.clear();
          //IF ALL REGION COORDS DOESN'T CONTAIN COORD, ADD TO QUEUE
          current_region_coords.insert(coord);
          all_region_coords.insert(coord);
          current_region_queue.push_back(coord);
          while(current_region_queue.size()>0){
            current_coord = current_region_queue.back();
            current_region_queue.pop_back();
            for(x=-LEAF_SIZE; x<=LEAF_SIZE; x+=LEAF_SIZE*2){
              coord.reset(current_coord.x()+x, current_coord.y(), current_coord.z());
              if(_cave_acc.isValueOn(coord)){
                it = current_region_coords.find(coord);
                if(it == current_region_coords.end()){
                  current_region_queue.push_back(coord);
                  current_region_coords.insert(coord);
                  all_region_coords.insert(coord);
                }
              }
            }
            for(y=-LEAF_SIZE; y<=LEAF_SIZE; y+=LEAF_SIZE*2){
              coord.reset(current_coord.x(), current_coord.y()+y, current_coord.z());
              if(_cave_acc.isValueOn(coord)){
                it = current_region_coords.find(coord);
                if(it == current_region_coords.end()){
                  current_region_queue.push_back(coord);
                  current_region_coords.insert(coord);
                  all_region_coords.insert(coord);
                }
              }
            }
            for(z=-LEAF_SIZE; z<=LEAF_SIZE; z+=LEAF_SIZE*2){
              coord.reset(current_coord.x(), current_coord.y(), current_coord.z()+z);
              if(_cave_acc.isValueOn(coord)){
                it = current_region_coords.find(coord);
                if(it == current_region_coords.end()){
                  current_region_queue.push_back(coord);
                  current_region_coords.insert(coord);
                  all_region_coords.insert(coord);
                }
              }
            }
          }
        }
          _regions.push_back(current_region_coords);
      }
      int max_size=0;
      _region_index = 0;
      for(auto it = _regions.begin(); it != _regions.end(); it++){
          if(max_size<it->size()){
            max_size = it->size();
            _region_index = it-_regions.begin();
          }
      }
    }

    void CaveGen::keepLargestRegion(){
      std::cout << "Extracting Largest Region" << std::endl;
      openvdb::FloatGrid::Ptr _temp_grid = openvdb::FloatGrid::create(0.0);
      openvdb::FloatGrid::Accessor _temp_acc = _temp_grid->getAccessor();
      _temp_grid->setTransform(openvdb::math::Transform::createLinearTransform(_grid_res));
      _temp_grid->setGridClass(openvdb::GRID_LEVEL_SET);
      for(auto it = _regions[_region_index].begin(); it!= _regions[_region_index].end(); it++){
        _temp_acc.setValueOn(*it,1.0);
      }
      _cave_grid = _temp_grid->deepCopy();
    }

    void CaveGen::addNoise(){

    }
    int CaveGen::getActiveNeighbourCount(int coord_x, int coord_y, int coord_z){
      int x,y,z;
      int active_neighbour_count=0;
      auto _cave_acc = _cave_grid -> getAccessor();
      for(x=-LEAF_SIZE;x<=LEAF_SIZE;x+=LEAF_SIZE){
        for(y=-LEAF_SIZE;y<=LEAF_SIZE;y+=LEAF_SIZE){
          for(z=-LEAF_SIZE;z<=LEAF_SIZE;z+=LEAF_SIZE){
            if(!(x==0 && y==0 && z==0)){
              if(_cave_acc.isValueOn(openvdb::Coord(coord_x+x, coord_y+y,coord_z+z))){
                active_neighbour_count+=1;
              }
            }
          }
        }
      }
      return active_neighbour_count;
    }

    int CaveGen::get2DNeighbourCount(int coord_x, int coord_y){
      int x,y,z;
      int active_neighbour_count=0;
      auto _cave_acc = _cave_grid -> getAccessor();
      for(x=-LEAF_SIZE;x<=LEAF_SIZE;x+=LEAF_SIZE){
        for(y=-LEAF_SIZE;y<=LEAF_SIZE;y+=LEAF_SIZE){
            if(!(x==0 && y==0)){
              std::cout << coord_x+x << std::endl;
              if(_cave_acc.isValueOn(openvdb::Coord(coord_x+x, coord_y+y,-_map_dimensions[2]))){
                active_neighbour_count+=1;
                //std::cout << active_neighbour_count << std::endl;
              }
            }
        }
      }
      return active_neighbour_count;
    }

    void CaveGen::invertAndSave(){
      std::cout << "Starting Invert and Save" << std::endl;
      int x,y,z;
      int sum;
      auto _cave_acc = _cave_grid->getAccessor();
      for (openvdb::FloatGrid::ValueOnIter iter = _cave_grid->beginValueOn(); iter.test(); ++iter) {
          sum+=1;
          _cave_acc.addTile(1, openvdb::Coord(iter.getCoord()), 1.0 ,true);
      }
      std::cout << "Number of nodes: "<< sum << std::endl;
      openvdb::FloatGrid::Ptr _save_grid;
      _save_grid = openvdb::FloatGrid::create(0.0);
      //_save_grid->setTransform(openvdb::math::Transform::createLinearTransform(_grid_res));
      _save_grid->setGridClass(openvdb::GRID_LEVEL_SET);

      auto tree = _cave_acc.getTree();
      std::cout << "Voxelizing Active Tiles" << std::endl;
      tree->voxelizeActiveTiles();
      //std::cout << "Eroding..." << std::endl;
      //openvdb::tools::erodeActiveValues(*tree, 2);
      //openvdb::tools::dilateActiveValues(*tree, 2, openvdb::tools::NN_FACE,openvdb::tools::EXPAND_TILES);
      std::cout << "Shelling..." << std::endl;
      _save_grid = openvdb::tools::topologyToLevelSet(*_cave_grid);
      std::cout << "Saving..." << std::endl;
      _save_grid->setName("grid");
      openvdb::io::File(_file_out).write({_save_grid});
    }

    void CaveGen::Save(){
      int x,y,z;
      auto _cave_acc = _cave_grid->getAccessor();
      int sum;
      for (openvdb::FloatGrid::ValueOnIter iter = _cave_grid->beginValueOn(); iter.test(); ++iter) {
          sum+=1;
          _cave_acc.addTile(1, openvdb::Coord(iter.getCoord()), 1.0 ,true);
      }
      _cave_grid->tree().voxelizeActiveTiles();
      openvdb::io::File(_file_out).write({_cave_grid});
    }

    void CaveGen::gen(){
      populateMap();
      smoothMap();
      getRegions();
      invertAndSave();
    }


    void CaveGen::loadConfig(std::string file_in){
      std::ifstream *input = new std::ifstream(file_in.c_str(), std::ios::in | std::ios::binary);
      std::string line;
      bool done = false;
      std::string seed;
      int random_seed;
      *input >> line;
      if (line.compare("#cavegenconfig") != 0) {
        std::cout << "Error: first line reads [" << line << "] instead of [#cavegenconfig]" << std::endl;
        delete input;
        return;
      }
      while(input->good() && !done) {
        *input >> line;
        if (line.compare("file_out") == 0){
          *input >> _file_out;
        }
        else if (line.compare("map_size") == 0){
          *input >> _map_size[0] >> _map_size[1] >> _map_size[2];
        }
        else if (line.compare("grid_res")==0){
          *input >> _grid_res;
        }
        else if (line.compare("fill_percentage")==0){
          *input >> _fill_percentage;
        }
        else if (line.compare("smoothing_count")==0){
          *input >> _smoothing_count;
        }
        else if (line.compare("active")==0){
          *input >> _active_max >> _active_min;
        }
        else if (line.compare("random_seed")==0){
          *input >> random_seed;
          if(random_seed ==1){
            _use_rand_seed = true;
          }else{
            _use_rand_seed = false;
          }
        }
        else if (line.compare("seed")==0){
          *input >> seed;
          done = true;
        }
        if(!_use_rand_seed){
          std::hash<std::string> string_hash;
          _seed = string_hash(seed);
        }else{
          std::hash<std::time_t> time_hash;
          _seed = time_hash(time(0));
        }

      }

      std::cout << "input_read" << std::endl;
      input->close();
    }
  }//namespace mapgen
}//namespace nanomap

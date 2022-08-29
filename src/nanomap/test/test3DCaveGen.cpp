// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file test3DCaveGen.cpp
///
/// @author Violet Walker
///

#include "nanomap/mapgen/procedural/CaveGen.h"

int main(int argc, char **argv){
  std::string config_file;
  config_file = argv[1];
  nanomap::mapgen::CaveGen generator(config_file);
  generator.populateMap();
  generator.smoothMap();
  generator.setFloorAndCeil();
  generator.getRegions();
  generator.keepLargestRegion();
  generator.invertAndSave();
}

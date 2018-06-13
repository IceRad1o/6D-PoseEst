#pragma once

#include "types.h"

/** Read several custom data formats. */

namespace jp
{

    void readData(const std::string dFile, jp::img_depth_t& image);
    
    void readData(const std::string bgrFile, jp::img_bgr_t& image);
    
    void readData(const std::string bgrFile, const std::string dFile, jp::img_bgrd_t& image);
  
    void readData(const std::string coordFile, jp::img_coord_t& image);
    
    bool readData(const std::string infoFile, jp::info_t& info);
}
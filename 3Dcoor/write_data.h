#pragma once

#include "types.h"



namespace jp
{
    /**
    存储一个深度图片信息
    */
    void writeData(const std::string dFile, jp::img_depth_t& image);

    /**
    background file
    */
    void writeData(const std::string bgrFile, jp::img_bgr_t& image);

    
    void writeData(const std::string bgrFile, const std::string dFile, jp::img_bgrd_t& image);

    /**
    存储一个coordfile
    */
    void writeData(const std::string coordFile, jp::img_coord_t& image); 

    /**
    存储一个Info文件
     */
    void writeData(const std::string infoFile, jp::info_t& info);
}
#pragma once

#include "types.h"



namespace jp
{
    /**
    �洢һ�����ͼƬ��Ϣ
    */
    void writeData(const std::string dFile, jp::img_depth_t& image);

    /**
    background file
    */
    void writeData(const std::string bgrFile, jp::img_bgr_t& image);

    
    void writeData(const std::string bgrFile, const std::string dFile, jp::img_bgrd_t& image);

    /**
    �洢һ��coordfile
    */
    void writeData(const std::string coordFile, jp::img_coord_t& image); 

    /**
    �洢һ��Info�ļ�
     */
    void writeData(const std::string infoFile, jp::info_t& info);
}
#pragma once

#include "../core/types.h"



struct sample_t 
{
    int x, y; // position of the center pixel
    float scale; // patch scale factor 

    sample_t() : x(0), y(0), scale(1) {}
    sample_t(int x, int y) : x(x), y(y), scale(1) {}
    sample_t(int x, int y, float scale) : x(x), y(y), scale(scale) {}
};


void samplePixels(const jp::img_id_t& segmentation, const jp::info_t& info, std::vector<sample_t>& samples, const jp::img_depth_t* depth);


void samplePixels(const jp::img_stat_t& prob, const jp::info_t& info, std::vector<sample_t>& samples, const jp::img_depth_t* depth);
#pragma once

#include "types.h"
#include "thread_rand.h"
#include "properties.h"
#include "util.h"
#include "training_samples.h"

namespace jp
{

    struct FeaturePoints
    {
	FeaturePoints()
	{
	  xc = yc = x1 = y1 = x2 = y2 = 0;
	}
      
	int xc, yc; // 特征像素点中心坐标
	int x1, y1; // 特征向量
	int x2, y2; 
    };
    
  
    inline FeaturePoints getFeaturePoints(
	int x, int y, int off1_x, int off1_y, int off2_x, int off2_y, 
	float scale, int width, int height)
    {
	FeaturePoints featurePoints;
	GlobalProperties* gp = GlobalProperties::getInstance();      
	
	// ensure position is within image border
	x = clamp(x, 0, width - 1);
	y = clamp(y, 0, height - 1);
	
	featurePoints.xc = x;
	featurePoints.yc = y;


	float x1 = off1_x * scale;
	float y1 = off1_y * scale;
	float x2 = off2_x * scale;
	float y2 = off2_y * scale;	
	

	if(gp->fP.training && !gp->rotations.empty())
	{
	
	    int r = irand(0, gp->rotations.size());
	  
	    featurePoints.x1 = x + (gp->rotations[r].at<double>(0, 0) * x1 + gp->rotations[r].at<double>(0, 1) * y1 + gp->rotations[r].at<double>(0, 2));
	    featurePoints.y1 = y + (gp->rotations[r].at<double>(1, 0) * x1 + gp->rotations[r].at<double>(1, 1) * y1 + gp->rotations[r].at<double>(1, 2));
	    featurePoints.x2 = x + (gp->rotations[r].at<double>(0, 0) * x2 + gp->rotations[r].at<double>(0, 1) * y2 + gp->rotations[r].at<double>(0, 2));
	    featurePoints.y2 = y + (gp->rotations[r].at<double>(1, 0) * x2 + gp->rotations[r].at<double>(1, 1) * y2 + gp->rotations[r].at<double>(1, 2));
	}
	else
	{
	    featurePoints.x1 = x + x1;
	    featurePoints.y1 = y + y1;
	    featurePoints.x2 = x + x2;
	    featurePoints.y2 = y + y2;
	}
	
	// ensure offsets are within image border
	featurePoints.x1 = clamp(featurePoints.x1, 0, width - 1);
	featurePoints.y1 = clamp(featurePoints.y1, 0, height - 1);
	featurePoints.x2 = clamp(featurePoints.x2, 0, width - 1);
	featurePoints.y2 = clamp(featurePoints.y2, 0, height - 1);
	
	return featurePoints;
    }
    
    
    inline FeaturePoints getFeaturePoints(
	int x, int y, int off1_x, int off1_y, 
	float scale, int width, int height)
    {
	FeaturePoints featurePoints;
	GlobalProperties* gp = GlobalProperties::getInstance();      
	
	// ensure position is within image border
	x = clamp(x, 0, width - 1);
	y = clamp(y, 0, height - 1);
	
	featurePoints.xc = x;
	featurePoints.yc = y;


	float x1 = off1_x * scale;
	float y1 = off1_y * scale;
	

	if(gp->fP.training && !gp->rotations.empty())
	{

	    int r = irand(0, gp->rotations.size());
	    
	    featurePoints.x1 = x + (gp->rotations[r].at<double>(0, 0) * x1 + gp->rotations[r].at<double>(0, 1) * y1 + gp->rotations[r].at<double>(0, 2));
	    featurePoints.y1 = y + (gp->rotations[r].at<double>(1, 0) * x1 + gp->rotations[r].at<double>(1, 1) * y1 + gp->rotations[r].at<double>(1, 2));
	}
	else
	{
	    featurePoints.x1 = x + x1;
	    featurePoints.y1 = y + y1;
	}
	
	// image board..
	featurePoints.x1 = clamp(featurePoints.x1, 0, width - 1);
	featurePoints.y1 = clamp(featurePoints.y1, 0, height - 1);
	
	return featurePoints;
    }    

}

template<typename TFeatureSampler>
typename TFeatureSampler::feature_t sampleFromRandomPixel(
    const jp::img_data_t& data,
    const TFeatureSampler& sampler)
{
    
    std::vector<sample_t> objPixels;
    objPixels.reserve(data.seg.cols * data.seg.rows);

    for(unsigned y = 0; y < data.seg.rows; y++)
    for(unsigned x = 0; x < data.seg.cols; x++)
    {
	if(data.seg(y, x)) objPixels.push_back(sample_t(x, y));
    }

    
    int pixelIdx = irand(0, objPixels.size());

   
    typename TFeatureSampler::feature_t feat = sampler.sampleFeature();
    feat.setThreshold(feat.computeResponse(objPixels[pixelIdx].x, objPixels[pixelIdx].y, objPixels[pixelIdx].scale, data));

    return feat;
}


template<typename TFeatureSampler>
std::vector<typename TFeatureSampler::feature_t> sampleFromRandomPixels(
    const std::vector<jp::img_data_t>& data,
    unsigned int count, const TFeatureSampler& sampler)
{
    std::vector<size_t> imageIdx(count);
    for (unsigned int i = 0; i < count; ++i)
	imageIdx[i] = irand(0, data.size());
    std::sort(imageIdx.begin(), imageIdx.end());

    // Sample feature tests
    std::vector<typename TFeatureSampler::feature_t> rv(count);
    for (unsigned int i = 0; i < count; ++i)
	rv[i] = sampleFromRandomPixel(data[imageIdx[i]], sampler);

    return rv;
}
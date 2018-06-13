#pragma once

#include "types.h"


jp::coord3_t l2Mean(const std::vector<cv::Vec3f>& objCoords)
{
    cv::Vec3f mean(0, 0, 0);
    for(unsigned i = 0; i < objCoords.size(); i++)
	mean += objCoords[i];
    mean /= (float) objCoords.size();
    
    return jp::coord3_t(mean[0], mean[1], mean[2]);
}


jp::coord3_t l1Mean(const std::vector<cv::Vec3f>& objCoords)
{
    cv::Vec3f mean = l2Mean(objCoords);
    int it = 0, maxIt = 10; // will stop after maxIt iterations
    float delta, minDelta = 0.001; // will stop when change after last iteration is smaller than minDelta
    
    do
    {
	cv::Vec3f T1(0, 0, 0);
	cv::Vec3f newMean;
	float weightSum = 0.f;
	bool setHit = false;
	float supportInSet = 0;
      
	for(unsigned i = 0; i < objCoords.size(); i++)
	{
	    float dist = cv::norm(objCoords[i] - mean); // check whether the l1 estimate coincides with on of the set points
	    if(dist < EPS)
	    {
		setHit = true;
		supportInSet ++;
		continue;
	    }
	    
	    T1 += objCoords[i] / dist;
	    weightSum += 1 / dist;
	}
	T1 /= weightSum;
	
	if(!setHit)
	{
	    newMean = T1;
	}
	else // point in the set was hit by the estimate, calculate estimate without this point
	{
	    cv::Vec3f R(0, 0, 0);
	    
	    for(unsigned i = 0; i < objCoords.size(); i++)
	    {
		float dist = cv::norm(objCoords[i] - mean);
		if(dist < EPS)
		    continue;

		R += (objCoords[i] - mean) / dist;
	    }
	    
	    float gamma = std::min(1.0, supportInSet / cv::norm(R));
	    newMean = (1-gamma) * T1 + gamma * mean;
	}
	
	delta = cv::norm(mean - newMean);	
	it++;
	
	mean = newMean;
    }
    while((it < maxIt) && (delta > minDelta));
    
    return jp::coord3_t(mean[0], mean[1], mean[2]);
}

jp::coord3_t coordFilter(
    int cX, int cY, 
    jp::id_t objID,
    const jp::img_label_t& objProb, 
    const std::vector<jp::RegressionTree<jp::feature_t>>& forest, 
    const std::vector<jp::img_leaf_t>& leafImgs)
{
    int k = 3; // Size of the pixel neighbourhood.
  
    // create interval of the local neighbourhood (respecting image borders)
    int minX = std::max(0, cX - (k / 2));
    int maxX = std::min(objProb.cols-1, cX + (k / 2));
    int minY = std::max(0, cY - (k / 2));
    int maxY = std::min(objProb.rows-1, cY + (k / 2));
    
    std::vector<cv::Vec3f> objCoords;
    objCoords.reserve(k * k * forest.size());
    
    // iterate over neighbourhood and collect object coordinates (used to calculate the geometric median)
    for(int x = minX; x <= maxX; x++)
    for(int y = minY; y <= maxY; y++)
    {
	if(objProb(y, x) == 0) continue; // ignore background pixels
	for(int t = 0; t < forest.size(); t++)
	{
	    const std::vector<jp::mode_t>* modes = forest[t].getModes(leafImgs[t](y, x), objID);
	    for(int m = 0; m < 1; m++)
	    {
		jp::coord3_t curM = modes->at(m).mean;
		if(curM[0] == 0 && curM[1] == 0 && curM[2] == 0) continue; // ignore empty predictions
		objCoords.push_back(cv::Vec3f(curM[0], curM[1], curM[2]));
	    }
	}
    }
    
    if(objCoords.empty()) 
	return jp::coord3_t(0, 0, 0);
    return l1Mean(objCoords); // calculate the geometric median
}


void computeAutoContextChannels(
    const std::vector<jp::RegressionTree<jp::feature_t>>& rForest,
    const std::vector<jp::img_leaf_t>& leafImgs,
    const std::vector<jp::img_stat_t>& rProbabilities, 
    jp::img_data_t& inputData)
{
    GlobalProperties* gp = GlobalProperties::getInstance();
  
    int imgWidth = inputData.seg.cols;
    int imgHeight = inputData.seg.rows;
	
    int acSubSample = gp->fP.acSubsample; // auto-context feature channels can be stored sub-sampled to save memory
    int acImgWidth = imgWidth / acSubSample;
    int acImgHeight = imgHeight / acSubSample;	
  
    // object class feature channel are median smoothed object probability maps
    #pragma omp parallel for
    for(unsigned o = 0; o < rProbabilities.size(); o++)
    {
	jp::img_label_t dProb(acImgHeight, acImgWidth);
		
	for(unsigned x=0; x < acImgWidth; x++)
	for(unsigned y=0; y < acImgHeight; y++)
	    dProb(y, x) = rProbabilities[o](y*acSubSample, x*acSubSample) * 255;
	
	cv::medianBlur(dProb, inputData.labelData[o], 5);
    }
    
    // object coordinate feature channel are robustly (l1-)smoothed object coordinate predictions
    std::vector<jp::img_leaf_t> subLeafImgs;
    for(unsigned tIdx = 0; tIdx < leafImgs.size(); tIdx++)
	subLeafImgs.push_back(jp::img_leaf_t(acImgHeight, acImgWidth));
    
    // sub sample the leaf images
    #pragma omp parallel for
    for(unsigned x=0; x < acImgWidth; x++)
    for(unsigned y=0; y < acImgHeight; y++)
    for(unsigned tIdx = 0; tIdx < leafImgs.size(); tIdx++)
	subLeafImgs[tIdx](y, x) = leafImgs[tIdx](y*acSubSample, x*acSubSample);
    
    std::vector<jp::img_coord_t> coordPredictions;
    for(unsigned o = 0; o < gp->fP.objectCount; o++)
	coordPredictions.push_back(jp::img_coord_t::zeros(acImgHeight, acImgWidth));		
	
    // create smoothed object coordinate predictions
    #pragma omp parallel for
    for(unsigned x=0; x < acImgWidth; x++)
    for(unsigned y=0; y < acImgHeight; y++)
    for(int objID = 1; objID < gp->fP.objectCount+1; objID++)
	coordPredictions[objID-1](y, x) = coordFilter(x, y, objID, inputData.labelData[objID-1], rForest, subLeafImgs);		
    
    for(unsigned o = 0; o < coordPredictions.size(); o++)
	inputData.coordData[o] = coordPredictions[o];  
}
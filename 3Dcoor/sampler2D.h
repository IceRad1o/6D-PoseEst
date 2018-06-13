#pragma once

#include "types.h"
#include "thread_rand.h"
#include <array>

class Sampler2D
{
public:
   
    Sampler2D(const jp::img_stat_t& probs)
    {
	cv::integral(probs, integral);
    }

    
    cv::Point2f drawInRect(const cv::Rect& bb2D)
    {
	int minX = bb2D.tl().x;
	int maxX = bb2D.br().x - 1;
	int minY = bb2D.tl().y;
	int maxY = bb2D.br().y - 1;
      
	// choose the accumulated weight of the pixels to sample 
	double randNum = drand(0, 1) * getSum(minX, minY, maxX, maxY);
	
	// search for the pixel that statisfies the accumulated weight
	return drawInRect(minX, minY, maxX, maxY, randNum);
    }
    
public:
  
  
    inline double getSum(int minX, int minY, int maxX, int maxY)
    {
	double sum = integral(maxY + 1, maxX + 1);
	if(minX > 0) sum -= integral(maxY + 1, minX);
	if(minY > 0) sum -= integral(minY, maxX + 1);
	if(minX > 0 && minY > 0) sum += integral(minY, minX);
	return sum;
    }
    
    cv::Point2f drawInRect(int minX, int minY, int maxX, int maxY, double randNum)
    {
	double halfInt;
	
	// first search in X direction
	if(maxX - minX > 0)
	{
	    // binary search, does the pixel lie in the left or right half of the search window?
	    halfInt = getSum(minX, minY, (minX + maxX) / 2, maxY);
	    
	    if(randNum > halfInt) 
		return drawInRect((minX + maxX) / 2 + 1, minY, maxX, maxY, randNum - halfInt);
	    else 
		return drawInRect(minX, minY, (minX + maxX) / 2, maxY, randNum);
	}

	// search in Y direction
	if(maxY - minY > 0)
	{
	    // binary search, does the pixel lie in the upper or lower half of the search window?
	    halfInt = getSum(minX, minY, maxX, (minY + maxY) / 2);
	    
	    if(randNum > halfInt) 
		return drawInRect(minX, (minY + maxY) / 2 + 1, maxX, maxY, randNum - halfInt);
	    else 
		return drawInRect(minX, minY, minX, (minY + maxY) / 2, randNum);
	}

	return cv::Point2f(maxX, maxY);
    }    
  
    cv::Mat_<double> integral; // integral image (map of accumulated weights) used in binary search
};

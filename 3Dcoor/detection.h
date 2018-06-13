#pragma once

#include "types.h"


inline std::vector<cv::Point3f> getBB3D(const cv::Vec<float, 3>& extent)
{
    std::vector<cv::Point3f> bb;
    
    float xHalf = extent[0] * 500;
    float yHalf = extent[1] * 500;
    float zHalf = extent[2] * 500;
    
    bb.push_back(cv::Point3f(xHalf, yHalf, zHalf));
    bb.push_back(cv::Point3f(-xHalf, yHalf, zHalf));
    bb.push_back(cv::Point3f(xHalf, -yHalf, zHalf));
    bb.push_back(cv::Point3f(-xHalf, -yHalf, zHalf));
    
    bb.push_back(cv::Point3f(xHalf, yHalf, -zHalf));
    bb.push_back(cv::Point3f(-xHalf, yHalf, -zHalf));
    bb.push_back(cv::Point3f(xHalf, -yHalf, -zHalf));
    bb.push_back(cv::Point3f(-xHalf, -yHalf, -zHalf));
    
    return bb;
} 


inline cv::Rect getBB2D(
  int imageWidth, int imageHeight,
  const std::vector<cv::Point3f>& bb3D,
  const cv::Mat& camMat,
  const jp::cv_trans_t& trans)
{
    GlobalProperties* gp = GlobalProperties::getInstance();
    
    if(gp->fP.fullScreenObject) 
	return cv::Rect(0, 0, gp->fP.imageWidth, gp->fP.imageHeight);
    
  
    std::vector<cv::Point2f> bb2D;
    cv::projectPoints(bb3D, trans.first, trans.second, camMat, cv::Mat(), bb2D);
    
 
    int minX = imageWidth - 1;
    int maxX = 0;
    int minY = imageHeight - 1;
    int maxY = 0;
    
    for(unsigned j = 0; j < bb2D.size(); j++)
    {
	minX = std::min((float) minX, bb2D[j].x);
	minY = std::min((float) minY, bb2D[j].y);
	maxX = std::max((float) maxX, bb2D[j].x);
	maxY = std::max((float) maxY, bb2D[j].y);
    }
    
    minX = clamp(minX, 0, imageWidth - 1);
    maxX = clamp(maxX, 0, imageWidth - 1);
    minY = clamp(minY, 0, imageHeight - 1);
    maxY = clamp(maxY, 0, imageHeight - 1);
    
    return cv::Rect(minX, minY, (maxX - minX + 1), (maxY - minY + 1));
}


inline float getIoU(const cv::Rect& bb1, const cv::Rect bb2)
{
    cv::Rect intersection = bb1 & bb2;
    return (intersection.area() / (float) (bb1.area() + bb2.area() - intersection.area()));
}
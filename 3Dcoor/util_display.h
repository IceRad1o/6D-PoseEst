#pragma once

#include "types.h"
#include "properties.h"
#include "regression_tree.h"
#include "combined_feature.h"

#include "Hypothesis.h"
/*可视化结果*/


cv::Mat convertForDisplay(const jp::img_coord_t& img, jp::info_t info)
{
    cv::Mat result(img.size(), CV_8UC3);
	
    for(int x = 0; x < img.cols; x++)
    for(int y = 0; y < img.rows; y++)
    for(int channel = 0; channel < 3; channel++)
    {
	float maxExtent = info.extent(channel) * 1000.f; // in meters
	int coord = (int) (img(y, x)(channel) + maxExtent / 2.f); // shift zero point so all values are positive
	result.at<cv::Vec3b>(y, x)[channel] = (uchar) ((coord / maxExtent) * 255); // rescale to RGB range
    }
    
    return result;
}

/**
  Convert a depth image to a grayscale image.
 */
cv::Mat convertForDisplay(const jp::img_depth_t& img)
{
    jp::depth_t minDepth = 10000;
    jp::depth_t maxDepth = 0;
    
    for(int x = 0; x < img.cols; x++)
    for(int y = 0; y < img.rows; y++)
    {
	maxDepth = std::max(maxDepth, img(y, x));
	if(img(y, x) > 0)
	    minDepth = std::min(minDepth, img(y, x));
    }
    
    cv::Mat result(img.size(), CV_8U);

    for(int x = 0; x < img.cols; x++)
    for(int y = 0; y < img.rows; y++)
    {
	if(img(y, x) == 0)
	    result.at<uchar>(y, x) = 0; 
	else
	    result.at<uchar>(y, x) = 255 - (uchar) (((img(y, x) - minDepth) / (float) (maxDepth - minDepth)) * 255);
    }

    return result;
}

void drawBB(jp::img_bgr_t& img, const jp::cv_trans_t& trans, const std::vector<cv::Point3f>& bb3D, const cv::Scalar& color)
{
    std::vector<cv::Point2f> bb2D;
    
    cv::Mat_<float> camMat = GlobalProperties::getInstance()->getCamMat();
    cv::projectPoints(bb3D, trans.first, trans.second, camMat, cv::Mat(), bb2D);

    int lineW = 2; 
    int lineType = CV_AA; 
    
  
    cv::line(img, bb2D[0], bb2D[1], cv::Scalar(0, 0, 0), lineW+2, lineType);
    cv::line(img, bb2D[1], bb2D[3], cv::Scalar(0, 0, 0), lineW+2, lineType);
    cv::line(img, bb2D[3], bb2D[2], cv::Scalar(0, 0, 0), lineW+2, lineType);
    cv::line(img, bb2D[2], bb2D[0], cv::Scalar(0, 0, 0), lineW+2, lineType);

    cv::line(img, bb2D[2], bb2D[6], cv::Scalar(0, 0, 0), lineW+2, lineType);
    cv::line(img, bb2D[0], bb2D[4], cv::Scalar(0, 0, 0), lineW+2, lineType);    
    cv::line(img, bb2D[1], bb2D[5], cv::Scalar(0, 0, 0), lineW+2, lineType);
    cv::line(img, bb2D[3], bb2D[7], cv::Scalar(0, 0, 0), lineW+2, lineType);
    
    cv::line(img, bb2D[4], bb2D[5], cv::Scalar(0, 0, 0), lineW+2, lineType);
    cv::line(img, bb2D[5], bb2D[7], cv::Scalar(0, 0, 0), lineW+2, lineType);
    cv::line(img, bb2D[7], bb2D[6], cv::Scalar(0, 0, 0), lineW+2, lineType);
    cv::line(img, bb2D[6], bb2D[4], cv::Scalar(0, 0, 0), lineW+2, lineType);
    
    cv::line(img, bb2D[0], bb2D[1], color, lineW, lineType);
    cv::line(img, bb2D[1], bb2D[3], color, lineW, lineType);
    cv::line(img, bb2D[3], bb2D[2], color, lineW, lineType);
    cv::line(img, bb2D[2], bb2D[0], color, lineW, lineType);
    
    cv::line(img, bb2D[2], bb2D[6], color, lineW, lineType);
    cv::line(img, bb2D[0], bb2D[4], color, lineW, lineType);
    cv::line(img, bb2D[1], bb2D[5], color, lineW, lineType);
    cv::line(img, bb2D[3], bb2D[7], color, lineW, lineType);
    
    cv::line(img, bb2D[4], bb2D[5], color, lineW, lineType);
    cv::line(img, bb2D[5], bb2D[7], color, lineW, lineType);
    cv::line(img, bb2D[7], bb2D[6], color, lineW, lineType);
    cv::line(img, bb2D[6], bb2D[4], color, lineW, lineType);
}


void drawBBs(
    jp::img_bgr_t& img, 
    const std::vector<jp::info_t>& infos, 
    const std::vector<std::vector<cv::Point3f>>& bb3Ds, 
    const std::vector<cv::Scalar>& colors)
{
    for(unsigned o = 0; o < infos.size(); o++)
    {
	if(!infos[o].visible) continue;
	drawBB(img, jp::our2cv(infos[o]), bb3Ds[o], colors[o]);
    }  
}


cv::Mat getModeImg(
    const std::vector<jp::RegressionTree<jp::feature_t>>& forest,
    const std::vector<jp::img_leaf_t>& leafImgs,
    jp::info_t info, int treeIdx, jp::id_t objID)
{
    if(leafImgs.empty()) return cv::Mat();
  
    treeIdx = std::max(0, treeIdx);
    
    jp::img_coord_t modeImg(leafImgs[0].rows, leafImgs[0].cols);
  
    for(int x = 0; x < modeImg.cols; x++)
    for(int y = 0; y < modeImg.rows; y++)
    {
	size_t leaf = leafImgs[treeIdx](y, x);
	const std::vector<jp::mode_t>* modes = forest[treeIdx].getModes(leaf, objID);
	
	if(modes->empty() || !leaf)
	    modeImg(y, x) = jp::coord3_t(0, 0, 0); 
	else
	    modeImg(y, x) = modes->at(0).mean;
    }
    
    return convertForDisplay(modeImg, info);
}


void drawForestEstimation(
    jp::img_bgr_t& segImg,
    jp::img_bgr_t& objImg,
    const std::vector<jp::RegressionTree<jp::feature_t>>& forest,
    const std::vector<jp::img_leaf_t>& leafImgs,
    const std::vector<jp::info_t>& infos,
    const std::vector<jp::img_stat_t>& probabilities,
    const std::vector<cv::Scalar>& colors)
{
    std::vector<jp::img_bgr_t> estObjImgs;
    
    for(unsigned o = 0; o < infos.size(); o++)
	estObjImgs.push_back(getModeImg(forest, leafImgs, infos[o], 0, o+1));
    
    #pragma omp parallel for
    for(unsigned x = 0; x < segImg.cols; x++)
    for(unsigned y = 0; y < segImg.rows; y++)
    for(unsigned p = 0; p < probabilities.size(); p++)
    {
	float prob = probabilities[p](y, x); 
			
	segImg(y, x)[0] += prob * colors[p][0];
	segImg(y, x)[1] += prob * colors[p][1];
	segImg(y, x)[2] += prob * colors[p][2];
	
	objImg(y, x)[0] += prob * estObjImgs[p](y, x)[0];
	objImg(y, x)[1] += prob * estObjImgs[p](y, x)[1];
	objImg(y, x)[2] += prob * estObjImgs[p](y, x)[2];
    }  
}


void drawGroundTruth(
    jp::img_bgr_t& segImg,
    jp::img_bgr_t& objImg,
    const std::vector<jp::img_coord_t>& gtObjImgs,
    const std::vector<jp::img_id_t>& gtSegImgs,
    const std::vector<jp::info_t>& infos,
    const std::vector<cv::Scalar>& colors)
{
    for(unsigned o = 0; o < infos.size(); o++)
    {
	if(!infos[o].visible) continue; // skip invisible objects
	
	jp::img_bgr_t curObjImg = convertForDisplay(gtObjImgs[o], infos[o]);
      
	#pragma omp parallel for
	for(unsigned x = 0; x < segImg.cols; x++)
	for(unsigned y = 0; y < segImg.rows; y++)
	{
	    if(!gtSegImgs[o](y, x)) continue; // skip all pixels that do not belong to the object
			    
	    segImg(y, x)[0] = colors[o][0];
	    segImg(y, x)[1] = colors[o][1];
	    segImg(y, x)[2] = colors[o][2];
	    
	    objImg(y, x)[0] = curObjImg(y, x)[0];
	    objImg(y, x)[1] = curObjImg(y, x)[1];
	    objImg(y, x)[2] = curObjImg(y, x)[2];
	}
    }
}

std::vector<cv::Scalar> getColors()
{
    std::vector<cv::Scalar> colors;
    colors.push_back(cv::Scalar(255, 0, 0));
    colors.push_back(cv::Scalar(0, 255, 0));
    colors.push_back(cv::Scalar(0, 0, 255));
    
    colors.push_back(cv::Scalar(255, 255, 0));
    colors.push_back(cv::Scalar(255, 0, 255));
    colors.push_back(cv::Scalar(0, 255, 255));
    
    colors.push_back(cv::Scalar(255, 127, 0));
    colors.push_back(cv::Scalar(255, 0, 127));
    colors.push_back(cv::Scalar(255, 127, 127));
    
    colors.push_back(cv::Scalar(127, 255, 0));
    colors.push_back(cv::Scalar(0, 255, 127));
    colors.push_back(cv::Scalar(127, 255, 127));

    colors.push_back(cv::Scalar(127, 0, 255));
    colors.push_back(cv::Scalar(0, 127, 255));
    colors.push_back(cv::Scalar(127, 127, 255));
    
    for(unsigned c = colors.size(); c < 50; c++)
	colors.push_back(cv::Scalar(127, 127, 127));
    
    return colors;
}
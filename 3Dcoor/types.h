#pragma once

#include "opencv2/opencv.hpp"

#define EPS 0.00000001
#define PI 3.1415926



namespace jp
{
    // objID
    typedef unsigned char id_t;

    // object coordinates 
    typedef short coord1_t; // 一维坐标
    typedef cv::Vec<coord1_t, 3> coord3_t; // 三维坐标

    // label types
    typedef unsigned short cell_t; 
    typedef unsigned short label_t; 
    typedef std::vector<unsigned int> histogram_t; 

    // rgb-d
    typedef cv::Vec<uchar, 3> bgr_t;
    typedef unsigned short depth_t;

    // image types
    typedef cv::Mat_<coord3_t> img_coord_t; // object coodinate images
    typedef cv::Mat_<bgr_t> img_bgr_t; // color images
    typedef cv::Mat_<depth_t> img_depth_t; // depth images
    typedef cv::Mat_<label_t> img_label_t; // label images (quantized object coordinates + object ID)
    typedef cv::Mat_<id_t> img_id_t; // segmentation images

    struct mode_t
    {
     
	mode_t()
	{
	    mean = jp::coord3_t(0, 0, 0);
	    covar = cv::Mat::zeros(3, 3, CV_32F);
	    support = 0;
	}
      
	jp::coord3_t mean; 
	cv::Mat_<float> covar; 
	unsigned support; 
	
	bool operator<(const mode_t& mode) const
	{ 
	    return (this->support < mode.support);
	}
    };  
    
    typedef cv::Mat_<size_t> img_leaf_t; 
    typedef cv::Mat_<float> img_stat_t; 
    

   
    struct img_bgrd_t
    {
	img_bgr_t bgr;
	img_depth_t depth;
    };

  
    struct img_data_t
    {
	img_id_t seg; 
	img_bgr_t colorData; 
	
	std::vector<img_label_t> labelData; 
	std::vector<img_coord_t> coordData; 
    };
    
    struct info_t
    {
	std::string name;
	cv::Mat_<float> rotation; 
	cv::Vec<float, 3> center; 
	cv::Vec<float, 3> extent; 
	bool visible; 
	float occlusion; 
	
	
	info_t(bool v = true)
	{
	    rotation = cv::Mat_<float>::eye(3, 3);
	    center = cv::Vec<float, 3>(0, 0, -1);
	    extent = cv::Vec<float, 3>(1, 1, 1);
	    visible = v;
	    occlusion = 0;
	}
    };  
    
    typedef std::pair<cv::Mat, cv::Mat> cv_trans_t; 
    typedef std::pair<cv::Mat, cv::Point3d> jp_trans_t; 
    
  
    inline cv::Mat float2double(cv::Mat& fmat) 
    {
	cv::Mat_<double> dmat(fmat.rows, fmat.cols);
	
	for(unsigned i = 0; i < fmat.cols; i++)
	for(unsigned j = 0; j < fmat.rows; j++)
	    dmat(j, i) = fmat.at<float>(j, i);
	
	return dmat;
    }
   
    inline cv::Mat double2float(cv::Mat& dmat) 
    {
	cv::Mat_<float> fmat(dmat.rows, dmat.cols);
	
	for(unsigned i = 0; i < dmat.cols; i++)
	for(unsigned j = 0; j < dmat.rows; j++)
	    fmat(j, i) = dmat.at<double>(j, i);
	
	return fmat;
    }     
    
    
 
    inline cv_trans_t our2cv(const jp_trans_t& trans)
    {
	cv::Mat rmat = trans.first.clone(), rvec;
	rmat.row(1) = -rmat.row(1);
	rmat.row(2) = -rmat.row(2);
	cv::Rodrigues(rmat, rvec);
	
	cv::Mat tvec(3, 1, CV_64F);
	tvec.at<double>(0, 0) = trans.second.x;
	tvec.at<double>(1, 0) = -trans.second.y;
	tvec.at<double>(2, 0) = -trans.second.z;

	return cv_trans_t(rvec, tvec);
    }

     
    inline cv_trans_t our2cv(const jp::info_t& info)
    {
	cv::Mat rmat = info.rotation.clone(), rvec;
	rmat.row(1) = -rmat.row(1);
	rmat.row(2) = -rmat.row(2);
	rmat = float2double(rmat);
	cv::Rodrigues(rmat, rvec);
	
	cv::Mat tvec(3, 1, CV_64F);
	tvec.at<double>(0, 0) = info.center[0] * 1000.0;
	tvec.at<double>(1, 0) = -info.center[1] * 1000.0;
	tvec.at<double>(2, 0) = -info.center[2] * 1000.0;

	return cv_trans_t(rvec, tvec);
    }
    
    inline jp_trans_t cv2our(const cv_trans_t& trans)
    {

	cv::Mat rmat;
	cv::Rodrigues(trans.first, rmat);
	cv::Point3d tpt(trans.second.at<double>(0, 0), trans.second.at<double>(1, 0), trans.second.at<double>(2, 0));
	

	rmat.row(1) = -rmat.row(1);
	rmat.row(2) = -rmat.row(2);
	tpt.y = -tpt.y;
	tpt.z = -tpt.z;


	if(cv::determinant(rmat) < 0)
	{
	    tpt = -tpt;
	    rmat = -rmat;
	}
	
	return jp_trans_t(rmat, tpt);      
    }
}
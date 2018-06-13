#pragma once

#include "properties.h"
#include "util.h"
#include "read_data.h"
#include <stdexcept>

/** 用于读写数据和一些基本操作的接口.*/

namespace jp
{
    /**
	 * objID物体id   objCell分割bin的标签  return ojb标签
     */
    jp::label_t getLabel(jp::id_t objID, jp::cell_t objCell);

   
    jp::coord3_t pxToEye(int x, int y, jp::depth_t depth);

    /**
    物体坐标是否为空
     */
    bool onObj(const jp::coord3_t& pt); 
    
    /**
    给定标签是否为背景
     */    
    bool onObj(const jp::label_t& pt);
    
    
    class Dataset
    {
    public:

      	Dataset()
	{
	}
      
	/**
	 basePath: 数据集地址
	 objID : 物体号
	 */
	Dataset(const std::string& basePath, jp::label_t objID) : objID(objID)
	{
	    readFileNames(basePath);
	}

	
	jp::id_t getObjID() const
	{
	    return objID;
	}
	
	
	size_t size() const 
	{ 
	    return bgrFiles.size();
	}

	
	std::string getFileName(size_t i) const
	{
	    return bgrFiles[i];
	}
	
	
	bool getInfo(size_t i, jp::info_t& info) const
	{
	    if(infoFiles.empty()) return false;
	    if(!readData(infoFiles[i], info))
		return false;
	    return true;
	}	
	
	
	void getBGR(size_t i, jp::img_bgr_t& img, bool noseg) const
	{
	    std::string bgrFile = bgrFiles[i];
	    
	    readData(bgrFile, img);
	    
	    if(!noseg) // 根据ground truth的分割
	    {
		jp::img_id_t seg;
		getSegmentation(i, seg);
		
		for(unsigned x = 0; x < img.cols; x++)
		for(unsigned y = 0; y < img.rows; y++)
		    if(!seg(y, x)) img(y, x) = jp::bgr_t(0, 0, 0);
	    }
	    
	    GlobalProperties* gp = GlobalProperties::getInstance();
	    if(!gp->fP.rawData) return;
	    
	    
	    float scaleFactor = gp->fP.focalLength / gp->fP.secondaryFocalLength; 
	    float transCorrX = (1 - scaleFactor) * (gp->fP.imageWidth * 0.5 + gp->fP.rawXShift); 
	    float transCorrY = (1 - scaleFactor) * (gp->fP.imageHeight * 0.5 + gp->fP.rawYShift); 
		
	    
	    cv::Mat trans = cv::Mat::eye(2, 3, CV_64F) * scaleFactor;
	    trans.at<double>(0, 2) = transCorrX;
	    trans.at<double>(1, 2) = transCorrY;
	    jp::img_bgr_t temp;
	    cv::warpAffine(img, temp, trans, img.size());
	    img = temp;
	}

	
	void getDepth(size_t i, jp::img_depth_t& img, bool noseg) const
	{
	    std::string dFile = depthFiles[i];

	    readData(dFile, img);
	    
	    if(!noseg) 
	    {
		jp::img_id_t seg;
		getSegmentation(i, seg);
		
		for(unsigned x = 0; x < img.cols; x++)
		for(unsigned y = 0; y < img.rows; y++)
		    if(!seg(y, x)) img(y, x) = 0;
	    }
	}
	
	
	void getBGRD(size_t i, jp::img_bgrd_t& img, bool noseg) const
	{
	    getBGR(i, img.bgr, noseg);
	    getDepth(i, img.depth, noseg);
	}

	
	void getSegmentation(size_t i, jp::img_id_t& seg) const
	{
	    jp::img_bgr_t segTemp;
	    readData(segFiles[i], segTemp);
	  
	    seg = jp::img_id_t::zeros(segTemp.rows, segTemp.cols);
	    int margin = 3; 
	    for(unsigned x = margin; x < seg.cols-margin; x++)
	    for(unsigned y = margin; y < seg.rows-margin; y++)
		if(segTemp(y,x)[0]) seg(y,x) =  1;
	}
	

	void getObj(size_t i, jp::img_coord_t& img) const
	{
	    readData(objFiles[i], img);
	}

	
	void getEye(size_t i, jp::img_coord_t& img) const
	{
	    jp::img_depth_t imgDepth;
	    getDepth(i, imgDepth, true);
	    
	    img = jp::img_coord_t(imgDepth.rows, imgDepth.cols);
	    
	    #pragma omp parallel for
	    for(int x = 0; x < img.cols; x++)
	    for(int y = 0; y < img.rows; y++)
	    {
	       img(y, x) = pxToEye(x, y, imgDepth(y, x));
	    }
	}
	
    private:
	  
      /**
      读取obj信息
       */
      void readFileNames(const std::string& basePath)
	{
	    std::cout << "Reading file names... " << std::endl;
	    std::string segPath = "/seg/", segSuf = ".png";
	    std::string bgrPath = "/rgb_noseg/", bgrSuf = ".png";
	    std::string dPath = "/depth_noseg/", dSuf = ".png";
	    std::string objPath = "/obj/", objSuf = ".png";
	    std::string infoPath = "/info/", infoSuf = ".txt";

	    bgrFiles = getFiles(basePath + bgrPath, bgrSuf);
	    depthFiles = getFiles(basePath + dPath, dSuf);
	    infoFiles = getFiles(basePath + infoPath, infoSuf, true);
	    segFiles = getFiles(basePath + segPath, segSuf, true);
	    objFiles = getFiles(basePath + objPath, objSuf, true);
	}

	jp::id_t objID; // object ID 
    
	// 图片数据文件
	std::vector<std::string> bgrFiles;
	std::vector<std::string> depthFiles; 
	// 真实数据文件
	std::vector<std::string> segFiles; 
	std::vector<std::string> objFiles;
	std::vector<std::string> infoFiles; 
    };
}
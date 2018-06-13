#pragma once

#include "types.h"
#include "Hypothesis.h"

double evaluatePoseAligned(const std::vector<cv::Point3d>& pointCloud, 
    Hypothesis& estHyp, Hypothesis& gtHyp)
{
    double sumDist = 0.0;
  
    for(unsigned i = 0; i < pointCloud.size(); i++)
    {
	cv::Point3d estPt = estHyp.transform(pointCloud[i]);
	cv::Point3d gtPt = gtHyp.transform(pointCloud[i]);
	
	sumDist += cv::norm(cv::Mat(estPt), cv::Mat(gtPt));
    }
    
    return sumDist / pointCloud.size();
}

double evaluatePose2D(
    const std::vector<cv::Point3d>& pointCloud, 
    const std::pair<cv::Mat, cv::Mat>& estTrans, 
    const std::pair<cv::Mat, cv::Mat>& gtTrans)
{
    std::vector<cv::Point3f> pc;
    for(unsigned i = 0; i < pointCloud.size(); i++)
	pc.push_back(cv::Point3f(pointCloud[i].x, pointCloud[i].y, pointCloud[i].z));
  
    std::vector<cv::Point2f> projectionsEst, projectionsGT;
    cv::Mat camMat = GlobalProperties::getInstance()->getCamMat();
    
    cv::projectPoints(pc, estTrans.first, estTrans.second, camMat, cv::Mat(), projectionsEst);
    cv::projectPoints(pc, gtTrans.first, gtTrans.second, camMat, cv::Mat(), projectionsGT);    
    
    double sumDist = 0.0;

    for(unsigned i = 0; i < pointCloud.size(); i++)
	sumDist += cv::norm(projectionsEst[i] - projectionsGT[i]);
    
    return sumDist / pointCloud.size();
}

double evaluatePoseUnaligned(const std::vector<cv::Point3d>& pointCloud, 
    Hypothesis& estHyp, Hypothesis& gtHyp)
{
    double sumDist = 0.0;
  
    std::vector<cv::Point3d> pcEst(pointCloud.size());
    std::vector<cv::Point3d> pcGT(pointCloud.size());
    
    #pragma omp parallel for
    for(unsigned i = 0; i < pointCloud.size(); i++)
    {
	pcEst[i] = estHyp.transform(pointCloud[i]);
	pcGT[i] = gtHyp.transform(pointCloud[i]);
    }

    #pragma omp parallel for    
    for(unsigned i = 0; i < pcEst.size(); i++)
    {
        // for each vertex search for the closest corresponding vertex
	double minDist = cv::norm(cv::Mat(pcEst[i]), cv::Mat(pcGT[0]));
	
	for(unsigned j = 0; j < pcGT.size(); j++)
	{
	    double currentDist = cv::norm(cv::Mat(pcEst[i]), cv::Mat(pcGT[j]));
	    minDist = std::min(minDist, currentDist);
	}
	
	#pragma omp critical
	{
	    sumDist += minDist;
	}
    }
    
    return sumDist / pointCloud.size();
}


 double evaluatePose(const std::vector<cv::Point3d>& pointCloud, 
    Hypothesis& estHyp, Hypothesis& gtHyp, bool rotationObject)
{
    if(rotationObject)
	return evaluatePoseUnaligned(pointCloud, estHyp, gtHyp);
    else
	return evaluatePoseAligned(pointCloud, estHyp, gtHyp);
}
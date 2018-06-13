#pragma once

#include "types.h"
#include "util.h"
#include "sampler2D.h"
#include "detection.h"
#include "stop_watch.h"
#include "Hypothesis.h"

#include <nlopt.hpp>
#include <omp.h>

/**
 * finding poses based on object coordinate predictions in the RGB-D case.
 */
class Ransac3D
{
public:
    Ransac3D()
    {
    };
  
    struct TransHyp
    {
	TransHyp() {}
	TransHyp(jp::id_t objID, jp::cv_trans_t pose) : pose(pose), objID(objID), inliers(0), maxPixels(0), effPixels(0), refSteps(0), likelihood(0) {}
      
	jp::id_t objID; // hypothesis的objID
	jp::cv_trans_t pose; // the actual transformation
	
	cv::Rect bb; // 2D bounding box of the object under this pose hypothesis
	
	std::vector<std::pair<cv::Point3d, cv::Point3d>> inlierPts; // obj coord的列表
	std::vector<const jp::mode_t*> inlierModes; // 
	
	int maxPixels; // 这个hyp的最大像素点数
	int effPixels; 
	
	int inliers;
	float likelihood; // likelihood of this hypothesis (optimization using uncertainty)

	int refSteps; // how many iterations has this hyp been refined?
	
	
	float getScore() const 	{ return inliers; }
	
	
	float getInlierRate() const { return inliers / (float) effPixels; }
	
	
	bool operator < (const TransHyp& hyp) const { return (getScore() > hyp.getScore()); } 
    };
    
    
    struct DataForOpt
    {
	TransHyp* hyp; 
	Ransac3D* ransac; 
    };
    
  
    inline void filterInliers(
	TransHyp& hyp,
	int maxInliers)
    {
	if(hyp.inlierPts.size() < maxInliers) return; 
      		
	std::vector<std::pair<cv::Point3d, cv::Point3d>> inlierPts; 
	std::vector<const jp::mode_t*> inlierModes;
	
	
	for(unsigned i = 0; i < maxInliers; i++)
	{
	    int idx = irand(0, hyp.inlierPts.size());
	    
	    inlierPts.push_back(hyp.inlierPts[idx]);
	    inlierModes.push_back(hyp.inlierModes[idx]);
	}
	
	hyp.inlierPts = inlierPts;
	hyp.inlierModes = inlierModes;
    }
    
  
    inline double likelihood(cv::Point3d eyePt, const cv::Mat_<float>& mean, const cv::Mat_<float>& covar)
    {
	cv::Mat_<float> eyeMat(3, 1);
	eyeMat(0, 0) = eyePt.x;
	eyeMat(1, 0) = eyePt.y;
	eyeMat(2, 0) = eyePt.z;
	
	// evaluate Gaussian at the camera coordinate
	eyeMat = eyeMat - mean;
	eyeMat = eyeMat.t() * covar.inv() * eyeMat;
	
	float l = eyeMat(0, 0);
	l = std::pow(2 * PI, -1.5) * std::pow(cv::determinant(covar), -0.5) * std::exp(-0.5 * l);
	
	return l;
    }    

  
    inline void updateHyp3D(TransHyp& hyp,
	const cv::Mat& camMat, 
	int imgWidth, 
	int imgHeight, 
	const std::vector<cv::Point3f>& bb3D,
	int maxPixels)
    {
	if(hyp.inlierPts.size() < 4) return;
	filterInliers(hyp, maxPixels); // limit the number of correspondences
      
	// data conversion
	jp::jp_trans_t pose = jp::cv2our(hyp.pose);
	Hypothesis trans(pose.first, pose.second);	
	
	// recalculate pose
	trans.refine(hyp.inlierPts);
	hyp.pose = jp::our2cv(jp::jp_trans_t(trans.getRotation(), trans.getTranslation()));
	
	// update 2D bounding box
	hyp.bb = getBB2D(imgWidth, imgHeight, bb3D, camMat, hyp.pose);
    }
    

    void createSamplers(
	std::vector<Sampler2D>& samplers,
	const std::vector<jp::img_stat_t>& probs,
	int imageWidth,
	int imageHeight)
    {	
	samplers.clear();
    	jp::img_stat_t objProb = jp::img_stat_t::zeros(imageHeight, imageWidth);
	
	#pragma omp parallel for
	for(unsigned x = 0; x < objProb.cols; x++)
	for(unsigned y = 0; y < objProb.rows; y++)
	for(auto prob : probs)
	    objProb(y, x) += prob(y, x);
	
	samplers.push_back(Sampler2D(objProb));
	for(auto prob : probs)
	    samplers.push_back(Sampler2D(prob));
    }
    

    inline jp::id_t drawObjID(
	const cv::Point2f& pt,
	const std::vector<jp::img_stat_t>& probs)
    {
 
	std::map<float, jp::id_t> cumProb; 
	float probCur, probSum = 0;
	
	for(unsigned idx = 0; idx < probs.size(); idx++)
	{
	    probCur = probs[idx](pt.y, pt.x);

	    if(probCur < EPSILON) 
		continue;
	    
	    probSum += probCur;
	    cumProb[probSum] = idx + 1;
	}
	
	return cumProb.upper_bound(drand(0, probSum))->second;
    }
    

    std::vector<TransHyp*> getWorkingQueue(std::map<jp::id_t, std::vector<TransHyp>>& hypMap, int maxIt)
    {
	std::vector<TransHyp*> workingQueue;
      
	for(auto it = hypMap.begin(); it != hypMap.end(); it++)
	for(int h = 0; h < it->second.size(); h++)
	    if(it->second.size() > 1 || it->second[h].refSteps < maxIt) 
		workingQueue.push_back(&(it->second[h]));

	return workingQueue;
    }
    
    
   
    float estimatePose(
	const jp::img_coord_t& eyeData,
	const std::vector<jp::img_stat_t>& probs, 
	const std::vector<jp::RegressionTree<jp::feature_t>>& forest, 
	const std::vector<jp::img_leaf_t>& leafImgs,
	const std::vector<std::vector<cv::Point3f>>& bb3Ds)
    {
	GlobalProperties* gp = GlobalProperties::getInstance(); 
      

	int maxIterations = gp->tP.ransacMaxDraws;
	float minDist3D = 10; 
	float minArea = 400;
		
	float inlierThreshold3D = gp->tP.ransacInlierThreshold3D;
	int ransacIterations = gp->tP.ransacIterations;
	int refinementIterations = gp->tP.ransacRefinementIterations;
	int preemptiveBatch = gp->tP.ransacBatchSize;
	int maxPixels = gp->tP.ransacMaxInliers;
	int minPixels = gp->tP.ransacMinInliers;
	int refIt = gp->tP.ransacCoarseRefinementIterations;

	bool fullRefine = gp->tP.ransacRefine;
	
	int imageWidth = gp->fP.imageWidth;
	int imageHeight = gp->fP.imageHeight;

	cv::Mat camMat = gp->getCamMat();	
	
	std::vector<Sampler2D> samplers;
	createSamplers(samplers, probs, imageWidth, imageHeight);
		
	std::map<jp::id_t, std::vector<TransHyp>> hypMap;
	
	float ransacTime = 0;
	StopWatch stopWatch;
	
	#pragma omp parallel for
	for(unsigned h = 0; h < ransacIterations; h++)
	for(unsigned i = 0; i < maxIterations; i++)
	{
	    std::vector<cv::Point3f> eyePts;
	    std::vector<cv::Point3f> objPts;
	  
	    cv::Rect bb2D(0, 0, imageWidth, imageHeight);
	    
	    cv::Point2f pt1 = samplers[0].drawInRect(bb2D);
	    jp::id_t objID = drawObjID(pt1, probs);

	    if(objID == 0) continue;
	    
	    if(!samplePoint(objID, eyePts, objPts, pt1, forest, leafImgs, eyeData, minDist3D))
		continue;
	    
	    float searchRadius = (gp->fP.focalLength * getMaxDist(bb3Ds[objID-1], objPts[0]) / -eyePts[0].z) / 2;

	    int minX = clamp(pt1.x - searchRadius, 0, imageWidth - 1);
	    int maxX = clamp(pt1.x + searchRadius, 0, imageWidth - 1);
	    int minY = clamp(pt1.y - searchRadius, 0, imageHeight - 1);
	    int maxY = clamp(pt1.y + searchRadius, 0, imageHeight - 1);

	    bb2D = cv::Rect(minX, minY, (maxX - minX + 1), (maxY - minY + 1));

	    if(!samplePoint(objID, eyePts, objPts, samplers[objID].drawInRect(bb2D), forest, leafImgs, eyeData, minDist3D))
		continue;
	    
	    if(!samplePoint(objID, eyePts, objPts, samplers[objID].drawInRect(bb2D), forest, leafImgs, eyeData, minDist3D))
		continue;

	    std::vector<std::pair<cv::Point3d, cv::Point3d>> pts3D;
	    for(unsigned j = 0; j < eyePts.size(); j++)
	    {
		pts3D.push_back(std::pair<cv::Point3d, cv::Point3d>(
		    cv::Point3d(objPts[j].x, objPts[j].y, objPts[j].z),
		    cv::Point3d(eyePts[j].x, eyePts[j].y, eyePts[j].z)
		));
	    }
		
	    Hypothesis trans(pts3D);

	    bool foundOutlier = false;
	    for(unsigned j = 0; j < pts3D.size(); j++)
	    {
		if(cv::norm(pts3D[j].second - trans.transform(pts3D[j].first)) < inlierThreshold3D) continue;
		foundOutlier = true;
		break;
	    }
	    if(foundOutlier) continue;

	    jp::jp_trans_t pose;
	    pose.first = trans.getRotation();
	    pose.second = trans.getTranslation();
	    
	    TransHyp hyp(objID, jp::our2cv(pose));
	    
	    hyp.bb = getBB2D(imageWidth, imageHeight, bb3Ds[objID-1], camMat, hyp.pose);

	    if(hyp.bb.area() < minArea)
		continue;	    
	    
	    #pragma omp critical
	    {
		hypMap[objID].push_back(hyp);
	    }

	    break;
	}
	
	ransacTime += stopWatch.stop();
	std::cout << "Time after drawing hypothesis: " << ransacTime << "ms." << std::endl;

	std::vector<jp::id_t> objList;
	std::cout << std::endl;
	for(std::pair<jp::id_t, std::vector<TransHyp>> hypPair : hypMap)
	{
	    std::cout << "Object " << (int) hypPair.first << ": " << hypPair.second.size() << std::endl;
	    objList.push_back(hypPair.first);
	}
	std::cout << std::endl;

	std::vector<TransHyp*> workingQueue = getWorkingQueue(hypMap, refIt);
	
	while(!workingQueue.empty())
	{
	    #pragma omp parallel for
	    for(int h = 0; h < workingQueue.size(); h++)
		countInliers3D(*(workingQueue[h]), forest, leafImgs, eyeData, inlierThreshold3D, minArea, preemptiveBatch);
	    	    
	    #pragma omp parallel for 
	    for(unsigned o = 0; o < objList.size(); o++)
	    {
		jp::id_t objID = objList[o];
		if(hypMap[objID].size() > 1)
		{
		    std::sort(hypMap[objID].begin(), hypMap[objID].end());
		    hypMap[objID].erase(hypMap[objID].begin() + hypMap[objID].size() / 2, hypMap[objID].end());
		}
	    }
	    workingQueue = getWorkingQueue(hypMap, refIt);
	    
	    // refine
	    #pragma omp parallel for
	    for(int h = 0; h < workingQueue.size(); h++)
	    {
		updateHyp3D(*(workingQueue[h]), camMat, imageWidth, imageHeight, bb3Ds[workingQueue[h]->objID-1], maxPixels);
		workingQueue[h]->refSteps++;
	    }
	    
	    workingQueue = getWorkingQueue(hypMap, refIt);
	}

	ransacTime += stopWatch.stop();
	std::cout << "Time after preemptive RANSAC: " << ransacTime << "ms." << std::endl;

	poses.clear();	
	
	std::cout << std::endl << "---------------------------------------------------" << std::endl;
	for(auto it = hypMap.begin(); it != hypMap.end(); it++)
	for(int h = 0; h < it->second.size(); h++)
	{
	    std::cout << BLUETEXT("Estimated Hypothesis for Object " << (int) it->second[h].objID << ":") << std::endl;
	    
	    // apply refinement using uncertainty (if enabled)
	    if(fullRefine && it->second[h].inliers > minPixels) 
	    {
		filterInliers(it->second[h], maxPixels);
		it->second[h].likelihood = refineWithOpt(it->second[h], refinementIterations);
	    }
	  
	    // store pose in class member
	    poses[it->second[h].objID] = it->second[h];
	    
	    std::cout << "Inliers: " << it->second[h].inliers;
	    std::printf(" (Rate: %.1f\%)\n", it->second[h].getInlierRate() * 100);
	    std::cout << "Refined " << it->second[h].refSteps << " times. " << std::endl;
	    std::cout << "---------------------------------------------------" << std::endl;
	}
	std::cout << std::endl;
	
	if(fullRefine)
	{
	    ransacTime += stopWatch.stop();
	    std::cout << "Time after final refine: " << ransacTime << "ms." << std::endl << std::endl;
	}
	
	return ransacTime;
    }
    
private:
 
  
    inline void countInliers3D(
      TransHyp& hyp,
      const std::vector<jp::RegressionTree<jp::feature_t>>& forest,
      const std::vector<jp::img_leaf_t>& leafImgs,
      const jp::img_coord_t& eyeData,
      float inlierThreshold,
      int minArea,
      int pixelBatch)
    {
	hyp.inlierPts.clear();
	hyp.inlierModes.clear();
	hyp.inliers = 0;

	if(hyp.bb.area() < minArea) return;
	
	jp::jp_trans_t pose = jp::cv2our(hyp.pose);
	Hypothesis trans(pose.first, pose.second);

	hyp.effPixels = 0; 
	hyp.maxPixels += pixelBatch;	
	
	int maxPt = hyp.bb.area(); 
	float successRate = hyp.maxPixels / (float) maxPt; 
	
	std::mt19937 generator;
	std::negative_binomial_distribution<int> distribution(1, successRate); 
	
	for(unsigned ptIdx = 0; ptIdx < maxPt;)
	{
	    cv::Point2f pt2D(
		hyp.bb.x + ptIdx % hyp.bb.width, 
		hyp.bb.y + ptIdx / hyp.bb.width);
	    
	    if(eyeData(pt2D.y, pt2D.x)[2] == 0)
	    {
		ptIdx++;
		continue;
	    }
	  
	    cv::Point3d eye(eyeData(pt2D.y, pt2D.x)[0], eyeData(pt2D.y, pt2D.x)[1], eyeData(pt2D.y, pt2D.x)[2]);
	  
	    hyp.effPixels++;
	  
	    for(unsigned t = 0; t < forest.size(); t++)
	    {
		const std::vector<jp::mode_t>* modes = getModes(hyp.objID, pt2D, forest, leafImgs, t);
		for(int m = 0; m < modes->size(); m++)
		{
		    if(!jp::onObj(modes->at(m).mean)) continue; 
		    
		    cv::Point3d obj(modes->at(m).mean(0), modes->at(m).mean(1),modes->at(m).mean(2));

		    if(cv::norm(eye - trans.transform(obj)) < inlierThreshold)
		    {
			hyp.inlierPts.push_back(std::pair<cv::Point3d, cv::Point3d>(obj, eye)); 
			hyp.inlierModes.push_back(&(modes->at(m))); 
			hyp.inliers++; 
		    }
		}
	    }

	    if(successRate < 1)
		ptIdx += std::max(1, distribution(generator));
	    else
		ptIdx++;
	}
    }  
  
    inline const std::vector<jp::mode_t>* getModes(
	jp::id_t objID,
	const cv::Point2f& pt, 
	const std::vector<jp::RegressionTree<jp::feature_t>>& forest, 
	const std::vector<jp::img_leaf_t>& leafImgs,
	int treeIndex)
    {
	size_t leafIndex = leafImgs[treeIndex](pt.y, pt.x);
	return forest[treeIndex].getModes(leafIndex, objID);
    }  
  
  
    inline cv::Point3f getMode(
	jp::id_t objID,
	const cv::Point2f& pt, 
	const std::vector<jp::RegressionTree<jp::feature_t>>& forest, 
	const std::vector<jp::img_leaf_t>& leafImgs,
	int treeIndex = -1)
    {
	if(treeIndex < 0) treeIndex = irand(0, forest.size());

	size_t leafIndex = leafImgs[treeIndex](pt.y, pt.x);
	
	jp::coord3_t mode = forest[treeIndex].getModes(leafIndex, objID)->at(0).mean; 
	return cv::Point3f(mode(0), mode(1), mode(2));
    }

    template<class T>
    inline double getMinDist(const std::vector<T>& pointSet, const T& point)
    {
	double minDist = -1.f;
      
	for(unsigned i = 0; i < pointSet.size(); i++)
	{
	    if(minDist < 0) 
		minDist = cv::norm(pointSet.at(i) - point);
	    else
		minDist = std::min(minDist, cv::norm(pointSet.at(i) - point));
	}
	
	return minDist;
    }
   
    
    template<class T>
    inline double getMaxDist(const std::vector<T>& pointSet, const T& point)
    {
	double maxDist = -1.f;
      
	for(unsigned i = 0; i < pointSet.size(); i++)
	{
	    if(maxDist < 0) 
		maxDist = cv::norm(pointSet.at(i) - point);
	    else
		maxDist = std::max(maxDist, cv::norm(pointSet.at(i) - point));
	}
	
	return maxDist;
    }   
   
 
    inline double pointLineDistance(
	const cv::Point3f& pt1, 
	const cv::Point3f& pt2, 
	const cv::Point3f& pt3)
    {
	return cv::norm((pt2 - pt1).cross(pt3 - pt1)) / cv::norm(pt2 - pt1);
    }
    

    inline bool samplePoint(
	jp::id_t objID,
	std::vector<cv::Point3f>& eyePts, 
	std::vector<cv::Point3f>& objPts, 
	const cv::Point2f& pt2D,
	const std::vector<jp::RegressionTree<jp::feature_t>>& forest, 
	const std::vector<jp::img_leaf_t>& leafImgs,
	const jp::img_coord_t& eyeData,
	float minDist3D)
    {
	cv::Point3f eye(eyeData(pt2D.y, pt2D.x)[0], eyeData(pt2D.y, pt2D.x)[1], eyeData(pt2D.y, pt2D.x)[2]); 
	if(eye.z == 0) return false; 
	double minDist = getMinDist(eyePts, eye); 
	if(minDist > 0 && minDist < minDist3D) return false;
	
	cv::Point3f obj = getMode(objID, pt2D, forest, leafImgs); 
	if(obj.x == 0 && obj.y == 0 && obj.z == 0) return false; 
	minDist = getMinDist(objPts, obj); 
	if(minDist > 0 && minDist < minDist3D) return false;
	
	eyePts.push_back(eye);
	objPts.push_back(obj);
	
	return true;
    }
    
  
    static double optEnergy(const std::vector<double> &pose, std::vector<double> &grad, void *data)
    {
	DataForOpt* dataForOpt=(DataForOpt*) data;
	
	cv::Mat tvec(3, 1, CV_64F);
	cv::Mat rvec(3, 1, CV_64F);
      
	for(int i = 0; i < 6; i++)
	{
	    if(i > 2) 
		tvec.at<double>(i-3, 0) = pose[i] * 1000.0;
	    else 
		rvec.at<double>(i, 0) = pose[i];
	}
	
	jp::cv_trans_t trans(rvec, tvec);
      
	float energy = -dataForOpt->ransac->likelihood3D(
	    dataForOpt->hyp->objID, 
	    &(dataForOpt->hyp->inlierPts),
	    &(dataForOpt->hyp->inlierModes),
	    &trans);
	
	return energy;
    }

    double refineWithOpt(
	TransHyp& hyp,
	int iterations) 
    {
	nlopt::opt opt(nlopt::LN_NELDERMEAD, 6); 
      
	DataForOpt data;
	data.hyp = &hyp;
	data.ransac = this;

	std::vector<double> vec(6);
	for(int i = 0; i < 6; i++)
	{
	    if(i > 2) 
		vec[i] = hyp.pose.second.at<double>(i-3, 0) / 1000.0;
	    else vec[i] = 
		hyp.pose.first.at<double>(i, 0);
	}
	
	double rotRange = 10;
	rotRange *= PI / 180;
	double tRangeXY = 0.1;
	double tRangeZ = 0.5;
	
	std::vector<double> lb(6);
	lb[0] = vec[0]-rotRange; lb[1] = vec[1]-rotRange; lb[2] = vec[2]-rotRange;
	lb[3] = vec[3]-tRangeXY; lb[4] = vec[4]-tRangeXY; lb[5] = vec[5]-tRangeZ;
	opt.set_lower_bounds(lb);
      
	std::vector<double> ub(6);
	ub[0] = vec[0]+rotRange; ub[1] = vec[1]+rotRange; ub[2] = vec[2]+rotRange;
	ub[3] = vec[3]+tRangeXY; ub[4] = vec[4]+tRangeXY; ub[5] = vec[5]+tRangeZ;
	opt.set_upper_bounds(ub);
      
	std::cout << "Likelihood before refinement: ";

	std::cout << this->likelihood3D(
	    data.hyp->objID, 
	    &(data.hyp->inlierPts),
	    &(data.hyp->inlierModes),
	    &(hyp.pose));

	std::cout << std::endl;
	
	opt.set_min_objective(optEnergy, &data);
	opt.set_maxeval(iterations);

	double energy;
	nlopt::result result = opt.optimize(vec, energy);

	for(int i = 0; i < 6; i++)
	{
	    if(i > 2) 
		hyp.pose.second.at<double>(i-3, 0) = vec[i] * 1000.0;
	    else 
		hyp.pose.first.at<double>(i, 0) = vec[i];
	}
	
	std::cout << "Likelihood after refinement: " << -energy << std::endl;    
	return energy;
    }    

  
    float likelihood3D(
	jp::id_t objID,
	std::vector<std::pair<cv::Point3d, cv::Point3d>>* inlierPts,
	std::vector<const jp::mode_t*>* inliersModes,
	const jp::cv_trans_t* cvTrans) 
    {
     
	double likelihood3D = 0;
	
	double likelihood3DThreshMin = -100;
	double likelihood3DThreshMax = 100;
	float covarThresh = 1000;

	jp::jp_trans_t jpTrans = jp::cv2our(*cvTrans);
	jpTrans.first = jp::double2float(jpTrans.first);
	

	std::vector<double> localLikelihoods(omp_get_max_threads(), 0);
	
	#pragma omp parallel for
	for(int pt = 0; pt < inlierPts->size(); pt++) 
	{
	    unsigned threadID = omp_get_thread_num();
	    const jp::mode_t* mode = inliersModes->at(pt);

	    cv::Mat_<float> covar = mode->covar;
	    if(cv::determinant(covar) < covarThresh) 
		continue;
	    
	    cv::Mat_<float> obj(3, 1);
	    obj(0, 0) = mode->mean(0);
	    obj(1, 0) = mode->mean(1);
	    obj(2, 0) = mode->mean(2);
		
	    // convert mode center from object coordinates to camera coordinates
	    cv::Mat_<float> transObj = jpTrans.first * obj;
	    transObj(0, 0) += jpTrans.second.x;
	    transObj(1, 0) += jpTrans.second.y;
	    transObj(2, 0) += jpTrans.second.z;

	    cv::Mat_<float> transCovar = jpTrans.first * covar * jpTrans.first.t();
	    

	    localLikelihoods[threadID] += std::min(std::max(likelihood3DThreshMin, std::log(likelihood(inlierPts->at(pt).second, transObj, transCovar))), likelihood3DThreshMax);
	}
	
	// combine thread results
	for(unsigned t = 0; t < localLikelihoods.size(); t++)
	    likelihood3D += localLikelihoods[t];
	
	return likelihood3D;
    }
    
public:
    std::map<jp::id_t, TransHyp> poses; // Poses that have been estimated. At most one per object. Run estimatePose to fill this member.
};
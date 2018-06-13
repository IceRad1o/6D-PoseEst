#pragma once

#include "types.h"
#include "util.h"
#include "sampler2D.h"
#include "detection.h"
#include "stop_watch.h"
#include "Hypothesis.h"
#include <exception>

#include <nlopt.hpp>
#include <omp.h>


class Ransac2D
{
public:
    Ransac2D()
    {
    };
  
    
    struct TransHyp
    {
	TransHyp() {}
	TransHyp(jp::id_t objID, jp::cv_trans_t pose) : pose(pose), objID(objID), inliers(0), maxPixels(0), effPixels(0), refSteps(0), likelihood(0) {}
      
	jp::id_t objID; // ID of the object this hypothesis belongs to
	jp::cv_trans_t pose; // the actual transformation
	
	cv::Rect bb; // 2D bounding box of the object under this pose hypothesis
	
	// 2D - 3D inlier correspondences
	std::vector<cv::Point3f> inliers3D; // object coordinate inliers
	std::vector<cv::Point2f> inliers2D; // pixel positions associated with the object coordinate inliers
	std::vector<const jp::mode_t*> inliersM; // object coordinate distribution modes associated with the object coordinate inliers
	
	int maxPixels; // how many pixels should be maximally drawn to score this hyp
	int effPixels; // how many pixels habe effectively drawn (bounded by projection size)
	
	int inliers; // how many of them were inliers
	float likelihood; // likelihood of this hypothesis (optimization using uncertainty)

	int refSteps; // how many iterations has this hyp been refined?
	
	
  	float getScore() const 	{ return inliers; }
  	
	
	float getInlierRate() const { return inliers / (float) effPixels; }
	
	
	bool operator < (const TransHyp& hyp) const { return (getScore() > hyp.getScore()); } 
    };

    
    struct DataForOpt
    {
	TransHyp* hyp; // pointer to the data attached to the hypothesis being optimized.
	Ransac2D* ransac; // pointer to the RANSAC object for access of various methods.
    };
    
    inline void filterInliers(
	TransHyp& hyp,
	int maxInliers)
    {
	if(hyp.inliers2D.size() < maxInliers) return; // maximum number not reached, do nothing
      		
	// filtered list of inlier correspondences
	std::vector<cv::Point3f> inliers3D;
	std::vector<cv::Point2f> inliers2D;
	std::vector<const jp::mode_t*> inliersM;
	
	// select random correspondences to keep
	for(unsigned i = 0; i < maxInliers; i++)
	{
	    int idx = irand(0, hyp.inliers2D.size());
	    
	    inliers2D.push_back(hyp.inliers2D[idx]);
	    inliers3D.push_back(hyp.inliers3D[idx]);
	    inliersM.push_back(hyp.inliersM[idx]);
	}
	
	hyp.inliers2D = inliers2D;
	hyp.inliers3D = inliers3D;
	hyp.inliersM = inliersM;
    }
     
    inline double likelihood(int x, int y, const cv::Mat_<float>& camMat, const cv::Mat_<float>& mean, const cv::Mat_<float>& covar)
    {
	// calculate normalized image coordinates
	x = -(x - camMat.at<float>(0, 2));
	y = +(y - camMat.at<float>(1, 2));
	
	// calculate the pixel ray, we assume camera center 0 0 0 is the anchor for all points
	cv::Mat_<float> rayDir(3, 1);
	rayDir(2, 0) = -1000.f;
	rayDir(0, 0) = x * rayDir(2, 0) / camMat.at<float>(0, 0);
	rayDir(1, 0) = y * rayDir(2, 0) / camMat.at<float>(1, 1);
	
	rayDir /= cv::norm(rayDir);
	
	// center ray
	cv::Mat_<float> rayC(3, 1);
	rayC(2, 0) = 1.f;
	rayC(0, 0) = 0;
	rayC(1, 0) = 0;
	
	// rotate the pixel ray to the center ray (z-axis)
	cv::Mat_<float> axis = rayDir.cross(rayC); // rotation axis
	    
	float rotS = cv::norm(axis);
	float rotC = rayDir.dot(rayC);
	    
	cv::Mat_<float> crossM = cv::Mat_<float>::zeros(3, 3);
	crossM(1, 0) = axis(2, 0);
	crossM(2, 0) = -axis(1, 0);
	
	crossM(0, 1) = -axis(2, 0);
	crossM(2, 1) = axis(0, 0);		
	
	crossM(0, 2) = axis(1, 0);
	crossM(1, 2) = -axis(0, 0);		

	cv::Mat_<float> rotation = cv::Mat_<float>::eye(3, 3) + crossM + crossM * crossM * (1 - rotC) / rotS / rotS;		
	
	// apply ray rotation to the Gaussian
	cv::Mat_<float> rotMean = rotation * mean;
	cv::Mat_<float> rotCovar = rotation * covar * rotation.t();
		
	// easy access to inv covar matrix
 	cv::Mat_<float> invCov = rotCovar.inv();
	float a = invCov(0, 0);
	float b = invCov(0, 1);
	float c = invCov(0, 2);
	float d = invCov(1, 1);
	float e = invCov(1, 2);
	float f = invCov(2, 2);
	
	// constant factor of normal distribution
	double k = std::pow(2.0 * PI, -1.5) * std::pow(cv::determinant(rotCovar), -0.5);
	
	// easy access of mean
	double mx = rotMean(0, 0);
	double my = rotMean(1, 0);
	double mz = rotMean(2, 0);

	// gaussian parameters (x and y are zero)
	double gf = f/2.0;
	double gg = c*mx + e*my + f*mz;
	double gh = -(b*mx*my + c*mx*mz + e*my*mz + a*mx*mx/2 + d*my*my/2 + f*mz*mz/2);
	
	// return integral over z with factor z*z
	return k * std::sqrt(PI) * (2.0*gf + gg*gg) / (4.0*std::pow(gf, 2.5)) * std::exp(gg*gg / 4.0 / gf + gh);
    }    

    inline void updateHyp2D(TransHyp& hyp,
	const cv::Mat& camMat, 
	int imgWidth, 
	int imgHeight, 
	const std::vector<cv::Point3f>& bb3D,
	int maxPixels)
    {
	if(hyp.inliers2D.size() < 4) return;
	filterInliers(hyp, maxPixels); // limit the number of correspondences
      
	// recalculate pose
	cv::solvePnP(hyp.inliers3D, hyp.inliers2D, camMat, cv::Mat(), hyp.pose.first, hyp.pose.second, true, CV_EPNP);
	
	// update 2D bounding box
	hyp.bb = getBB2D(imgWidth, imgHeight, bb3D, camMat, hyp.pose);
    }
    
    inline void createSamplers(
	std::vector<Sampler2D>& samplers,
	const std::vector<jp::img_stat_t>& probs,
	int imageWidth,
	int imageHeight)
    {
	samplers.clear();
    	jp::img_stat_t objProb = jp::img_stat_t::zeros(imageHeight, imageWidth);
	
	// calculate accumulated probability (any object vs background)
	#pragma omp parallel for
	for(unsigned x = 0; x < objProb.cols; x++)
	for(unsigned y = 0; y < objProb.rows; y++)
	for(unsigned p = 0; p < probs.size(); p++)
	    objProb(y, x) += probs[p](y, x);
	
	// create samplers
	samplers.push_back(Sampler2D(objProb));
	for(auto prob : probs)
	    samplers.push_back(Sampler2D(prob));
    }
   
    inline jp::id_t drawObjID(
	const cv::Point2f& pt,
	const std::vector<jp::img_stat_t>& probs)
    {
	// create a map of accumulated object probabilities at the given pixel
	std::map<float, jp::id_t> cumProb; //map of accumulated probability -> object ID
	float probCur, probSum = 0;
	
	for(unsigned idx = 0; idx < probs.size(); idx++)
	{
	    probCur = probs[idx](pt.y, pt.x);

	    if(probCur < EPSILON) // discard probabilities close to zero
		continue;
	    
	    probSum += probCur;
	    cumProb[probSum] = idx + 1;
	}
	
	// choose an object based on the accumulated probability
	return cumProb.upper_bound(drand(0, probSum))->second;
    }
    
    std::vector<TransHyp*> getWorkingQueue(std::map<jp::id_t, std::vector<TransHyp>>& hypMap, int maxIt)
    {
	std::vector<TransHyp*> workingQueue;
      
	for(auto it = hypMap.begin(); it != hypMap.end(); it++)
	for(int h = 0; h < it->second.size(); h++)
	    if(it->second.size() > 1 || it->second[h].refSteps < maxIt) //exclude a hypothesis if it is the only one remaining for an object and it has been refined enough already
		workingQueue.push_back(&(it->second[h]));

	return workingQueue;
    }
    
    float estimatePose(
	const std::vector<jp::img_stat_t>& probs, 
	const std::vector<jp::RegressionTree<jp::feature_t>>& forest, 
	const std::vector<jp::img_leaf_t>& leafImgs,
	const std::vector<std::vector<cv::Point3f>>& bb3Ds)
    {
	GlobalProperties* gp = GlobalProperties::getInstance();
      
	int pnpMethod = CV_P3P; // pnp algorithm to be used to calculate initial poses from 4 correspondences
	float minDist2D = 10; // in px, initial pixel coordinates sampled to generate a hypothesis should be at least this far apart (for stability)
	float minDist3D = 10; // in mm, initial object coordinates sampled to generate a hypothesis should be at least this far apart (for stability)
	float minDepth = 300; // when estimating the seach radius for the initial pixel correspondences its assumed that the object cannot be nearer than this (in mm)
	float minArea = 400; // a hypothesis covering less projected area (2D bounding box) can be discarded (too small to estimate anything reasonable)

	//set parameters, see documentation of GlobalProperties
	int maxIterations = gp->tP.ransacMaxDraws;
	float inlierThreshold2D = gp->tP.ransacInlierThreshold2D;
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

	// create samplers for choosing pixel positions according to probability maps
	std::vector<Sampler2D> samplers;
	createSamplers(samplers, probs, imageWidth, imageHeight);

	// hold for each object a list of pose hypothesis, these are optimized until only one remains per object
	std::map<jp::id_t, std::vector<TransHyp>> hypMap;
	std::map<jp::id_t, unsigned> drawMap; // holds for each object the number of hypothesis drawn including the ones discarded for constrain violation
	
	float ransacTime = 0;
	StopWatch stopWatch;
	
	// sample initial pose hypotheses
	#pragma omp parallel for
	for(unsigned h = 0; h < ransacIterations; h++)
	{  
	    for(unsigned i = 0; i < maxIterations; i++)
	    {
		// 2D pixel - 3D object coordinate correspondences
		std::vector<cv::Point2f> points2D;
		std::vector<cv::Point3f> points3D;
	      
		cv::Rect bb2D(0, 0, imageWidth, imageHeight); // initialize 2D bounding box to be the full image
		
		// sample first point and choose object ID
		cv::Point2f pt1 = samplers[0].drawInRect(bb2D);
		jp::id_t objID = drawObjID(pt1, probs);

		if(objID == 0) continue;
		
		#pragma omp critical
		{
		    drawMap[objID]++;
		}
		
		// sample first correspondence
		samplePoint(objID, points3D, points2D, pt1, forest, leafImgs, minDist2D, minDist3D);
		
		// set a sensible search radius for other correspondences and update 2D bounding box accordingly
		float searchRadius = (gp->fP.focalLength * getMaxDist(bb3Ds[objID-1], points3D[0]) / minDepth) / 2; // project the object 3D bb into the image under a worst case (i.e. very small) distance to the camera
		searchRadius *= 0.3; // decreasing the sample window by a certain amaount so chance of pixels drawn of the same object increases, value is heuristic
		
		int minX = clamp(points2D[0].x - searchRadius, 0, imageWidth - 1);
		int maxX = clamp(points2D[0].x + searchRadius, 0, imageWidth - 1);
		int minY = clamp(points2D[0].y - searchRadius, 0, imageHeight - 1);
		int maxY = clamp(points2D[0].y + searchRadius, 0, imageHeight - 1);

		bb2D = cv::Rect(minX, minY, (maxX - minX + 1), (maxY - minY + 1));

		// sample other points in search radius, discard hypothesis if minimum distance constrains are violated
		if(!samplePoint(objID, points3D, points2D, samplers[objID].drawInRect(bb2D), forest, leafImgs, minDist2D, minDist3D))
		    continue;
		
		if(!samplePoint(objID, points3D, points2D, samplers[objID].drawInRect(bb2D), forest, leafImgs, minDist2D, minDist3D))
		    continue;
		
		if(!samplePoint(objID, points3D, points2D, samplers[objID].drawInRect(bb2D), forest, leafImgs, minDist2D, minDist3D))
		    continue;

		// check for degenerated configurations
		if(pointLineDistance(points3D[0], points3D[1], points3D[2]) < minDist3D) continue;
		if(pointLineDistance(points3D[0], points3D[1], points3D[3]) < minDist3D) continue;
		if(pointLineDistance(points3D[0], points3D[2], points3D[3]) < minDist3D) continue;
		if(pointLineDistance(points3D[1], points3D[2], points3D[3]) < minDist3D) continue;

		// reconstruct camera
		jp::cv_trans_t trans;
		cv::solvePnP(points3D, points2D, camMat, cv::Mat(), trans.first, trans.second, false, pnpMethod);
		
		std::vector<cv::Point2f> projections;
		cv::projectPoints(points3D, trans.first, trans.second, camMat, cv::Mat(), projections);
		
		// check reconstruction, 4 sampled points should be reconstructed perfectly
		bool foundOutlier = false;
		for(unsigned j = 0; j < points2D.size(); j++)
		{
		    if(cv::norm(points2D[j] - projections[j]) < inlierThreshold2D) continue;
		    foundOutlier = true;
		    break;
		}
		if(foundOutlier) continue;	    
		
		// create a hypothesis object to store meta data
		TransHyp hyp(objID, trans);
		
		// update 2D bounding box
		hyp.bb = getBB2D(imageWidth, imageHeight, bb3Ds[objID-1], camMat, hyp.pose);

		//check if bounding box collapses
		if(hyp.bb.area() < minArea)
		    continue;	    
		
		#pragma omp critical
		{
		    hypMap[objID].push_back(hyp);
		}

		break;
	    }
	}
	
	ransacTime += stopWatch.stop();
	std::cout << "Time after drawing hypothesis: " << ransacTime << "ms." << std::endl;

	// create a list of all objects where hypptheses have been found
	std::vector<jp::id_t> objList;
	std::cout << std::endl;
	for(std::pair<jp::id_t, std::vector<TransHyp>> hypPair : hypMap)
	{
	    std::cout << "Object " << (int) hypPair.first << ": " << hypPair.second.size() << " (drawn: " << drawMap[hypPair.first] << ")" << std::endl;
	    objList.push_back(hypPair.first);
	}
	std::cout << std::endl;

	// create a working queue of all hypotheses to process
	std::vector<TransHyp*> workingQueue = getWorkingQueue(hypMap, refIt);
	
	// main preemptive RANSAC loop, it will stop if there is max one hypothesis per object remaining which has been refined a minimal number of times
	while(!workingQueue.empty())
	{
	    // draw a batch of pixels and check for inliers, the number of pixels looked at is increased in each iteration
	    #pragma omp parallel for
	    for(int h = 0; h < workingQueue.size(); h++)
		countInliers2D(*(workingQueue[h]), forest, leafImgs, camMat, inlierThreshold2D, minArea, preemptiveBatch);
	    
	    // sort hypothesis according to inlier count and discard bad half
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
		updateHyp2D(*(workingQueue[h]), camMat, imageWidth, imageHeight, bb3Ds[workingQueue[h]->objID-1], maxPixels);
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
 
    inline void countInliers2D(
      TransHyp& hyp,
      const std::vector<jp::RegressionTree<jp::feature_t>>& forest,
      const std::vector<jp::img_leaf_t>& leafImgs,
      const cv::Mat& camMat,
      float inlierThreshold,
      int minArea,
      int pixelBatch)
    {
	// reset data of last RANSAC iteration
	hyp.inliers2D.clear();
	hyp.inliers3D.clear();
	hyp.inliersM.clear();
	hyp.inliers = 0;

	// abort if 2D bounding box collapses
	if(hyp.bb.area() < minArea) return;
	
	// obj coordinate predictions are collected first and then reprojected as a batch
	std::vector<cv::Point3f> points3D;
	std::vector<cv::Point2f> points2D;
	std::vector<cv::Point2f> projections;
	std::vector<const jp::mode_t*> modeList;	

	hyp.effPixels = 0; // num of pixels drawn
	hyp.maxPixels += pixelBatch; // max num of pixels to be drawn	
	
	int maxPt = hyp.bb.area(); // num of pixels within bounding box
	float successRate = hyp.maxPixels / (float) maxPt; // probability to accept a pixel
	
	std::mt19937 generator;
	std::negative_binomial_distribution<int> distribution(1, successRate); // lets you skip a number of pixels until you encounter the next pixel to accept
	
	for(unsigned ptIdx = 0; ptIdx < maxPt;)
	{
	    hyp.effPixels++;
	    
	    // convert pixel index back to x,y position
	    cv::Point2f pt2D(
		hyp.bb.x + ptIdx % hyp.bb.width, 
		hyp.bb.y + ptIdx / hyp.bb.width);
	        
	    // each tree in the forest makes one or more predictions, collect all of them
	    for(unsigned t = 0; t < forest.size(); t++)
	    {
		const std::vector<jp::mode_t>* modes = getModes(hyp.objID, pt2D, forest, leafImgs, t);
		for(int m = 0; m < modes->size(); m++)
		{
		    if(!jp::onObj(modes->at(m).mean)) continue;  // skip empty predictions
		    
		    // store 3D object coordinate - 2D pixel correspondence and associated distribution modes
		    points3D.push_back(cv::Point3d(modes->at(m).mean(0), modes->at(m).mean(1),modes->at(m).mean(2)));
		    points2D.push_back(pt2D);
		    modeList.push_back(&(modes->at(m)));
		}
	    }

	    // advance to the next accepted pixel
	    if(successRate < 1)
		ptIdx += std::max(1, distribution(generator));
	    else
		ptIdx++;
	}
	
	// reproject collected object coordinates
	if(points3D.empty()) return;
	cv::projectPoints(points3D, hyp.pose.first, hyp.pose.second, camMat, cv::Mat(), projections);
	
	// check for inliers
	for(unsigned p = 0; p < projections.size(); p++)
	{	    
	    if(cv::norm(points2D[p] - projections[p]) < inlierThreshold)
	    {
	        // keep inlier correspondences
		hyp.inliers2D.push_back(points2D[p]);
		hyp.inliers3D.push_back(points3D[p]);
		hyp.inliersM.push_back(modeList[p]);
		hyp.inliers++; // keep track of the number of inliers (correspondences might be thinned out for speed later)
	    }
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
	//choose equally probable
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
	std::vector<cv::Point3f>& pts3D, 
	std::vector<cv::Point2f>& pts2D, 
	const cv::Point2f& pt2D,
	const std::vector<jp::RegressionTree<jp::feature_t>>& forest, 
	const std::vector<jp::img_leaf_t>& leafImgs,
	float minDist2D,
	float minDist3D)
    {
	bool violation = false;
      
	if(getMinDist(pts2D, pt2D) < minDist2D) violation = violation || true; // check distance to previous pixel positions
      
	cv::Point3f pt3D = getMode(objID, pt2D, forest, leafImgs); // read out object coordinate
	
	if(getMinDist(pts3D, pt3D) < minDist3D) violation = violation || true; // check distance to previous object coordinates
	
	pts2D.push_back(pt2D);
	pts3D.push_back(pt3D);
	
	return !violation;
    }

    static double optEnergy(const std::vector<double> &pose, std::vector<double> &grad, void *data)
    {
	DataForOpt* dataForOpt=(DataForOpt*) data;
	
	// convert pose to our format
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
      
	// calculate the energy = negative log likelihood of the pose
	float score = -dataForOpt->ransac->likelihood2D(
	    dataForOpt->hyp->objID, 
	    &(dataForOpt->hyp->inliers2D),
	    &(dataForOpt->hyp->inliersM),
	    &trans);
	
	return score;
    }
  
    double refineWithOpt(
	TransHyp& hyp,
	int iterations) 
    {
	// set up optimization algorithm (gradient free)
	nlopt::opt opt(nlopt::LN_NELDERMEAD, 6); 
      
	// provide pointers to data and methods used in the energy calculation
	DataForOpt data;
	data.hyp = &hyp;
	data.ransac = this;

	// convert pose to rodriguez vector and translation vector in meters
	std::vector<double> vec(6);
	for(int i = 0; i < 6; i++)
	{
	    if(i > 2) 
		vec[i] = hyp.pose.second.at<double>(i-3, 0) / 1000.0;
	    else vec[i] = 
		hyp.pose.first.at<double>(i, 0);
	}
	
	// set optimization bounds 
	double rotRange = 10;
	rotRange *= PI / 180;
	double tRangeXY = 0.1;
	double tRangeZ = 0.5; // pose uncertainty is larger in Z direction
	
	std::vector<double> lb(6);
	lb[0] = vec[0]-rotRange; lb[1] = vec[1]-rotRange; lb[2] = vec[2]-rotRange;
	lb[3] = vec[3]-tRangeXY; lb[4] = vec[4]-tRangeXY; lb[5] = vec[5]-tRangeZ;
	opt.set_lower_bounds(lb);
      
	std::vector<double> ub(6);
	ub[0] = vec[0]+rotRange; ub[1] = vec[1]+rotRange; ub[2] = vec[2]+rotRange;
	ub[3] = vec[3]+tRangeXY; ub[4] = vec[4]+tRangeXY; ub[5] = vec[5]+tRangeZ;
	opt.set_upper_bounds(ub);
      
	std::cout << "Likelihood before refinement: ";

	std::cout << this->likelihood2D(
	    data.hyp->objID, 
	    &(data.hyp->inliers2D),
	    &(data.hyp->inliersM),
	    &(hyp.pose));

	std::cout << std::endl;
	
	// configure NLopt
	opt.set_min_objective(optEnergy, &data);
	opt.set_maxeval(iterations);

	// run optimization
	double energy;
	
	try
	{
	    nlopt::result result = opt.optimize(vec, energy);
	}
	catch(std::exception& e)
	{
	    std::cout << REDTEXT("Optimization threw an error!") << std::endl;
	}

	// read back optimized pose
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
  
    float likelihood2D(
	jp::id_t objID,
	std::vector<cv::Point2f>* inliers2D,
	std::vector<const jp::mode_t*>* inliersM,
	const jp::cv_trans_t* cvTrans) 
    {
	// accumulate likelihood over correspondences
	double likelihood2D = 0;
	
	// for stability limit the magnitude of the log likelihood for each correspondence (an outlier pixel could spoil the complete likelihood)
	double likelihood2DThreshMin = -100;
	double likelihood2DThreshMax = 100;
	// for stability discard covariance matrices which collapse (not enough samples during training)
	float covarThresh = 1000;

	cv::Mat_<float> camMat = GlobalProperties::getInstance()->getCamMat();
	
	// pose conversion
	jp::jp_trans_t jpTrans = jp::cv2our(*cvTrans);
	jpTrans.first = jp::double2float(jpTrans.first);
	
	// accumulate likelihood in different threads, combine in the end
	std::vector<double> localLikelihoods(omp_get_max_threads(), 0);
	
	#pragma omp parallel for
	for(int pt = 0; pt < inliers2D->size(); pt++) // iterate over correspondences
	{
	    int x = inliers2D->at(pt).x;
	    int y = inliers2D->at(pt).y;
	  
	    unsigned threadID = omp_get_thread_num();
	    const jp::mode_t* mode = inliersM->at(pt);

	    cv::Mat_<float> covar = mode->covar;
	    if(cv::determinant(covar) < covarThresh) // discard if covariance collapses
		continue;
	    
	    // read out center of the mode
	    cv::Mat_<float> obj(3, 1);
	    obj(0, 0) = mode->mean(0);
	    obj(1, 0) = mode->mean(1);
	    obj(2, 0) = mode->mean(2);
		
	    // convert mode center from object coordinates to camera coordinates
	    cv::Mat_<float> transObj = jpTrans.first * obj;
	    transObj(0, 0) += jpTrans.second.x;
	    transObj(1, 0) += jpTrans.second.y;
	    transObj(2, 0) += jpTrans.second.z;

	    // conver mode covariance from object coordinates to camera coordinates
	    cv::Mat_<float> transCovar = jpTrans.first * covar * jpTrans.first.t();
	    
	    // calculate likelihood, but clamp its magnitude
	    localLikelihoods[threadID] += std::min(std::max(likelihood2DThreshMin, std::log(likelihood(x, y, camMat, transObj, transCovar))), likelihood2DThreshMax);
	}
	
	// combine thread results
	for(unsigned t = 0; t < localLikelihoods.size(); t++)
	    likelihood2D += localLikelihoods[t];
	
	return likelihood2D;
    }
    
public:
    std::map<jp::id_t, TransHyp> poses; // Poses that have been estimated. At most one per object. Run estimatePose to fill this member.
};
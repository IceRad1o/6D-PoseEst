#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/highgui/highgui.hpp>

#include "../core/properties.h"
#include "../core/types.h"


//µ¥¸öpose²Ù×÷.trans. vec
class Hypothesis{
public:
	
        Hypothesis();
	
	Hypothesis(cv::Mat rot,cv::Point3d trans);
	
	Hypothesis(cv::Mat transform);
	
	Hypothesis(std::vector<std::pair<cv::Point3d,cv::Point3d>> points);
	
	Hypothesis(std::vector<double> rodVecAndTrans );
	
	Hypothesis(jp::info_t info);
	
	void refine(std::vector<std::pair<cv::Point3d,cv::Point3d>> points);

	void refine(cv::Mat& coV,cv::Point3d pointsA,cv::Point3d pointsB);
	
	cv::Point3d getTranslation() const;

	cv::Mat getRotation() const;
	
	cv::Mat getInvRotation() const;
	
	cv::Mat getRodriguesVector() const;
	
	std::vector<double> getRodVecAndTrans() const;
	
	void setRotation(cv::Mat rot);
	
	void setTranslation(cv::Point3d trans);
	
	cv::Mat getTransformation() const;
	
	Hypothesis getInv();

	Hypothesis operator*(const Hypothesis& other) const;
	
	Hypothesis operator/(const Hypothesis& other) const;
	
	cv::Point3d transform(cv::Point3d p, bool isNormal = false);

	cv::Point3d invTransform(cv::Point3d p);
	

	double calcAngularDistance(Hypothesis h);

	~Hypothesis();
	
	static std::pair<cv::Mat,cv::Point3d> calcRigidBodyTransform(cv::Mat& coV,cv::Point3d pointsA, cv::Point3d pointsB);
private:
	cv::Mat rotation; 
	cv::Mat invRotation;
	cv::Point3d translation;
	std::vector<std::pair<cv::Point3d,cv::Point3d>> points; // point correspondences used to calculated this pose, stored for refinement later
	
	static std::pair<cv::Mat,cv::Point3d> calcRigidBodyTransform(std::vector<std::pair<cv::Point3d, cv::Point3d>> points);
};
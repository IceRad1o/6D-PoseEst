#pragma once
 
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <iostream>
#include "Hypothesis.h"
#include "../core/util.h"

void inline loadPointCloud9Cols(std::string path, double scale, std::vector<cv::Point3d>& points, std::vector<cv::Point3d>& normals)
{
    double maxX, maxY, minX, minY, maxZ, minZ;
    std::ifstream myfile(path.c_str());
    
    double x, y, z, nx, ny, nz, r, g, b;
    std::string firstLine;
    
    if (myfile.is_open())
    {
	// remove first line
	std::getline(myfile, firstLine);
	
	bool first = true;
	int i = 0;
	
	while(myfile >> x >> y >> z >> nx >> ny >> nz >> r >> g >> b)
	{
	    if(i%20==0)
	    {
		//To rotations by 90 degrees; (difference in coordinate system definition)
		std::swap(x, z);
		x *= -1;
		
		std::swap(x, y);
		x *= -1;
		
		std::swap(nx, nz);
		nx *= -1;
		
		std::swap(nx, ny);
		nx *= -1;

		if(first)
		{
		    maxX = x; minX = x; maxY = y; minY = y; maxZ = z; minZ = z;
		}
	
		first = false;
		
		maxX = std::max(maxX, x); minX = std::min(minX, x);
		maxY = std::max(maxY, y); minY = std::min(minY, y);
		maxZ = std::max(maxZ, z); minZ = std::min(minZ, z);
		
		points.push_back(cv::Point3d(scale*x,scale*y, scale*z));
		normals.push_back(cv::Point3d(nx, ny, nz));
	    }
	    i++;
	}
	
	myfile.close();
    }
    else std::cout << "Unable to open point cloud file!"; 
    
    cv::Point3d center(scale * (minX + maxX) / 2.0, scale * (minY + maxY) / 2.0, scale * (minZ + maxZ) / 2.0);
    
    for(int i = 0; i < points.size(); i++)
    {
	points[i] = points[i] - center;
    }
}

std::vector<cv::Point3d> inline loadPointCloud(std::string path, double scale = 1)
{
    double maxX, maxY, minX, minY, maxZ, minZ;
    
    std::ifstream myfile(path.c_str());
    std::vector<cv::Point3d> result;
    
    double x,y,z;
    std::string firstLine;
    
    if(myfile.is_open())
    {
	bool first = true;
	int i = 0;
    
	while(myfile >> x >> y >> z)
	{
	    if(i%10==0)
	    {
		if(first)
		{
		    maxX = x; minX = x; maxY = y; minY = y; maxZ = z; minZ = z;
		}
	    
		first = false;

		maxX = std::max(maxX, x); minX = std::min(minX, x);
		maxY = std::max(maxY, y); minY = std::min(minY, y);
		maxZ = std::max(maxZ, z); minZ = std::min(minZ, z);
		
		result.push_back(cv::Point3d(scale * x, scale * y, scale * z));
	    }
	    i++;
	}
	
	myfile.close();
    }
    else std::cout << "Unable to open point cloud file!";  
    
    cv::Point3d center(scale * (minX + maxX) / 2.0, scale * (minY + maxY) / 2.0, scale * (minZ + maxZ) / 2.0);
    
    for(int i = 0; i < result.size(); i++)
    {
	result[i] = result[i] - center;
    }
    return result;
}

std::vector<cv::Point3d> inline loadPointCloud6Col(std::string path, double scale = 1)
{
    double maxX, maxY, minX, minY, maxZ, minZ;
    
    std::ifstream myfile(path.c_str());
    std::vector<cv::Point3d> result;
    
    double x, y, z, r, g, b;
    uchar prefix;
    std::string firstLine;
    
    if(myfile.is_open())
    {
	bool first = true;
	int i = 0;
	
	while(myfile >> prefix >> x >> y >> z >> r >> g >> b)
	{
	    if(i % 10 == 0)
	    {
		if(first)
		{
		    maxX = x; minX = x; maxY = y; minY = y; maxZ = z; minZ = z;
		}
		
		first = false;

		maxX = std::max(maxX, x); minX = std::min(minX, x);
		maxY = std::max(maxY, y); minY = std::min(minY, y);
		maxZ = std::max(maxZ, z); minZ = std::min(minZ, z);
		result.push_back(cv::Point3d(scale * x, scale * y, scale * z));
	    }
	    i++;
	}
	  
	myfile.close();
    }
    else std::cout << "Unable to open point cloud file!";  
    
    cv::Point3d center(scale * (minX + maxX) / 2.0, scale * (minY + maxY) / 2.0, scale * (minZ + maxZ) / 2.0);
    
    for(int i = 0; i < result.size(); i++)
    {
	result[i] = result[i] - center;
    }
    return result;
}

void inline loadPointCloudGeneric(std::string path, std::vector<cv::Point3d>& points, int maxPoints)
{
  // check first line of file to decide on the file format
  std::ifstream myfile(path.c_str());
  
  if(myfile.is_open())
  {
      std::string firstLine;
      std::getline(myfile, firstLine);
      myfile.close();
      
      std::vector<std::string> tokens = split(firstLine);
	  
      if(tokens.size() == 2) // hinterstoisser file format
      {
	  std::vector<cv::Point3d> normalCloud;
	  loadPointCloud9Cols(path, 10, points, normalCloud);
      }
      else if(tokens.size() == 3) // franks file format
      {
	  points = loadPointCloud(path, 1000);
      }
      else 
      {
	  points = loadPointCloud6Col(path, 1);
      }
  }
  
  if(points.size() < maxPoints) 
      return;
  
  std::vector<cv::Point3d> filteredPoints;
  for(unsigned i = 0; i < maxPoints; i++)
      filteredPoints.push_back(points[irand(0, points.size())]);

  points = filteredPoints;
}

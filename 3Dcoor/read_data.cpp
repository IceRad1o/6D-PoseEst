#include "read_data.h"
#include "util.h"

#include <fstream>
#include "png++/png.hpp"

namespace jp
{
    void readData(const std::string dFile, jp::img_depth_t& image)
    {
	png::image<depth_t> imgPng(dFile);
	image = jp::img_depth_t(imgPng.get_height(), imgPng.get_width());

	for(int x = 0; x < imgPng.get_width(); x++)
	for(int y = 0; y < imgPng.get_height(); y++)
	{
	    image(y, x) = (jp::depth_t) imgPng.get_pixel(x, y);
	}
    }

    void readData(const std::string bgrFile, jp::img_bgr_t& image)
    {
	png::image<png::basic_rgb_pixel<uchar>> imgPng(bgrFile);
	image = jp::img_bgr_t(imgPng.get_height(), imgPng.get_width());
	
	for(int x = 0; x < imgPng.get_width(); x++)
	for(int y = 0; y < imgPng.get_height(); y++)
	{
	    image(y, x)(0) = (uchar) imgPng.get_pixel(x, y).blue;
	    image(y, x)(1) = (uchar) imgPng.get_pixel(x, y).green;
	    image(y, x)(2) = (uchar) imgPng.get_pixel(x, y).red;
	}
    }
  
    void readData(const std::string bgrFile, const std::string dFile, jp::img_bgrd_t& image)
    {
	readData(bgrFile, image.bgr);
	readData(dFile, image.depth);
    }

    void readData(const std::string coordFile, jp::img_coord_t& image)
    {
	png::image<png::basic_rgb_pixel<unsigned short>> imgPng(coordFile);
	image = jp::img_coord_t(imgPng.get_height(), imgPng.get_width());
	
	for(int x = 0; x < imgPng.get_width(); x++)
	for(int y = 0; y < imgPng.get_height(); y++)
	{
	    image(y, x)(0) = (jp::coord1_t) imgPng.get_pixel(x, y).red;
	    image(y, x)(1) = (jp::coord1_t) imgPng.get_pixel(x, y).green;
	    image(y, x)(2) = (jp::coord1_t) imgPng.get_pixel(x, y).blue;
	}
    }
    
    bool readData(const std::string infoFile, jp::info_t& info)
    {
	std::ifstream file(infoFile);
	if(!file.is_open())
	{
	    info.visible = false;
	    return false;
	}
	
	std::string line;
	int lineCount = 0;
	std::vector<std::string> tokens;
	
	while(true)
	{
	    std::getline(file, line);
	    tokens = split(line);
	    
	    if(file.eof())	
	    {
		info.visible = false;
		return false;
	    }
	    if(tokens.empty()) continue;
	    lineCount++;
	    
	    if(lineCount == 3) info.name = tokens[0];
	    
	    if(tokens[0] == "occlusion:")
	    {
		std::getline(file, line);
		info.occlusion = (float) atof(line.c_str());
	    }	
	    else if(tokens[0] == "rotation:")
	    {
		info.rotation = cv::Mat_<float>(3, 3);
	      
		for(unsigned i = 0; i < 3; i++)
		{
		    std::getline(file, line);
		    tokens = split(line);
		    
		    for(unsigned j = 0; j < 3; j++) 
			info.rotation(i, j) = (float) atof(tokens[j].c_str());
		}
	    }
	    else if(tokens[0] == "center:")
	    {
		std::getline(file, line);
		tokens = split(line);	      
	      
		for(unsigned j = 0; j < 3; j++) 
		    info.center(j) = (float) atof(tokens[j].c_str());
	    }
	    else if(tokens[0] == "extent:" || tokens[0] == "extend:") // there was a typo in some of our files ;)
	    {
		std::getline(file, line);
		tokens = split(line);	      
	      
		for(unsigned j = 0; j < 3; j++) 
		    info.extent(j) = (float) atof(tokens[j].c_str());
		
		info.visible = true;
		return true;
	    }
    
	}
    }
}



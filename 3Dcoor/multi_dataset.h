#pragma once

#include "dataset.h"

namespace jp
{
  
  class MultiDataset 
    {
    public:
	
      
        MultiDataset(std::vector<Dataset> datasets) : datasets(datasets)
	{
	    // build dataset mapping (accumulated size to dataset index)
	    idxTable[0] = std::pair<size_t, size_t>(0, 0);
	    sizeSum = 0;
	    size_t lastSum = 0;

	    for(int i = 0; i < datasets.size(); i++)
	    {
		sizeSum += datasets[i].size();
		idxTable[sizeSum - 1] = std::pair<size_t, size_t>(lastSum, i);
		lastSum = sizeSum;
	    }

	    std::cout << "Training images found: " << sizeSum << std::endl;
	}

	
	size_t size() const 
	{ 
	    return sizeSum; 
	}

	size_t size(jp::id_t objID) const 
	{ 
	    return datasets[objID - 1].size(); 
	}

	jp::id_t getObjID(size_t i) const
	{
	    return datasets.at(getDB(i)).getObjID();
	}
	
	std::string getFileName(size_t i) const
	{
	    return datasets.at(getDB(i)).getFileName(getIdx(i));
	}
	
	void getBGR(size_t i, jp::img_bgr_t& img, bool noseg) const
	{
	    datasets.at(getDB(i)).getBGR(getIdx(i), img, noseg);
	}
	
	void getDepth(size_t i, jp::img_depth_t& img, bool noseg) const
	{
	    datasets.at(getDB(i)).getDepth(getIdx(i), img, noseg);
	}
	
	void getBGRD(size_t i, jp::img_bgrd_t& img, bool noseg) const
	{
	    datasets.at(getDB(i)).getBGRD(getIdx(i), img, noseg);
	}	
	
	void getSegmentation(size_t i, jp::img_id_t& seg) const
	{
	    datasets.at(getDB(i)).getSegmentation(getIdx(i), seg);
	}
	
	void getObj(size_t i, jp::img_coord_t& img) const
	{
	    return datasets.at(getDB(i)).getObj(getIdx(i), img);
	}
	
	void getEye(size_t i, jp::img_coord_t& img) const
	{
	    return datasets.at(getDB(i)).getEye(getIdx(i), img);
	}

	void getInfo(size_t i, jp::info_t& info) const
	{
	    datasets.at(getDB(i)).getInfo(getIdx(i), info);
	}

    private:
      
        size_t getDB(size_t i) const
	{
	    return (*idxTable.lower_bound(i)).second.second;
	}

	size_t getIdx(size_t i) const
	{
	    return i - (*idxTable.lower_bound(i)).second.first;
	}

	std::vector<Dataset> datasets; // list of object datasets

	// maps combined indices to datasets and their indices
	std::map<size_t, std::pair<size_t, size_t>> idxTable; 
	size_t sizeSum;
    };
}
#pragma once

#include "regression_tree.h"
#include "combined_feature.h"

namespace jp
{
    //forest prediction
    template <typename TFeature>
    class RegressionForestClassifier
    {
    public:
       
        RegressionForestClassifier(std::vector<RegressionTree<TFeature>>& forest) : forest(forest) {}
    
       
        void classify(const jp::img_data_t& data, std::vector<img_leaf_t>& leafImgs, const jp::img_depth_t* depth)
	{
	    if(leafImgs.size() != forest.size())
		leafImgs.resize(forest.size());
	  
	    for(unsigned t = 0; t < forest.size(); t++)
		forest[t].structure.getLeafImg(data, leafImgs[t], depth);
	}

	inline float getProbability(
	    const std::vector<jp::RegressionTree<jp::feature_t>>& forest, 
	    const std::vector<jp::img_leaf_t>& leafImgs,
	    unsigned treeIdx,
	    jp::id_t objID,
	    unsigned x,
	    unsigned y) const
	{
	    size_t leaf = leafImgs[treeIdx](y, x); // which tree leaf corresponds to the query pixel?
	    return forest[treeIdx].getObjPixels(leaf, objID) / (forest[treeIdx].getLeafPixels(leaf) + 1.f); // calculate probability from sample frequencies, +1 in the denominator for a robust estimate
	}

	void getObjsProb(
	    const std::vector<jp::RegressionTree<jp::feature_t>>& forest, 
	    const std::vector<jp::img_leaf_t>& leafImgs, 
	    std::vector<jp::img_stat_t>& probs) const
	{
	    int objectCount = GlobalProperties::getInstance()->fP.objectCount;

	    //init probability images
	    for(jp::id_t objID = 1; objID <= objectCount; objID++)
		probs[objID - 1] = jp::img_stat_t::ones(leafImgs[0].rows, leafImgs[0].cols);
	    jp::img_stat_t sumProbs = jp::img_stat_t::ones(leafImgs[0].rows, leafImgs[0].cols);
	    
	    #pragma omp parallel for
	    for(unsigned x = 0; x < leafImgs[0].cols; x++)
	    for(unsigned y = 0; y < leafImgs[0].rows; y++)
	    {
		bool allZero = true; //if all leaf indices are zero its most likely (sic!) that this pixel is undefined 
	 
		// accumulate probs for background
		for(unsigned treeIdx = 0; treeIdx < forest.size(); treeIdx++)
		    sumProbs(y, x) *= getProbability(forest, leafImgs, treeIdx, objectCount + 1, x, y);
	 
		// accumulate probs for objects
		for(jp::id_t objID = 1; objID <= objectCount; objID++)
		{
		    for(unsigned treeIdx = 0; treeIdx < forest.size(); treeIdx++)
		    {
			allZero = allZero && (leafImgs[treeIdx](y, x) == 0);
			probs[objID - 1](y, x) *= getProbability(forest, leafImgs, treeIdx, objID, x, y);
		    }
		    sumProbs(y, x) += probs[objID - 1](y, x);
		}

		//early out if undefined pixes
		if(allZero)
		{
		    for(jp::id_t objID = 1; objID <= objectCount; objID++)
			probs[objID - 1](y, x) = 0.f;
		}
		else
		{
		    for(jp::id_t objID = 1; objID <= objectCount; objID++)
			if(sumProbs(y, x) > 0) probs[objID - 1](y, x) /= sumProbs(y, x);
		}
	    }
	}	
	
    private:
      
	std::vector<RegressionTree<TFeature>>& forest; // Random forest to do classifications with.
    };
}
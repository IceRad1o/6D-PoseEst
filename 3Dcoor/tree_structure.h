#pragma once

#include <vector>
#include <ios>
#include <stack>
#include <list>
#include <queue>
#include <omp.h>

#include "../core/generic_io.h"
#include "training_samples.h"
#include "../core/image_cache.h"


namespace jp
{
  
    template <class feature_t>
    struct split_t
    {
	split_t() {}
	split_t(const feature_t& t, int left, int right) : left(left), right(right), feature(t) {}

	int left; 
	int right; 
	feature_t feature; 
    };  
   
    template <typename T> //¶þ²æÊ÷
    class TreeStructure
    {
    public:
	TreeStructure() {}

	bool isLeaf(int node) const 
	{ 
	    return node < 0; 
	}
	
	size_t child2Leaf(int child) const 
	{ 
	    return -child - 1; 
	}
		
	int leaf2Child(size_t leaf) const 
	{ 
	    return -leaf - 1; 
	}
	
	size_t getNodeCount() const
	{
	    return size() * 2 + 1;
	}

	size_t size() const 
	{ 
	    return splits.size(); 
	}

	bool empty() const
	{
	    return splits.empty();
	}
			
	int root() const 
	{
	    return splits.empty() ? -1 : 0; 
	}

	int addNode(const T& data)
	{
	    split_t<T> node(data, -1, -1);
	    splits.push_back(node);
	    return splits.size() - 1;
	}

	int getSplitChild(int split, bool direction) const
	{
	    return direction ? splits[split].right : splits[split].left;
	}

	void setSplitChild(int split, bool direction, int child)
	{
	    (direction ? splits[split].right : splits[split].left) = child;
	}      


	template <typename Test>
	size_t getLeaf(const Test& test) const
	{
	    int i = root();
	    while (!isLeaf(i))
		i = getSplitChild(i, test(splits[i].feature));
	    return child2Leaf(i);
	}

	void getLeafImg(const jp::img_data_t& data, img_leaf_t& leafs, const jp::img_depth_t* depth) const
	{
	    if((leafs.rows != data.seg.rows) || (leafs.cols != data.seg.cols))
		leafs = img_leaf_t(data.seg.rows, data.seg.cols);
	    
	    #pragma omp parallel for
	    for(unsigned x = 0; x < data.seg.cols; x++)
	    for(unsigned y = 0; y < data.seg.rows; y++)
	    {
    		float scale = 1.f;
		if(depth != NULL) // in case a depth channel is given, the patch is scaled according to the depth of the pixel
		{
		    jp::depth_t curDepth = depth->operator()(y, x);
		    if(curDepth > 0) scale = 1.f / curDepth * 1000.f; 
		}	      
	      
		size_t leaf = getLeaf([&](const T& test)
		{
		    return test(x, y, scale, data);
		});
		leafs(y, x) = leaf;
	    }	  
	}
	

	void store(std::ofstream& file) const
	{
	    jp::write(file, splits);
	}

	void restore(std::ifstream& file)
	{
	    jp::read(file, splits);
	}
	
	void print() const
	{
	    for(unsigned i = 0; i < splits.size(); i++)
	    {
		std::cout << "Node: " << std::endl;
		std::cout << "Left child: " << splits[i].left << std::endl;
		std::cout << "Right child: " << splits[i].right << std::endl;
		std::cout << "Feature: " << std::endl; 
		splits[i].data.print();
		std::cout << std::endl;
	    }
	}

	std::vector<split_t<T>> splits; // list of split nodes (inner nodes)
    };
 
    template <class TFeature>
    void write(std::ofstream& file, const typename jp::split_t<TFeature>& split)
    {
	write(file, split.left);
	write(file, split.right);
	write(file, split.feature);
    }
  
    template <class TFeature>
    void read(std::ifstream& file, typename jp::split_t<TFeature>& split)
    {
	read(file, split.left);
	read(file, split.right);
	read(file, split.feature);      
    }
 
    template <typename TFeatureSampler>
    class TreeTrainer
    {
    public:
	typedef typename TFeatureSampler::feature_t feature_t; // feature type to use 
	typedef jp::TreeStructure<feature_t> structure_t; // binary decision tree structure type

	struct leaf_stat_t
	{
	    leaf_stat_t() : terminated(false), splitIdx(-1), parent(-1), direction(false) {}
	    leaf_stat_t(int parent, bool direction) : terminated(false), splitIdx(-1), parent(parent), direction(direction) {}

	    int parent; 
	    bool direction; 
	    
	    int splitIdx; 
	    bool terminated; 
	};
	
	struct split_stat_t
	{
	    split_stat_t() : leafIdx(-1) {}
	    split_stat_t(int leafIdx) : leafIdx(leafIdx) {}
	  
	    int leafIdx;
	    std::vector<std::pair<unsigned, sample_t>> leafPixels; // list of pixels that arrived at this leaf
	    
	    feature_t feature; 
	    double score; 
	};
	
	TreeTrainer(const jp::ImageCache* imgCache, const TFeatureSampler& sampler) : 
	    imgCache(imgCache), sampler(sampler)
	{   
	    gp = GlobalProperties::getInstance();
	}


	bool trainingRound(structure_t& structure, std::vector<leaf_stat_t>& leafStats)
	{
	  
	    std::vector<split_stat_t> splitCandidates;

	    for(int leafIdx = 0; leafIdx < leafStats.size(); leafIdx++)
	    {

		if(!leafStats[leafIdx].terminated)
		{
		    leafStats[leafIdx].splitIdx = splitCandidates.size();
		    splitCandidates.push_back(split_stat_t(leafIdx));
		}
		else
		{
		    leafStats[leafIdx].splitIdx = -1;
		}
	    }
	    
	    if(splitCandidates.empty()) return false; 
	    
	    std::cout << std::endl << "Evaluating tree..." << std::endl;

	    for(unsigned i = 0; i < imgCache->dataCache.size(); i++)
	    {
		std::vector<sample_t> sampling(imgCache->sampleCounts[i]);
		const jp::img_depth_t* imgDepth = NULL;
		if(gp->fP.useDepth) imgDepth = &(imgCache->depthCache[i]);
		
		if(i < imgCache->bgPointer)
		    samplePixels(imgCache->sampleSegCache[i], imgCache->poseCache[i], sampling, imgDepth);
		else
		{
		    std::vector<sample_t> sampling1(imgCache->sampleCounts[i] / 2);
		    samplePixels(imgCache->sampleSegCache[i], imgCache->poseCache[i], sampling1, imgDepth);
		    std::vector<sample_t> sampling2(imgCache->sampleCounts[i] / 2);
		    samplePixels(imgCache->objProbCache[i], imgCache->poseCache[i], sampling2, imgDepth);
		    sampling.clear();
		    sampling.insert(sampling.end(), sampling1.begin(), sampling1.end());
		    sampling.insert(sampling.end(), sampling2.begin(), sampling2.end());
		}
		
		for(auto var = sampling.begin(); var != sampling.end(); ++var)
		{
		    sample_t px = *var;
		  
		    size_t leafIndex = structure.getLeaf([&](const feature_t& feature) 
		    { 
			return feature(px.x, px.y, px.scale, imgCache->dataCache[i]); 
		    });

		    int splitIdx = leafStats[leafIndex].splitIdx;
    
		    if(splitIdx >= 0) 
			splitCandidates[splitIdx].leafPixels.push_back(std::pair<unsigned, sample_t>(i, px));
		}
	    }
	    
	    std::cout << "Processing " << splitCandidates.size() << " leafs:" << std::endl;
	  
	    std::vector<feature_t> features = sampleFromRandomPixels(
		imgCache->dataCache, 
		gp->fP.featureCount, 
		sampler);

	    for(unsigned splitIdx = 0; splitIdx < splitCandidates.size(); splitIdx++)
	    {
		std::cout << ".";
		std::cout.flush();

		std::vector<float> featureScores(gp->fP.featureCount, 0.f);

		#pragma omp parallel for
		for(unsigned featureIndex = 0; featureIndex < features.size(); featureIndex++)
		{	
		    histogram_t histLeft(gp->fP.getLabelCount(), 0);
		    histogram_t histRight(gp->fP.getLabelCount(), 0);
		  
		    for(unsigned pxIdx = 0; pxIdx < splitCandidates[splitIdx].leafPixels.size(); pxIdx++)
		    {
			unsigned imgID = splitCandidates[splitIdx].leafPixels[pxIdx].first;
			sample_t pixelPos = splitCandidates[splitIdx].leafPixels[pxIdx].second;

			jp::label_t label = (imgID >= imgCache->bgPointer) ? 0 : imgCache->gtCache[imgID](pixelPos.y, pixelPos.x);
			bool response = features[featureIndex](pixelPos.x, pixelPos.y, pixelPos.scale, imgCache->dataCache[imgID]);
			
			if(response)
			    histRight[label]++;
			else
			    histLeft[label]++;
		    }

		    double score = 0.0;
		    if(histogramTotal(histLeft) > gp->fP.minSamples && histogramTotal(histRight) > gp->fP.minSamples)
			score = informationGain(histLeft, histRight);
		    
		    featureScores[featureIndex] = score;
		}

		double bestScore = 0.0;
		unsigned bestFeature = 0;		
		
		for(unsigned s = 0; s < featureScores.size(); s++)
		{
		    if(featureScores[s] > bestScore)
		    {
			bestScore = featureScores[s];
			bestFeature = s;
		    }		  
		}
		
		splitCandidates[splitIdx].feature = features[bestFeature];
		splitCandidates[splitIdx].score = bestScore;
	    }
	    

	    std::cout << std::endl << "Splitting leaf nodes ..." << std::endl;
	    double minScore = 0.0;
	    bool newLayer = false;
	    
	    for(unsigned splitIdx = 0; splitIdx < splitCandidates.size(); splitIdx++)
	    {
		int leafIdx = splitCandidates[splitIdx].leafIdx;
		int parent = leafStats[leafIdx].parent;
		bool direction = leafStats[leafIdx].direction;
		
		if(splitCandidates[splitIdx].score > minScore)
		{
		    size_t leafLeft = structure.empty()
			? 0 
			: structure.child2Leaf(structure.getSplitChild(parent, direction));
		    size_t leafRight = leafStats.size();
		      
		 
		    int split = structure.addNode(splitCandidates[splitIdx].feature);
		    structure.setSplitChild(split, false, structure.leaf2Child(leafLeft));
		    structure.setSplitChild(split, true, structure.leaf2Child(leafRight));		    
		    	    
	
		    if(structure.size() - 1 > 0) structure.setSplitChild(parent, direction, split);
		    
		
		    leafStats[leafLeft] = leaf_stat_t(split, false);
		    leafStats.push_back(leaf_stat_t(split, true));
		    		    
		    featuresChosen[splitCandidates[splitIdx].feature.getType()]++;
		    if(!newLayer)
		    {
			newLayer = true;
			treeDepth++;
		    }
		}
		else
		{
		    size_t leafIndex = structure.empty()
			? 0 
			: structure.child2Leaf(structure.getSplitChild(parent, direction));
		    leafStats[leafIndex].terminated = true;
		}
	    }
	    std::cout << std::endl;
	    
	    std::cout << "Tree depth: " << treeDepth << std::endl;
	    return (treeDepth < gp->fP.maxDepth); // continue if max depth has not been reached
	}


	void train(jp::TreeStructure<feature_t>& structure)
	{

	    featuresChosen.clear();
	    treeDepth = 0;
	    
 
	    structure.splits.clear();
	    std::vector<leaf_stat_t> leafStats(1, leaf_stat_t());
	    
	  
	    while(true)
	    {
		if(!trainingRound(structure, leafStats))
		    break;
		std::cout << "Tree now has " << structure.getNodeCount() << " nodes." << std::endl;
	    }

	    std::cout << std::endl << "------------------------------------------------" << std::endl;
	    
	    std::cout << std::endl << "Features chosen: " << std::endl;
	    for(auto pair : featuresChosen)
	    {
		if(pair.first == 0) std::cout << "Color: " << pair.second << std::endl;
		else if(pair.first == 9) std::cout << "Abs Cell: " << pair.second << std::endl;
		else if(pair.first == 10) std::cout << "Abs Cell: " << pair.second << std::endl;
		else std::cout << "Unknown: " << pair.second << std::endl;
	    }
	}

    private:


      inline unsigned long histogramTotal(const histogram_t& h) const
	{
	  long total = 0;
	  for(label_t i = 0; i < h.size(); i++)
	      total += h[i];
	  return total;
	}
      
      inline double histogramEntropy(const histogram_t& h) const
	{
	    double entropy = 0.0;
	    double scale = 1.0 / (histogramTotal(h) + EPS * h.size());
	    for(label_t label = 0; label < h.size(); label++)
	    {
		double f = scale * (EPS + h[label]);
		entropy -= f * std::log(f);
	    }
	    return entropy;
	}
	
	double informationGain(const histogram_t& left, const histogram_t& right) const
	{
	    assert(left.size() == right.size());
	    unsigned long totalLeft = histogramTotal(left);
	    unsigned long totalRight = histogramTotal(right);
	    unsigned long total = totalLeft + totalRight;
	    if(totalLeft == 0 || totalRight == 0) return 0.0;

	    histogram_t parent(left.size(), 0);
	    for(label_t label = 0; label < parent.size(); label++)
		parent[label] = left[label] + right[label];

	    double inv = 1.0 / total;
	    return histogramEntropy(parent) - inv * 
		(histogramEntropy(left) * totalLeft + histogramEntropy(right) * totalRight);
	}
      
	const jp::ImageCache* imgCache; 
	const TFeatureSampler sampler; 
	const GlobalProperties* gp; 
	
	std::map<int, int> featuresChosen; 
	unsigned treeDepth; 
    };    
}
#pragma once

#include "types.h"
#include "segm/ms.h"
#include "generic_io.h"
#include "tree_structure.h"

// regression tree

namespace jp
{
    template <typename TFeature>
    class RegressionTree
    {
    public:
	typedef typename jp::TreeStructure<TFeature> structure_t; 
	typedef std::vector<std::vector<jp::mode_t>> leaf_mode_t; 
	typedef std::vector<unsigned> leaf_count_t; 

	RegressionTree() {}

	unsigned getObjPixels(size_t leaf, id_t objID) const
	{
	    return leafCounts[leaf][objID - 1];
	}
	
	unsigned getLeafPixels(size_t leaf) const
	{
	    return leafSums[leaf];
	}

	const std::vector<jp::mode_t>* getModes(size_t leaf, id_t objID) const
	{
	    return &(leafModes[leaf][objID - 1]);
	}

	void store(std::ofstream& file) const
	{
 	    structure.store(file);
	    write(file, leafModes);
	    write(file, leafCounts);
	    write(file, leafSums);
	}
	
	void restore(std::ifstream& file)
	{
	    structure.restore(file);
	    read(file, leafModes);
	    read(file, leafCounts);
	    read(file, leafSums);
	}
	
	
	void print() const
	{
	    int leafLimit = 10;
	    int objLimit = 3;
	    int modeLimit = 1;
	  
	    std::cout << REDTEXT("Structure: ") << std::endl << std::endl;
	    structure.print();
	    
	    for(unsigned leafIdx = 0; leafIdx < std::min<int>(leafModes.size(), leafLimit); leafIdx++)
	    {
		std::cout << REDTEXT("Leaf mode " << leafIdx << ": ") << std::endl;
		std::cout << "Leaf sum: " << leafSums[leafIdx] << std::endl << std::endl;
	      
		for(unsigned objIdx = 0; objIdx < std::min<int>(leafModes[leafIdx].size(), objLimit); objIdx++)
		{
		    std::cout << YELLOWTEXT("Object " << objIdx << ": ") << std::endl << std::endl;
		    
		    for(unsigned modeIdx = 0; modeIdx < std::min<int>(leafModes[leafIdx][objIdx].size(), modeLimit); modeIdx++)
		    {
			jp::mode_t mode = leafModes[leafIdx][objIdx][modeIdx];
			std::cout << "Mean: " << mode.mean << std::endl;
			std::cout << "Covariance: " << mode.covar << std::endl;
			std::cout << "Support: " << mode.support << std::endl;
		    }
		    std::cout << "Leaf count: " << leafCounts[leafIdx][objIdx] << std::endl << std::endl;
		}
		std::cout << YELLOWTEXT("...") << std::endl << std::endl;
	    }
	    
	    std::cout << REDTEXT("...") << std::endl;
	    std::cout << "==================================================" << std::endl << std::endl;
	}

	size_t getLeafCount() const
	{
	    return leafModes.size();
	}

	void calculateLeafSums()
	{
	    leafSums = std::vector<unsigned>(leafCounts.size(), 0);
	    
	    for(unsigned i = 0; i < leafSums.size(); i++)
	    for(id_t objID = 1; objID <= leafCounts[i].size(); objID++)
		leafSums[i] += leafCounts[i][objID - 1];
	}
	
	structure_t structure; // Tree structure.
	std::vector<leaf_mode_t> leafModes; // List of leaf distributions.
	std::vector<leaf_count_t> leafCounts; // List of leaf samples per object.
	std::vector<unsigned> leafSums; // List of total leaf samples.
    };


    template <class TFeature>
    void write(std::ofstream& file, const RegressionTree<TFeature>& tree)
    {
	tree.store(file);
    }

    template <class TFeature>

    void read(std::ifstream& file, RegressionTree<TFeature>& tree)
    {
	tree.restore(file);
    }
    
 
    void write(std::ofstream& file, const jp::mode_t& mode);
    
    void read(std::ifstream& file, jp::mode_t& mode);

    template <typename TFeature>
    class RegressionTreeTrainer
    {
    public:
     
        RegressionTreeTrainer(const jp::ImageCache* imgCache) : imgCache(imgCache)
	{
	    gp = GlobalProperties::getInstance();
	}
	
	void train(RegressionTree<TFeature>& rTree)
	{
	    if(rTree.structure.empty())
		std::cout << REDTEXT("ERROR: The tree structure has to be trained first!") << std::endl;
	  
	    typedef std::vector<std::vector<jp::coord3_t>> leaf_distribution_t; // per object a list of 3d points (object coordinate samples)

	    std::vector<leaf_distribution_t> leafDistributions(rTree.structure.size() + 1); // one distribution per leaf
	    rTree.leafModes.resize(rTree.structure.size() + 1); // initialize with number of leafs in the structure
	    rTree.leafCounts.resize(rTree.structure.size() + 1);  // initialize with number of leafs in the structure

	    // initialize leaf data
	    for(unsigned i = 0; i < rTree.structure.size() + 1; i++)
	    {
		leafDistributions[i] = leaf_distribution_t(gp->fP.objectCount);
		rTree.leafModes[i] = typename RegressionTree<TFeature>::leaf_mode_t(gp->fP.objectCount);
		rTree.leafCounts[i] = typename RegressionTree<TFeature>::leaf_count_t(gp->fP.objectCount + 1, 0);
	    }

	    std::cout << "Processing images: ";

	    for(unsigned i = 0; i < imgCache->dataCache.size(); i++)
	    {
		std::cout << ".";
		std::cout.flush();
		
		// load depth if available (used to scale training samples)
		const jp::img_depth_t* imgDepth = NULL;
		if(gp->fP.useDepth) imgDepth = &(imgCache->depthCache[i]);
		
		// draw training samples
		std::vector<sample_t> samples(imgCache->sampleCounts[i] * gp->fP.trainingPixelFactorRegression);
		samplePixels(imgCache->sampleSegCache[i], imgCache->poseCache[i], samples, imgDepth);
		
		// push all samples through the tree structure and collect them in the leafs
		for(auto sample_it = samples.begin(); sample_it != samples.end(); ++sample_it)
		{
		    sample_t px = *sample_it;
		    
		    // look up the leaf the current sample ends up in
		    size_t leafIdx = rTree.structure.getLeaf([&](const TFeature& test)
		    {
			return test(px.x, px.y, px.scale, imgCache->dataCache[i]);
		    });
		    
		    if(i < imgCache->bgPointer) // check for object or background
		    {
			jp::id_t objID = imgCache->idCache[i];
			jp::coord3_t objPt = imgCache->objCache[i](px.y, px.x);
			
			if(objID > 0 && onObj(objPt)) 
			{
			    // store current sample for specific object
			    leafDistributions[leafIdx][objID - 1].push_back(objPt);
			    rTree.leafCounts[leafIdx][objID - 1]++;
			}
			else // image shows object but sample could still belong to background
			    rTree.leafCounts[leafIdx][gp->fP.objectCount]++;
		    }
		    else //background image
			rTree.leafCounts[leafIdx][gp->fP.objectCount]++;
		}
	    }
	    
	    std::cout << std::endl << std::endl << "Starting MeanShift ..." << std::endl;
	    
	    // fit distributions for each object in each leaf
	    #pragma omp parallel for
	    for(unsigned i = 0; i < leafDistributions.size(); i++) // iteration over leafs
	    for(unsigned j = 0; j < leafDistributions[i].size(); j++) // interation over objects
	    {
	        // omit samples if too many (to reduce training time)
		filterLeafDistributions(leafDistributions[i][j], gp->fP.maxLeafPoints); 
		
		// fit and store GMM distribution using mean-shift
		std::vector<jp::mode_t> currentMode;
		std::vector<jp::mode_t> allModes = modeSeeking(leafDistributions[i][j]);
		
		if(allModes.empty())
		    currentMode.push_back(jp::mode_t()); // store an empty mode if no modes have been found
		else
		    currentMode = allModes;
		  
		rTree.leafModes[i][j] = currentMode;
	    }

	    rTree.calculateLeafSums();
	    std::cout << "Done." << std::endl << std::endl;
	    
	}
    
    private:

        inline void filterLeafDistributions(std::vector<jp::coord3_t>& dist, unsigned maxCount)
	{
	    if(dist.size() <= maxCount) return;
	  
	    std::vector<jp::coord3_t> filteredDist;
	    
	    for(unsigned i = 0; i < maxCount; i++)
	    {
		int idx = irand(0, dist.size());
		filteredDist.push_back(dist[idx]);
	    }
	    
	    dist = filteredDist;
	}
      
        inline bool compareModes(jp::coord3_t mode1, double* mode2, int dim, double tolerance)
	{
	    for(int i = 0; i < dim; i++)
		if(abs(mode1[i] - mode2[i]) > tolerance) return false;
	    return true;
	}
      

        jp::mode_t makeMode(const jp::coord3_t& m, const std::vector<jp::coord3_t>& support)
	{
	    jp::mode_t modus;
	    
	    modus.support = support.size();
	    
	    //calculate covariance matrix of data points with pre-calculated mean
	    cv::Mat_<float> covarData(support.size(), 3);
	    cv::Mat_<float> covarMean(1, 3);
	    covarMean(0, 0) = m[0];
	    covarMean(0, 1) = m[1];
	    covarMean(0, 2) = m[2];
	    
	    cv::Mat_<float> covarMatrix;
	    
	    for(unsigned i = 0; i < support.size(); i++)
	    for(unsigned j = 0; j < 3; j++)
		covarData(i, j) = (float) support[i][j];
	    
	    cv::calcCovarMatrix(covarData, covarMatrix, covarMean, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE | CV_COVAR_USE_AVG, CV_32F);
	    
	    modus.mean[0] = covarMean(0, 0);
	    modus.mean[1] = covarMean(0, 1);
	    modus.mean[2] = covarMean(0, 2);
	    
	    modus.covar = covarMatrix;
	    
	    return modus;
	}
      
	std::vector<jp::mode_t> modeSeeking(const std::vector<jp::coord3_t>& distribution)
	{
	    int n = (int) distribution.size(); // number of samples
	    int dim = 3; // dimensionality of samples
	    double tolerance = 5.0; // tolerance to detect equal modes

	    if(n == 0) return std::vector<jp::mode_t>();

	   
	    MeanShift ms;
	    
	    kernelType kT[] = {Gaussian}; // kernel type
	    float h[] = {gp->fP.meanShiftBandWidth}; // bandwidth in millimeters
	    int P[] = {dim}; // subspace definition, we have only 1 space of dimension 3
	    int kp = 1; // subspace definition, we have only 1 space of dimension 3
	    ms.DefineKernel(kT, h, P, kp);
	
	    // set input data
	    float* data = new float[n*dim];
    
	    for(int i = 0; i < n; i++)
	    for(int j = 0; j < dim; j++)
	    {
		data[i * dim + j] = distribution[i][j];
	    }
	
	    ms.DefineInput(data, n, dim);
    
	    // find modes
	    std::vector<jp::coord3_t> tempModes; // initial list of modes (including very small ones)
	    std::vector<std::vector<jp::coord3_t>> support; // list of support
		    
	    double* mode = new double[dim]; // mode we want to find
	    double* point = new double[dim]; // point where we start the search (= sample point)
	    jp::coord3_t newSuppPoint; // sample point coordinate to be stored in the support ground of its mode
	    
	    for(int i = 0; i < n; i++)
	    {
	        // convert data
		for(int j = 0; j < dim; j++)
		{
		    mode[j] = 0.f;
		    point[j] = distribution[i][j];
		    newSuppPoint[j]  = distribution[i][j]; 
		}

		ms.FindMode(mode, point); // gets the mode center the sample wanders of to

		// assign the mode according to the mode center found or create a new one
		bool modeFound = false; 

		for(unsigned j = 0; j < tempModes.size(); j++) // iterate over all modes previously found
		{
		    if(compareModes(tempModes[j], mode, dim, tolerance))
		    {
			// found current mode in the list, add current sample to its support
			support[j].push_back(newSuppPoint);
			modeFound = true;
			break;
		    }
		}
		if(!modeFound)
		{
		    // mode not encountered before, create a new one with current sample as support
		    jp::coord3_t newMode;
		    for(int j = 0; j < dim; j++) newMode[j] = (jp::coord1_t) mode[j];

		    tempModes.push_back(newMode);
		    std::vector<jp::coord3_t> newSupport;
		    newSupport.push_back(newSuppPoint);
		    support.push_back(newSupport);
		}
	    }
    
	    // clean up
	    delete [] mode;
	    delete [] point;
	    delete [] data;
    
	    float supportMinRatio = 0.5f; // relative mode support size (relative to the largest support) under which modes are discarded
	    int supportMin = 10; // absolute minimal threshold of support size (smaller modes are discarded)
	    
	    // determine mode with largest support
	    int bestSupport = 0;
	    
	    for(int s = 0; s < support.size(); s++)
		if(support[s].size() > bestSupport) 
		    bestSupport = support[s].size();

	    // calculate effective support minimum frow absolute and relative threshold
	    supportMin = std::max(supportMin, (int) (supportMinRatio * bestSupport));
	    std::vector<jp::mode_t> modes; // final list of modes

	    for(int s = 0; s < support.size(); s++)
	    {
		if(support[s].size() <= supportMin) // mode to small discard
		    continue;
		
		modes.push_back(makeMode(tempModes[s], support[s])); // fit a GMM component and store this mode
	    }

	    // sort by support, largest support first
	    std::sort(modes.begin(), modes.end()); 
	    std::reverse(modes.begin(), modes.end());

	    return modes;
	}
      
	const jp::ImageCache* imgCache; // pointer to training data
	const GlobalProperties* gp; // pointer to properties
    };
}
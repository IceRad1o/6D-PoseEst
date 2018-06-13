#include <fstream>

#include "properties.h"
#include "multi_dataset.h"
#include "combined_feature.h"
#include "features.h"
#include "image_cache.h"

#include "regression_tree.h"
#include "regression_classifier.h"
#include "auto_context.h"
#include "util_display.h"
#include "stop_watch.h"


 */
jp::sampler_outer_t createSampler(
    float weightDABGR,
    float weightAbsCoord,
    float weightAbsCell)
{
    GlobalProperties* gp = GlobalProperties::getInstance();  
  
    // create samplers for each feature type
    jp::FeatureSamplerDABGR samplerDABGR(gp->fP.maxOffset);
    jp::FeatureSamplerAbsCoord samplerAbsCoord(gp->fP.maxOffset, gp->fP.objectCount);
    jp::FeatureSamplerAbsCell samplerAbsCell(gp->fP.maxOffset, gp->fP.objectCount);  
  

    float weightInner1 = ((weightAbsCoord + weightAbsCell) == 0) 
	? 0 : weightAbsCoord / (weightAbsCoord + weightAbsCell);
    float weightOuter = ((weightDABGR + weightAbsCoord + weightAbsCell) == 0) 
	? 0 : (weightDABGR) / (weightDABGR + weightAbsCoord + weightAbsCell);	
    
    jp::sampler_inner_t1 innerSampler1(samplerAbsCoord, samplerAbsCell, weightInner1);
    return jp::sampler_outer_t(samplerDABGR, innerSampler1, weightOuter);  
}

int main(int argc, const char* argv[])
{
    GlobalProperties* gp = GlobalProperties::getInstance();
    gp->parseConfig(); // read the default config file and set parameters accordingly
    gp->parseCmdLine(argc, argv); // parse parameters given via command line
    gp->fP.training = true; // some code (e.g. the features) work differently in training mode

    std::string dataDir = "./training/"; // directory of the object training data
    std::string bgDir = "./background/"; // directory of background images (optional), used a negative class

    std::cout << std::endl << gp->getFileName(0) << "\n_____\n";

    // read training sub folders
    std::vector<std::string> trainingSets = getSubPaths(dataDir);
    std::vector<std::string> backgroundSets = getSubPaths(bgDir);

    gp->fP.objectCount = trainingSets.size();
   
    //set feature weights for the first forest in the stack (no auto-context yet)
    float weightDABGR = gp->fP.fBGRWeight;
    float weightAbsCoord = 0;
    float weightAbsCell = 0;
   
    // create a feature sampler
    jp::sampler_outer_t sampler = createSampler(weightDABGR, weightAbsCoord, weightAbsCell);
    
    // create data sets for each object
    std::vector<jp::Dataset> trainingObjData;

    for(int i = 0; i < trainingSets.size(); i++)
	trainingObjData.push_back(jp::Dataset(trainingSets[i], i + 1));

    // combine data sets to one big data set
    jp::MultiDataset trainingData(trainingObjData);
        
    // create data set of background images
    jp::Dataset bgSet;
    if(!backgroundSets.empty()) 
	bgSet = jp::Dataset(backgroundSets[0], gp->fP.objectCount + 1);

    // pre-load all training data for faster training
    jp::ImageCache imgCache;
    imgCache.reload(trainingData, bgSet);
        
    // set number of auto-context layers
    int trainingRounds = gp->fP.acPasses;
    StopWatch stopWatch;
    
    std::vector<std::vector<jp::RegressionTree<jp::feature_t>>> rForests(trainingRounds); // list of forests (one forest per auto-context layer)
    std::vector<jp::RegressionForestClassifier<jp::feature_t>*> rClassifiers(trainingRounds); // classifier for each forest
    
    // main training loop
    for(unsigned trainingRound = 0; trainingRound < trainingRounds; trainingRound++)
    {
	gp->fP.training = true;
      
	if(trainingRound == 1) // starting from the second forest in the auto-context stack, auto context features are activated
	{
	    weightAbsCell = gp->fP.fACCWeight;
	    weightAbsCoord = gp->fP.fACRWeight;
	    sampler = createSampler(weightDABGR, weightAbsCoord, weightAbsCell);
	}
	
	//train a set of trees, initialize training classes
	jp::TreeTrainer<jp::sampler_outer_t> tTrainer(&imgCache, sampler);
	jp::RegressionTreeTrainer<jp::feature_t> rTrainer(&imgCache);
	rForests[trainingRound] = std::vector<jp::RegressionTree<jp::feature_t>>(gp->fP.treeCount);
	
	stopWatch.init();
	for (size_t ti = 0; ti < gp->fP.treeCount; ++ti) 
	{
	    std::cout << GREENTEXT("Training structure of tree " << ti << "... ") << std::endl;
	    tTrainer.train(rForests[trainingRound][ti].structure); // learn the tree structure

	    std::cout << std::endl << "Collecting leaf distributions... " << std::endl;
	    rTrainer.train(rForests[trainingRound][ti]); // learn the leaf distributions
	}

	std::cout << GREENTEXT("Done training in " << (stopWatch.stop() / 1000) << "s.") << std::endl;

	// save forest	
	std::ofstream rFile(gp->getFileName(trainingRound), std::ios::binary | std::ios_base::out | std::ios_base::trunc);
	jp::write(rFile, rForests[trainingRound]);
	rFile.close();
	
	std::cout << "Successfully serialized forest to file." << std::endl << std::endl;	
	
	if(trainingRound + 1 == trainingRounds)
	    break;	
	
	// load a new set of training images
	imgCache.reload(trainingData, bgSet);
	
	// apply last forest to training data as input for auto-context features
	std::cout << std::endl << "Classifying training data: ";
	gp->fP.training = false; // set mode to test for classification

	rClassifiers[trainingRound] = new jp::RegressionForestClassifier<jp::feature_t>(rForests[trainingRound]);
	
        std::vector<jp::img_leaf_t> leafImgs; // one leaf image per tree in the forest, per pixel the index of the tree leaf where it ended up
        std::vector<jp::img_stat_t> objProbs(gp->fP.objectCount); // one probability image per object, each pixels contains the probability to belong to this object (soft segmentation)
	
	// iterate over training images
	stopWatch.init();	
	for(unsigned dataIdx = 0; dataIdx < imgCache.dataCache.size(); dataIdx++)
	{
	    std::cout << ".";
	    std::cout.flush();
	  
	    // if training images are segmented, paste them onto a random background
	    jp::img_bgr_t cascadeColor = imgCache.dataCache[dataIdx].colorData.clone();
	    jp::img_id_t cascadeSeg = imgCache.dataCache[dataIdx].seg.clone();
	    
	    if(!backgroundSets.empty() && dataIdx < imgCache.bgPointer) // check whether image is an object image (not a background image)
	    {
		// choose a random background image
		int bgIdx = irand(imgCache.bgPointer, imgCache.dataCache.size());
		jp::img_bgr_t bgImg = imgCache.dataCache[bgIdx].colorData;

		// choose a random paste position
		int bgX = irand(0, bgImg.cols - cascadeColor.cols - 1);
		int bgY = irand(0, bgImg.rows - cascadeColor.rows - 1);

		// paste object onto background
		for(unsigned x = 0; x < cascadeColor.cols; x++)
		for(unsigned y = 0; y < cascadeColor.rows; y++)
		    if(!cascadeSeg(y, x))
			cascadeColor(y, x) = bgImg(bgY + y, bgX + x);
	    }

	    // create a data item of the current image for the classifier
	    jp::img_data_t cascadeData;
	    
	    // set (altered color) and segmentation data
	    cascadeData.seg = jp::img_id_t::ones(cascadeColor.rows, cascadeColor.cols);;
	    cascadeData.colorData = cascadeColor;
	    
	    // initialize auto-context feature channel
	    cascadeData.labelData = std::vector<jp::img_label_t>(gp->fP.objectCount);
	    cascadeData.coordData = std::vector<jp::img_coord_t>(gp->fP.objectCount);

	    // set depth if available
	    jp::img_depth_t* cascadeDepth = NULL;
	    if(gp->fP.useDepth) cascadeDepth = &(imgCache.depthCache[dataIdx]);
	    
	    // classify training image with the current random forest stack
	    for(unsigned pass = 0; pass <= trainingRound; pass++)
	    {
		leafImgs.clear();
		rClassifiers[pass]->classify(cascadeData, leafImgs, cascadeDepth); // get leaf indices
		rClassifiers[pass]->getObjsProb(rForests[pass], leafImgs, objProbs); // calculate object probabilities
		
		computeAutoContextChannels(rForests[pass], leafImgs, objProbs, cascadeData); // set auto-context feature channels
	    }	    

	    // store auto-context feature channels in data cache for the next training round
	    for(int lD = 0; lD < imgCache.dataCache[dataIdx].labelData.size(); lD++)
	    {
		imgCache.dataCache[dataIdx].labelData[lD] = cascadeData.labelData[lD];
		imgCache.dataCache[dataIdx].coordData[lD] = cascadeData.coordData[lD];
	    }
	    
	    if(dataIdx >= imgCache.bgPointer)
	    {
		// set object probability maps for background images (for hard negative mining)
		jp::img_stat_t objProb = jp::img_stat_t::zeros(cascadeColor.rows, cascadeColor.cols);
		for(unsigned x = 0; x < objProb.cols; x++)
		for(unsigned y = 0; y < objProb.rows; y++)
		for(unsigned o = 0; o < objProbs.size(); o++)
		    objProb(y, x) += objProbs[o](y, x);
		imgCache.objProbCache[dataIdx] = objProb;
	    }
	}
	
	std::cout << std::endl;
    }

    return 0;
}
#pragma once

#include "types.h"
#include "thread_rand.h"
#include "features.h"


namespace jp
{
  
    template<typename TFeature1, typename TFeature2>
    class FeatureCombined
    {
    public:
        
        FeatureCombined() {}
      

	FeatureCombined(const TFeature1& feat1) : feat1(feat1), selectFirst(true)
	{
	}
	
   
	FeatureCombined(const TFeature2& feat2) : feat2(feat2), selectFirst(false)
	{
	}


	unsigned char getType() const
	{
	    return (selectFirst) ? feat1.getType() : feat2.getType();
	}
	
	
	void setThreshold(double thresh)
	{
	    (selectFirst) ? feat1.setThreshold(thresh) : feat2.setThreshold(thresh);
	}
	

	double computeResponse(int x, int y, float scale, const jp::img_data_t& data) const
	{   
	    return (selectFirst) ? 
		feat1.computeResponse(x, y, scale, data): 
		feat2.computeResponse(x, y, scale, data);     
	}
	

	bool operator()(int x, int y, float scale, const jp::img_data_t& data) const
	{
	    return (selectFirst) ?
		feat1(x, y, scale, data):
		feat2(x, y, scale, data);
	}
	
	void store(std::ofstream& file) const
	{
	    write(file, selectFirst);
	    
	    (selectFirst) ? 
		write(file, feat1):
		write(file, feat2);
	}

	void restore(std::ifstream& file)
	{
	    read(file, selectFirst);
	  
	    (selectFirst) ? 
		read(file, feat1):
		read(file, feat2);
	}
	

	void print() const
	{
	    (selectFirst) ? 
		feat1.print(): 
		feat2.print();
	}
      
    private:
	TFeature1 feat1; // feature of type 1, either feat1 or feat2 is set
	TFeature2 feat2; // feature of type 2, either feat1 or feat2 is set
	bool selectFirst; // marks whether feat1 or feat2 is set
    };
    
    template<typename TFeatureSampler1, typename TFeatureSampler2>
    class FeatureSamplerCombined
    {
    public:
	typedef typename TFeatureSampler1::feature_t feature_t1;
	typedef typename TFeatureSampler2::feature_t feature_t2;
	typedef FeatureCombined<feature_t1, feature_t2> feature_t;

	FeatureSamplerCombined(const TFeatureSampler1& sampler1, const TFeatureSampler2& sampler2,
	    double fracFirst = 0.5)
	    : sampler1(sampler1), sampler2(sampler2), fracFirst(fracFirst)
	{
	    assert(fracFirst >= 0.0 && fracFirst <= 1.0);
	}

	feature_t sampleFeature() const
	{
	    if (drand(0, 1) <= fracFirst)
		return feature_t(sampler1.sampleFeature());
	    else
		return feature_t(sampler2.sampleFeature());
	}
	
	std::vector<feature_t> sampleFeatures(unsigned count) const
	{
	    std::vector<feature_t> features;
	    features.reserve(count);
	  
	    if (drand(0, 1) <= fracFirst)
	    {
		std::vector<feature_t1> features1 = sampler1.sampleFeatures(count);
		for(unsigned i = 0; i < features1.size(); i++)
		    features.push_back(feature_t(features1[i]));
	    }
	    else
	    {
		std::vector<feature_t2> features2 = sampler2.sampleFeatures(count);
		for(unsigned i = 0; i < features2.size(); i++)
		    features.push_back(feature_t(features2[i]));
	    }
	    
	    return features;
	}   
      
    private:
	std::uniform_real_distribution<double> rfirst; // distribution of sampling a feature of the first or second type

	TFeatureSampler1 sampler1; // sampler of feature type 1
	TFeatureSampler2 sampler2; // sampler of feature type 2
	double fracFirst; // probability of sampling feature type 1
    };

    template<typename TFeature1, typename TFeature2>
    void write(std::ofstream& file, const FeatureCombined<TFeature1, TFeature2>& feature)
    {
	feature.store(file);
    }
     
    template<typename TFeature1, typename TFeature2>
    void read(std::ifstream& file, FeatureCombined<TFeature1, TFeature2>& feature)
    {
	feature.restore(file);
    }
    
    typedef FeatureSamplerCombined<FeatureSamplerAbsCoord, FeatureSamplerAbsCell> sampler_inner_t1; // inner node of recursive feature type tree
    typedef FeatureSamplerCombined<FeatureSamplerDABGR, sampler_inner_t1> sampler_outer_t; // inner node of recursive feature type tree
    typedef sampler_outer_t::feature_t feature_t; // feature type definition used throughout this code
}
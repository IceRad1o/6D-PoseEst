#pragma once

namespace jp
{
   
    class FeatureAbsCoord
    {
    public:
        
	FeatureAbsCoord() : off_x(0), off_y(0), channel(0), dimension(0), direction(0)
	{
	}


	FeatureAbsCoord(int off_x, int off_y, int channel, int dimension, int direction) 
	    : off_x(off_x), off_y(off_y), channel(channel), dimension(dimension), direction(direction)
	{
	}
	uchar getType() const { return 10; }

	double computeResponse(int x, int y, float scale, const jp::img_data_t& data) const
	{
	    // auto-context feature channels might be stored sub-sampled, in this case the offset vector has also be sub-sampled
	    int acSubSample = GlobalProperties::getInstance()->fP.acSubsample;
	    
	    // scale and clamp the offset vector
	    FeaturePoints fP = getFeaturePoints(
		x/acSubSample, 
		y/acSubSample, 
		off_x/acSubSample, 
		off_y/acSubSample, 
		scale/acSubSample, 
		data.labelData[channel].cols, 
		data.labelData[channel].rows);	  
	  
	    // probe the auto-context object coordinate feature channel
	    return data.coordData[channel](fP.y1, fP.x1)[dimension];
	}

	bool operator()(int x, int y, float scale, const jp::img_data_t& data) const
	{
	    if(direction)
		return computeResponse(x, y, scale, data) <= thresh;
	    else
		return computeResponse(x, y, scale, data) > thresh;
	}

	void setThreshold(double thresh) { this->thresh = thresh; }

	void print() const
	{
	    std::cout << "Absolute Coord Feature (x: " << off_x << ", y: " << off_y 
		<< ", t: " << thresh << ")" << std::endl;
	}

	void store(std::ofstream& file) const
	{
	    write(file, off_x);
	    write(file, off_y);
	    write(file, thresh);
	    write(file, channel);
	    write(file, dimension);
	    write(file, direction);
	}

	void restore(std::ifstream& file)
	{
	    read(file, off_x);
	    read(file, off_y);
	    read(file, thresh);
	    read(file, channel);
	    read(file, dimension);
	    read(file, direction);
	}
	
    private:
	int off_x, off_y; // offset vector of the pixel probe
	int channel; // which auto context feature channel to probe (channels correspond to different objects)
	int dimension; //which object coordinate component (x, y, z) to probe
	int direction; //should the feature response be smaller or greater than the threshold?
	jp::coord1_t thresh; // feature threshold
    };

    template<>
    void write(std::ofstream& file, const FeatureAbsCoord& feature);
  
    template<>
    void read(std::ifstream& file, FeatureAbsCoord& feature);
    
    class FeatureSamplerAbsCoord
    {
    public:
	typedef FeatureAbsCoord feature_t;


	FeatureSamplerAbsCoord(int off_max, int maxChannel) : off_max(off_max), maxChannel(maxChannel)
	{
	}
	 	 
	feature_t sampleFeature() const
	{
	    return feature_t(getOffset(), getOffset(), getChannel(), getDimension(), getDirection());
	}

	std::vector<feature_t> sampleFeatures(unsigned count) const
	{
	    // create number of thresholds of identical features
	    int offset1 = getOffset();
	    int offset2 = getOffset();
	    int channel = getChannel();
	    int dimension = getDimension();
	    int direction = getDirection();

	    std::vector<feature_t> features;
	    for(unsigned i = 0; i < count; i++)
	    {
		features.push_back(feature_t(offset1, offset2, channel, dimension, direction));
	    }
		
	    return features;
	}

    private:
	int off_max;  // maximally allowed offset vector
	int maxChannel; // number of auto-context feature channels

	int getOffset() const { return irand(-off_max, off_max + 1); }

	int getChannel() const { return irand(0, maxChannel); }
	
	int getDimension() const { return irand(0, 3); }

	int getDirection() const { return irand(0, 2); }
    };  
}
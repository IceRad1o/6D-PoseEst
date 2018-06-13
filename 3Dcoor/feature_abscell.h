#pragma once


namespace jp
{
  
    class FeatureAbsCell
    {
    public:
                 
	FeatureAbsCell() : off_x(0), off_y(0), channel(0), direction(0)
	{
	}

	FeatureAbsCell(int off_x, int off_y, int channel, int direction) 
	    : off_x(off_x), off_y(off_y), channel(channel), direction(direction)
	{
	}


	uchar getType() const { return 9; }


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
	    
	    // probe the auto-context object class feature channel
	    return data.labelData[channel](fP.y1, fP.x1);
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
	    std::cout << "Absolute Cell Feature (x: " << off_x << ", y: " << off_y 
		<< ", t: " << thresh << ")" << std::endl;
	}
	
	void store(std::ofstream& file) const
	{
	    write(file, off_x);
	    write(file, off_y);
	    write(file, thresh);
	    write(file, channel);
	    write(file, direction);
	}

	void restore(std::ifstream& file)
	{
	    read(file, off_x);
	    read(file, off_y);
	    read(file, thresh);
	    read(file, channel);
	    read(file, direction);
	}
	
    private:
	int off_x, off_y;  // offset vector of the pixel probe
	int channel; // which auto context feature channel to probe (channels correspond to different objects)
	jp::label_t thresh; // feature threshold
	int direction; //should the feature response be smaller or greater than the threshold?
    };
  
    template<>
    void write(std::ofstream& file, const FeatureAbsCell& feature);
  
    template<>
    void read(std::ifstream& file, FeatureAbsCell& feature);
         
    class FeatureSamplerAbsCell
    {
    public:
	typedef FeatureAbsCell feature_t;

	FeatureSamplerAbsCell(int off_max, int maxChannel) : off_max(off_max), maxChannel(maxChannel)
	{
	}
	    	    
	feature_t sampleFeature() const
	{
	    return feature_t(getOffset(), getOffset(), getChannel(), getDirection());
	}

	std::vector<feature_t> sampleFeatures(unsigned count) const
	{
	    // create number of thresholds of identical features
	    int offset1 = getOffset();
	    int offset2 = getOffset();
	    int channel = getChannel();
	    int direction = getDirection();

	    std::vector<feature_t> features;
	    for(unsigned i = 0; i < count; i++)
	    {
		features.push_back(feature_t(offset1, offset2, channel, direction));
	    }
		
	    return features;
	}

    private:
	int off_max;  // maximally allowed offset vector
	int maxChannel; // number of auto-context feature channels
	

	int getOffset() const { return irand(-off_max, off_max + 1); }
	
	int getChannel() const { return irand(0, maxChannel); }
	
	int getDirection() const { return irand(0, 2); }
    };  
}
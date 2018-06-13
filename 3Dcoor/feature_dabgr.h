#pragma once
//rgb pixel featureÌØÕ÷

namespace jp
{
   
    class FeatureDABGR
    {
    public:
        FeatureDABGR() : channel1(0), off1_x(0), off1_y(0), channel2(0), off2_x(0), off2_y(0), training(false)
	{
	}

	FeatureDABGR(int channel1, int off1_x, int off1_y, int channel2, int off2_x, int off2_y, bool training) : 
	    channel1(channel1), off1_x(off1_x), off1_y(off1_y),
	    channel2(channel2), off2_x(off2_x), off2_y(off2_y),
	    training(training)
	{
	}

	uchar getType() const { return 0; }

	double computeResponse(int x, int y, float scale, const jp::img_data_t& data) const
	{
	    // scale and clamp offset vectors
	    FeaturePoints fP = getFeaturePoints(x, y, off1_x, off1_y, off2_x, off2_y, scale, data.seg.cols, data.seg.rows);

	    double val1, val2;
	    
	     //read out pixel probe 1
 	    if(data.seg(fP.y1, fP.x1))
		val1 = (double) data.colorData(fP.y1, fP.x1)(channel1);
 	    else
 		val1 = irand(0, 256); // return a random color outside ground truth segmentation

	     // read out pixel probe 2
 	    if(data.seg(fP.y2, fP.x2))
		val2 = (double) data.colorData(fP.y2, fP.x2)(channel2);
 	    else
 		val2 = irand(0, 256); // return a random color outside ground truth segmentation
	    
	    return val1 - val2; // feature response is difference of pixel probe values
	}

	bool operator()(int x, int y, float scale, const jp::img_data_t& data) const
	{
	    double resp = computeResponse(x, y, scale, data);
	    return resp >= thresh;
	}


	void setThreshold(double thresh) { this->thresh = thresh; }

	void print() const
	{
	    std::cout << "Color Feature (x1: " << off1_x << ", y1: " << off1_y 
		<< ", channel1: " << channel1 << ", x2: " << off2_x << ", y2: " << off2_y 
		<< ", channel2: " << channel2 << ", t: " << thresh << ")" << std::endl;
	}
	
	void store(std::ofstream& file) const
	{
	    write(file, channel1);
	    write(file, channel2);
	    write(file, off1_x);
	    write(file, off1_y);
	    write(file, off2_x);
	    write(file, off2_y);
	    write(file, thresh);
	}

	void restore(std::ifstream& file)
	{
	    read(file, channel1);
	    read(file, channel2);
	    read(file, off1_x);
	    read(file, off1_y);
	    read(file, off2_x);
	    read(file, off2_y);
	    read(file, thresh);	
	    training = false;
	}
	
    private:
	int channel1; // RGB channel number of the first pixel probe
	int channel2; // RGB channel number of the second pixel probe
	int off1_x, off1_y; // offset vector of the first pixel probe
	int off2_x, off2_y; // offset vector of the second pixel probe
	double thresh; // feature threshold
	bool training; // does the feature operate in training mode? (can be used to simulate noise during training)
    };

    template<>
    void write(std::ofstream& file, const FeatureDABGR& feature);
  
    template<>
    void read(std::ifstream& file, FeatureDABGR& feature);
    
    class FeatureSamplerDABGR
    {
    public:
	typedef FeatureDABGR feature_t;

	FeatureSamplerDABGR(int off_max) : off_max(off_max)
	{
	}
	    
	feature_t sampleFeature() const
	{
	    return feature_t(getChannel(), getOffset(), getOffset(), getChannel(), getOffset(), getOffset(), true);
	}

	std::vector<feature_t> sampleFeatures(unsigned count) const
	{
	    // create number of thresholds of identical features
	    int channel1 = getChannel();
	    int channel2 = getChannel();
	    int offset1 = getOffset();
	    int offset2 = getOffset();
	    int offset3 = getOffset();
	    int offset4 = getOffset();

	    std::vector<feature_t> features;
	    for(unsigned i = 0; i < count; i++)
	    {
		features.push_back(feature_t(channel1, offset1, offset2, channel2, offset3, offset4, true));
	    }
		
	    return features;
	}

    private:
	int off_max; // maximally allowed offset vector
	
	int getOffset() const { return irand(-off_max, off_max + 1); }
	
	int getChannel() const { return irand(0, 3); }
    };  
}
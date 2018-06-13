#include "features.h"

namespace jp
{
    template<>
    void write(std::ofstream& file, const FeatureDABGR& feature)
    {
	feature.store(file);
    }
    template<>
    void read(std::ifstream& file, FeatureDABGR& feature)
    {
	feature.restore(file);
    }
    template<>
    void write(std::ofstream& file, const FeatureAbsCell& feature)
    {
	feature.store(file);
    }
    template<>
    void read(std::ifstream& file, FeatureAbsCell& feature)
    {
	feature.restore(file);
    }    
    template<>
    void write(std::ofstream& file, const FeatureAbsCoord& feature)
    {
	feature.store(file);
    }
    template<>
    void read(std::ifstream& file, FeatureAbsCoord& feature)
    {
	feature.restore(file);
    }       
}
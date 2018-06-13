#include "dataset.h"
#include "Hypothesis.h"

namespace jp
{
    jp::label_t getLabel(jp::id_t objID, jp::cell_t objCell)
    {
	// maping of global labels to object ID and local labels as follows:
	// map 0		to 0:0
	// map 1		to 1:1
	// map cellCount	to 1:cellCount
	// map cellCount+1 	to 2:1
	// and so on...      
	if(objID < 1) throw std::runtime_error("Invalid object passed (ID < 1)");
	jp::cell_t cellCount = GlobalProperties::getInstance()->fP.getCellCount();
	return (objID - 1) * cellCount + objCell;
    }
    
    jp::coord3_t pxToEye(int x, int y, jp::depth_t depth)
    {
	jp::coord3_t eye;
	
	if(depth == 0) // depth hole -> no camera coordinate
	{
	    eye(0) = 0;
	    eye(1) = 0;
	    eye(2) = 0;
	    return eye;
	}
	
	GlobalProperties* gp = GlobalProperties::getInstance();
	
	eye(0) = (short) ((x - (gp->fP.imageWidth / 2.f + gp->fP.xShift)) / (gp->fP.focalLength / depth));
	eye(1) = (short) -((y - (gp->fP.imageHeight / 2.f + gp->fP.yShift)) / (gp->fP.focalLength / depth));
	eye(2) = (short) -depth; // camera looks in negative z direction
	
	return eye;
    }

    bool onObj(const jp::coord3_t& pt)
    {
	return ((pt(0) != 0) || (pt(1) != 0) || (pt(2) != 0));
    }

    bool onObj(const jp::label_t& pt)
    {
	return (pt != 0);
    }
}
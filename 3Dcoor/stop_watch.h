#pragma once

#include <chrono>


class StopWatch
{
public:
    StopWatch(){ init(); }
  
    void init()
    {
	start = std::chrono::high_resolution_clock::now();
    }
    
    float stop()
    {
	std::chrono::high_resolution_clock::time_point now;
	now = std::chrono::high_resolution_clock::now();
	
	std::chrono::high_resolution_clock::duration duration = now - start;
	
	start = now;
	
	return static_cast<float>(
	    1000.0 * std::chrono::duration_cast<std::chrono::duration<double>>(
	    duration).count());
    }
    
private:
    std::chrono::high_resolution_clock::time_point start;
};
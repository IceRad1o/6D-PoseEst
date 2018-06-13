#pragma once

#include <random>

class ThreadRand
{
public:
 
  static int irand(int min, int max, int tid = -1);
  
 
  static double drand(double min, double max, int tid = -1);
  
 
  static double dgauss(double mean, double stdDev, int tid = -1);
    

  static void forceInit(unsigned seed);
  
private:  
  
  static std::vector<std::mt19937> generators;
  
  static bool initialised;
 
  static void init(unsigned seed = 1305);
};


int irand(int incMin, int excMax, int tid = -1);

double drand(double incMin, double incMax, int tid = -1);

 
int igauss(int mean, int stdDev, int tid = -1);


double dgauss(double mean, double stdDev, int tid = -1);
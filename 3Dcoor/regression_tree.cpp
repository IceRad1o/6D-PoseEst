#include"regression_tree.h"

namespace jp
{
  void write(std::ofstream& file, const mode_t& mode)
  {
      write(file, mode.covar);
      write(file, mode.mean);
      write(file, mode.support);
  }


  void read(std::ifstream& file, mode_t& mode)
  {
      read(file, mode.covar);
      read(file, mode.mean);
      read(file, mode.support);
  }
}
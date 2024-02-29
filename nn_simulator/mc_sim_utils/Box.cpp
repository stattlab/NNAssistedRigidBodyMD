#include "Box.h"
// #include "MPIErrorHandling.h"

// #define OMPI_SKIP_MPICXX    1
// #include <mpi.h>

Box::Box(const Point& pSysDim)
{
   my_rank  = 0;
   size_mpi = 1;
   // MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   // MPI_Comm_size(MPI_COMM_WORLD, &size_mpi);

   // Initialize the private variables
   SysDim = pSysDim;

   if (SysDim.x[0]<=0 || SysDim.x[1]<=0  || SysDim.x[2]<=0 )
   {
     // MPIErrorHandling::error("Invalid Box Dimension detected in Box::Box(). Box Dimensions have to be >0 in all three spatial directions.");
     // MPIErrorHandling::error("Invalid Box Dimension detected in Box::Box(). Box Dimensions have to be >0 in all three spatial directions.");
     std::cout<<"Invalid Box Dimension detected in Box::Box(). Box Dimensions have to be >0 in all three spatial directions."<<std::endl;
   }

   HalfDim    = SysDim * 0.5;

   //const double eps = 10e-12;

   SysDimInv = Point(1,1,1)/SysDim;

   const double eps = 0.0;
   RightBorder = HalfDim - eps;
   LeftBorder  = HalfDim * (-1);
   const Point redSysdim = SysDim - eps;

   slabBoundary            = 0;
   slabWidthOfOuterBox     = 0;
   slabWidthOfOuterBoxHalf = slabWidthOfOuterBox / 2.0;

   V           = redSysdim.getProduct();
   biggestDim  = 0.0;
   smallestDim = 9999999;
   for (int i = 0; i < 3; i++)
   {
      if (SysDim.x[i] > biggestDim)
      {
         biggestDim = SysDim.x[i];
      }
      if (SysDim.x[i] < smallestDim)
      {
         smallestDim = SysDim.x[i];
      }
   }

}

Box::~Box()
{

}

// virtual methods -----------------------------------------------------------------------------------------------

void Box::usePBConditions(Point *)
{
   // MPIErrorHandling::error("Function Box::usePBConditions has to be overloaded");

  std::cout<<"Function Box::usePBConditions has to be overloaded."<<std::endl;
}

// void Box::setWallPotentials(PairPotentials *, const double)
// {
//    MPIErrorHandling::error("setWallPotentials in pbc Box shouldn't be used");
// }

void Box::recreateWallCellSystem(const double)
{
   // MPIErrorHandling::error("recreateWallCellSystem in pbc Box shouldn't be used");
   std::cout<<"ERROR 1232"<<std::endl;
}

double Box::getEnergyWall(const Particle&) const
{
   return(0.0);
}

bool Box::checkIfInBox(const Particle&) const
{
   // MPIErrorHandling::error("checkIfInBox in Box shouldn't be used");
   std::cout<<"checkIfInBox in Box shouldn't be used"<<std::endl;
   return(true);
}

bool Box::checkIfInBox(const Point&) const
{
  std::cout<<"checkIfInBox in Box shouldn't be used"<<std::endl;
   // MPIErrorHandling::error("checkIfInBox in Box shouldn't be used");
   return(true);
}

// ----------------------------------------------------------------------------------------------------------------


double Box::getX(const short i) const
{
   return(SysDim.x[i]);
}

double Box::getBiggestDim() const
{
   return(biggestDim);
}

double Box::getSmallestDim() const
{
   return(smallestDim);
}

const Point& Box::getDim() const
{
   return(SysDim);
}

const Point& Box::getHalfDim() const
{
   return(HalfDim);
}


double Box::getBoxVolume() const
{
   return(V);
}

const std::vector<bool> Box::getPeriodic() const
{
   return m_periodic;
}

const Point& Box::getRightBorder() const
{
   return(RightBorder);
}

const Point& Box::getLeftBorder() const
{
   return(LeftBorder);
}

void Box::setPeriodic(std::vector<bool> periodic)
{
  m_periodic[0] = periodic[0];
  m_periodic[1] = periodic[1];
  m_periodic[2] = periodic[2];

}

void Box::scale(const short c, const double r, std::vector <double> *backup)
{
   backup->push_back(SysDim.x[c]); SysDim.x[c]        *= r;
   backup->push_back(SysDimInv.x[c]); SysDimInv.x[c] = 1.0/SysDim.x[c];
   backup->push_back(HalfDim.x[c]); HalfDim.x[c]       = 0.5 * SysDim.x[c];
   backup->push_back(V); V *= r;
   backup->push_back(RightBorder.x[c]); RightBorder.x[c] *= r;
   backup->push_back(LeftBorder.x[c]); LeftBorder.x[c]   *= r;
   backup->push_back(biggestDim); biggestDim              = 0.0;
   backup->push_back(smallestDim); smallestDim            = 9999999;

   for (int i = 0; i < 3; i++)
   {
      if (SysDim.x[i] > biggestDim)
      {
         biggestDim = SysDim.x[i];
      }
      if (SysDim.x[i] < smallestDim)
      {
         smallestDim = SysDim.x[i];
      }
   }

}

void Box::scale(const double r, std::vector <Point> *backup, std::vector <double> *backup_d)
{
   backup->push_back(SysDim); SysDim        *= r;
   backup->push_back(SysDimInv); SysDimInv = Point(1,1,1)/SysDim;
   backup->push_back(HalfDim); HalfDim       = SysDim * 0.5;
   backup_d->push_back(V); V *= pow(r, 3.0);
   backup->push_back(RightBorder); RightBorder  *= r;
   backup->push_back(LeftBorder); LeftBorder    *= r;
   backup_d->push_back(biggestDim); biggestDim   = 0.0;
   backup_d->push_back(smallestDim); smallestDim = 9999999;
   for (int i = 0; i < 3; i++)
   {
      if (SysDim.x[i] > biggestDim)
      {
         biggestDim = SysDim.x[i];
      }
      if (SysDim.x[i] < smallestDim)
      {
         smallestDim = SysDim.x[i];
      }
   }

}

void Box::rescale(const short c, const std::vector <double>& backup)
{

   if (backup.size() != 8)
   {
      // MPIErrorHandling::error("backup vector does not have proper size in Box::rescale");

   }
   SysDim.x[c]      = backup[0];
   SysDimInv.x[c]   = backup[1];
   HalfDim.x[c]     = backup[2];
   V                = backup[3];
   RightBorder.x[c] = backup[4];
   LeftBorder.x[c]  = backup[5];
   biggestDim       = backup[6];
   smallestDim      = backup[7];
}

void Box::rescale(const std::vector <Point>& backup, const std::vector <double>& backup_d)
{

   if (backup.size() != 5)
   {
     //  std::cout<< backup.size()<<std::endl;
      // MPIErrorHandling::error("backup vector does not have proper size in Box::rescale");
   }
   if (backup_d.size() != 3)
   {
      // std::cout<< backup_d.size()<<std::endl;
      // MPIErrorHandling::error("backup_d vector does not have proper size in Box::rescale");
   }
   SysDim      = backup[0];
   SysDimInv   = backup[1];
   HalfDim     = backup[2];
   V           = backup_d[0];
   RightBorder = backup[3];
   LeftBorder  = backup[4];
   biggestDim  = backup_d[1];
   smallestDim = backup_d[2];
}


void Box::replace(Particle *oldCol, const Particle& newCol)
{
   *oldCol = newCol;
}

void Box::replace(Particle *oldCol, const Point& newOrigin)
{
   oldCol->setOrigin(newOrigin);
}

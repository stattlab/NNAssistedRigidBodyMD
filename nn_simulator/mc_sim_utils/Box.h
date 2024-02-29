#ifndef BOX_H_
#define BOX_H_


// #include "MPIErrorHandling.h"
// #include "CompilerDirectives.h"
// #include "MonteCarloUtils.h"
#include "Point.h"
#include "Particle.h"
#include "../gsd_utils/VectorMath.h"

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>

class Particle;
class Cell;

/**
 * @file Box.h
 * @brief Simulation box,  inheritance parent for all derived different box classes.
 */
/**
 *  The simulation box in three dimensions, which manages the cell systems and periodic boundary conditions. The middle
 *  of this box is defined as (0,0,0). To avoid conflicts with infinitely thin borders of the cell system and floating point precision, the
 *  system box is reduced at three (right) borders through a small epsilon. This class provides all the basic functions and virtual functions. It cannot be used directly in a simulation,
 *  one has to construct a derived class, to ensure correct treatment of confinement and similar effects.
 *  @brief Simulation box,  inheritance parent for all derived different box classes.
 **/
class Box {
public:

   /** Constructor of a box designed for binary mixtures.
    * @param Dim the linear dimensions of the box
    */
   Box(const Point& Dim);

   /** Destructor. */
   virtual ~Box();

   // ==============================================================================================================
   // virtual methods for wall versions

   virtual void usePBConditions(Point *P);
   // virtual void usePBC(linalg::aliases::double3 *P);

   // virtual void setWallPotentials(PairPotentials *wallpotentials, const double BWD);

   virtual void recreateWallCellSystem(const double newBWD);

   virtual double getEnergyWall(const Particle& C) const;

   virtual bool checkIfInBox(const Particle& P) const;

   virtual bool checkIfInBox(const Point& P) const;


   // ==============================================================================================================

   /** This method returns the dimension of a single side of the box.
    * @param i the axis number, x - 0, etc.
    * @return the specified linear dimension of the simulation box */
   double getX(const short i) const;


   /** Get the biggest linear dimension.
    * @return the biggest linear dimension */
   double getBiggestDim() const;

   /** Get the smallest linear dimension.
    * @return the smallest linear dimension */
   double getSmallestDim() const;

   /** This method returns the system box dimension.
    * @return dimension of the simulation box */
   const Point& getDim() const;

   /** This method returns the half system box dimension.
    * @return half dimension of the simulation box */
   const Point& getHalfDim() const;

   /** This method returns a quarter of the system box dimension.
    * @return half dimension of the simulation box */
   const Point& getQuarterDim() const;

   /** Get the volume of the box. This volume has already subtracted the epsilon area.
    *  @returns the volume */
   double getBoxVolume() const;

   const std::vector<bool> getPeriodic() const;

   void setPeriodic(std::vector<bool>  periodic);

   /** This method provides the linear dimensions of the right border.
    * @return linear dimensions of the right border */
   const Point& getRightBorder() const;

   /** This method provides the linear dimensions of the left border.
    * @return linear dimensions of the left border */
   const Point& getLeftBorder() const;

   /** Streches the box and all its related quantities, like the cells. In the case the cell system becomes too small, the
    * function  returns true, so that one can recreate the box insert all the particles again.
    * @param dim the dimension which should be streched
    * @param f the factor of change
    * @param backup the list of the old not rescaled values
    * @return true when there was a cell system change */
   void scale(const short dim, const double f, std::vector <double> *backup);

   /** Streches the box and all its related quantities, like the cells. In the case the cell system becomes too small, the
    * function  returns true, so that one can recreate the box insert all the particles again.
    * @param f the factor of change
    * @param backup the list of the old not rescaled vectors
    * @param backup_d the list of the old not rescaled scalars
    * @return true when there was a cell system change */
   void scale(const double f, std::vector <Point> *backup, std::vector <double> *backup_d);

   void rescale(const short dim, const std::vector <double>& backup);

   void rescale(const std::vector <Point>& backup, const std::vector <double>& backup_d);

   /** Overwrites a colloid
    * @param oldCol previous configuration
    * @param newCol new configuration */
   void replace(Particle *oldCol, const Particle& newCol);

   /** Corrects the origin of a colloid
    * No periodic boundary conditions are checked.
    * @param oldCol previous configuration
    * @param newOrigin new configuration */
   void replace(Particle *oldCol, const Point& newOrigin);



protected:

  std::vector<bool> m_periodic = {false,false,false};
   /** Dimension of the system */
   Point SysDim;

   /** Inverted dimension of the system */
   Point SysDimInv;

   /** Half of the system dimension */
   Point HalfDim;

   /** z coordinates of the two 'walls' needed for the slab geometry */
   double slabBoundary;

   /** full width of the outer boxes needed for the slab geometry */
   double slabWidthOfOuterBox;

   /** full width of the outer boxes needed for the slab geometry */
   double slabWidthOfOuterBoxHalf;

   /** Volume of the system box */
   double V;

   /** Largest box dimension */
   double biggestDim;

   /** Smallest box dimension */
   double smallestDim;


   /** 3D right border */
   Point RightBorder;

   /** 3D left border */
   Point LeftBorder;

   /** rank of the thread */
   int my_rank;

   /** size of the job */
   int size_mpi;
};

#endif /* BOX_H_ */

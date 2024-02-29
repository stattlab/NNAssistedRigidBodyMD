#ifndef BOXCUBOID_H_
#define BOXCUBOID_H_

#include "Box.h"
#include "Point.h"
#include "Particle.h"
//#include "Cell.h"
// #include "MonteCarloUtils.h"

#include <iostream>
#include <cstdlib>
// using namespace linalg::aliases;

/**
 * @file
 * @brief Cubic box, derived from Box
 */

/** Bulk system box derived from the box class. Periodic boundary conditions are applied in each direction.
 * @brief Cubic Box, derived from Box*/
class BoxCuboid : public Box {
public:

   /** Constructor of a box designed for binary mixtures.
    * @param Dim the linear dimensions of the box
    *  */
   BoxCuboid(const Point& Dim);

   /** Periodic boundary condition treatment. Overwritten to ensure the correct treatment of the different geometries.
    * @param P the coordinate to be treated. */
   void usePBConditions(Point *P);
   // void usePBC(linalg::aliases::double3 *P);

   /** Destructor. */
   ~BoxCuboid();

   /** Checks if the particle can be associated to a cell.
    * @param P the particle being checked
    * @return true if particle is in the box, false otherwise */
   bool checkIfInBox(const Particle& P) const;

   /** Checks if the ppoint lies within the cylinder geometry.
    * @param P the point being checked
    * @return true if point is in the box, false otherwise */
   bool checkIfInBox(const Point& P) const;
};

#endif /* BOXCUBOID_H_ */

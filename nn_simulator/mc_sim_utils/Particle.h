#ifndef PARTICLE_H_
#define PARTICLE_H_

// #include "MPIErrorHandling.h"
// #include "MonteCarloUtils.h"
// #include "CompilerDirectives.h"
// #include "PairPotentials.h"
// #include "Bond.h"
// #include "Molecule.h"
#include "Box.h"
#include "Point.h"

#include <cstdio>
#include <iostream>
#include <vector>
#include <set>

class Box;
// class Bond;
// class Molecule;

/**
 * @file Particle.h
 * @brief Particle (position + potential) class.
 */

/** A particle, defined by its position and its potential. The potentials of the particle are pointers
 *  to spherical potentials, setup by an external class. The first potential is for interaction with its
 *  own species, the second potential to interact with the other species. To handle translation with periodic boundary conditions,
 *  a sphere also knows its simulation box as a pointer. Further, a sphere has pointers for cell management. Note that no
 *  destructor has to defined explicitly, since no memory allocation for pointers is done by this class.
 *  @brief Particle (position + potential) class.
 */
class Particle {
public:

   /** Constructor that creates a sphere independend from a simulation box.
    * @param Pos the sphere origin
    * @param Pot the potentials for interaction with particles*/
   Particle(const Point& Pos);

   /** Constructor that creates a sphere independend from a simulation box.
    *  @param Pos the sphere origin
    *  @param Pot the potentials for interaction with particles
    *  @param Type type of the particle */
   Particle(const Point& Pos, unsigned int Type);

   /** Constructor that creates a sphere that knows explicitly it's box.
    *  @param Pos the sphere origin
    *  @param Pot the potential for interaction with particles
    *  @param pBox the pointer on the simulation box */
   Particle(const Point& Pos, Box *pBox);

   /** Constructor that creates a sphere that knows explicitly it's box.
    *  @param Pos the sphere origin
    *  @param Pot the potential for interaction with particles
    *  @param pBox the pointer on the simulation box
    *  @param Type type of the particle */
   Particle(const Point& Pos, Box *pBox, unsigned int Type);

   /** Get a coordinate of the origin. Here 0 represents x-value, ..., 2 corresponds
    *  to the z-coordinate.
    *  @param pn the index of the coordinate
    *  @return the value of the coordinate */
   double getX(const short pn) const;

   /** Sets the position coordinate in a single dimension to the given value.
    * @param pn dimension
    * @param v the new coordinate value */
   void setX(const short pn, const double v);

   /** Get the origin of the sphere as vector.
    *  @return a vector to the origin */
   const Point& getOrigin() const;

   /** Get the origin of the sphere as vector.
    *  @param newOrigin vector to the origin */
   void setOrigin(const Point& newOrigin);

   /** Get the type of the sphere.
    *  @return a vector to the origin */
   unsigned int getType() const;

   /** Set the type of a particle
    *  @param newType new type of the particle */
   void setType(unsigned int newType);

   /** Get the generic box pointer, useful for creating copies. */
   Box *getBoxPointer() const;

   /** Set the generic box pointer, needed for paricle size simulations
    * and volume moves for changing the cell system after init
    *  @param newBox new box pointer of the particle */
   void setBoxPointer(Box *newBox);

   /** This static method generates a particle with its potentials and random position. The box pointer is
    *  used to determine interval boundaries for the random position and is also given to the new particle.
    *  @param pBox the pointer on the simulation box
    *  @param Pot the potentials for interaction with particles
    *  @param Type particle type
    *  @return a new particle with random position */
   // static Particle getRandomParticle(PairPotentials *Pot, Box *pBox, unsigned int Type);


   /** This procedure translates randomly in the cube with side length of the amount. Periodic
    *  boundary conditions are also checked here.
    *  @param amount the side length of the cube in which the translation takes place. */
   void translate(const double amount);

   /** This procedure translates in the cube with side length of the amount. Periodic
    *   boundary conditions are also checked here.
    *   @param trans the translation vector */
   void translate(const Point& trans);

   void rotate(double **rotation_matrix,const Point& center);
   void rotate_WOpbc(double **rotation_matrix,const Point& center);

   /** A mirror reflection at a point is done. Also, the periodic boundary conditions are checked. WARNING: in the case of a
    * cylindrical box it is not checked if the origin is mirrored out of the box. This check has to be done by the simulation class.
    *  @param P the point, at which the mirror reflection takes place */
   void mirror(const Point& P);

   /** A mirror reflection in the z plane is done. Also, the periodic boundary conditions are checked.
    *  @param P P the point, at which the mirror reflection takes place neglecting the z-coordinate */
   void mirror_XY(const Point& P);

   /** A mirror reflection at the plane perpendicular to the z-axis is done. Also, the periodic boundary conditions are checked.
    *  @param P the z-coordinate which defines the plane of the mirror reflection*/
   void mirror_Z(const double P);

   /** Checks whether a particle is marked by its flag.
    * @return true if the praticle is marked. */
   bool isMarked() const;

   /** Switches on the marked flag */
   void mark();

   /** Switches off the marked flag */
   void unMark();

   /** Method to calculate the energy between this particle and the given one. The function is independent of the
    *  cell system.
    *  @param Particle the particle, to which the energy is calculated
    *  @return the corresponding potential energy */

   /** Calculates the distance between two particles using periodic boundary cond.
   * @param Particle the particle to calculate the distance to
    * @return the distance vector */
   Point getDist(const Particle& Particle) const;

   /** Prints information to standard out about the sphere with respect to the given level of information
    * @param InfLevel determines how much information is printed out. 1 - only the origin; 2 - Radius & Origin; 4 - Additionally the cell system status */
   void printObject(const int InfLevel = 2) const;

   /** Prints information about the sphere to the specified file.
    * @param File the location of the print-out
    * @param Style the format of the output, 0 - vmd; 1 - qmga;
    * @param Type model number for qmga style */
   void printObject(FILE *File, const int Style, const int Type) const;

   /** Prints information about the sphere as povray format to the specified file.
    * @param File the location of the print-out
    * @param Style the format of the output, 0 - red; 1 - yellow; 2 - blue; 3 - green;*/
   void printObject_pov(FILE *File, const int Style) const;


   // below returns 0 if particle is not in a molecule and 0 is a used molecule type
   // so should only be used when printing the configurations
private:

   /** The x,y,z coordinates of the origin of the sphere */
   Point Origin;
   /** A pointer on the system box to take periodic boundary conditions into account*/
   Box *m_Box;

   /** Potentials for interactions */

   /** A boolean flag for marking particles. It is ment for detecting overlapping particles
    * in cluster moves, so that they don't have to be removed from the cell system to ensure a
    * fast internal energy calculation. */
   bool marked;

   /** the type of the particle */
   unsigned int m_Type;

};

#endif /* PARTICLE_H_ */

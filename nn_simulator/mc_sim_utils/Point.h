#ifndef POINT_H_
#define POINT_H_

#include <cmath>
#include <iostream>
#include <cstdio>
#include <utility>

// #include "CompilerDirectives.h"

/**
 * @file
 * @brief Point in Cartesian coordinates.
 */

/** This class represents a three dimensional vector, or a point in Cartesian coordinates. A lot of methods, which are useful for
 *  typical operations are implemented as operators. Also some other typical properties like mirroring or
 *  normalization can be performed with this class. It can also handle periodic boundary conditions of cubic boxes.
 *  @author Alexander Winkler
 *  @version 2.02
 *  @brief Point in Cartesian coordinates.*/

class Point {
public:

   /// Constructor. Creates a point at the origin (0,0,0).
   Point();

   /** Constructor. Creates a point specified by the given parameters.
    *  @param x1 x-coordinate
    *  @param x2 y-coordinate
    *  @param x3 z-coordinate */
   Point(double x1, double x2, double x3);

   /** Constructor. Array version. */
   Point(const double x[]);

   /**Copy constructor */
   Point(const Point& P)
   {
      x[0] = P.x[0];
      x[1] = P.x[1];
      x[2] = P.x[2];
   }

   double getX() const;
   double getY() const;
   double getZ() const;

   void setX(double x);
   void setY(double y);
   void setZ(double z);

   /** Calculates the square of this vector. Square simply means the scalar product with itself.
    *  @return the square */
   double getSquare() const;

   /** Calculates the length of the vector. For a point this corresponds to the distance to the
    *  origin.
    *  @return the root of the sqare */
   double getDistToOrigin() const;

   /** Calculates the inverse of each component of this point. This means x->1/x etc.
    *  If a component of the point is zero, this will return inf!
    *  @return the inverse components as a new point */
   const Point getInv() const;

   /** Calculates the product of all components.
    *  @return the product x*y*z */
   inline double getProduct() const
   {
      return(x[0] * x[1] * x[2]);
   }

   /** Returns the largest of the three components. */
   double getLargestComp() const;

   /** Returns the smallest of the three components. */
   double getSmallestComp() const;

   /** This static method calculates the scalar product of the two given vectors.
    *  @param P1 the first vector
    *  @param P2 the second vector
    *  @return the scalar product P1 · P2 */
   static double getScalarProduct(const Point& P1, const Point& P2);

   /** This static method calculates the cross product of the two given vectors.
    *  @param P1 the first vector
    *  @param P2 the second vector
    *  @return the cross product P1 x P2 */
   static const Point getCrossProduct(const Point& P1, const Point& P2);

   /** Generates the unit vector from this vector.
    *  @return the unit vector */
   const Point normalize();

   /** Does the mirroring of this point at the given point. Periodic boundaries are not recognized here.
    *  @param V the point at which the mirroring takes place */
   void mirror(const Point& V);

   /** Does the mirroring of this point at the given point. Periodic boundaries are not recognized here.
    *  @param V the point at which the mirroring takes place */
   void mirror_XY(const Point& V);

   /** Does the mirroring of this point at the plane perpendicular to the z-direction. Periodic boundaries are not recognized here.
    *  @param V the z-coordinate defining the plane at which the mirroring takes place */
   void mirror_Z(const double V);

   /** Does the mirroring at this point at the axis specified by the given two vectors. Direction vector
    *  V2 can be unnormalized.
    *  @param V1 the position vector
    *  @param V2 the direction vector */
   void mirror(const Point& V1, const Point& V2);

   /** Extracts the orthogonal Component of this point to the reference vector given in p.
    *  @param p (unity) direction vector
    *  @returns orthogonal component of this point to p
    */
   const Point getOrthogonalComponent(const Point&p) const;

   /** Determines the position of the point by using periodic boundary conditions.
    *  @param LeftBorder left border of a cubic system box
    *  @param RightBorder right border of a cubic system box
    *  @param Dimension vector containing the linear dimension of a cubic system box */
   void usePBCondition(const Point& LeftBorder, const Point& RightBorder, const Point& Dimension);

   /** Determines the position of the point by using periodic boundary conditions in x and y direction.
    *          @param LeftBorder left border of a cubic system box
    *          @param RightBorder right border of a cubic system box
    *          @param xDim size in x-direction
    *          @param yDim size in y-direction */
   void usePBCondition(const Point& LeftBorder, const Point& RightBorder, const double xDim, const double yDim);

   /** Determines the position of the point by using periodic boundary conditions for a cylindrical simulation box.
    *  @param LeftBorder left border of the z-dimension of the cylindrical system box
    *  @param RightBorder right border of the z-dimension of the cylindrical system box
    *  @param zDim vector containing the full linear z-dimension */
   void usePBCondition(const double LeftBorder, const double RightBorder, const double zDim);

   /** Checks if all components are exactly zero. */
   bool isZero() const;

   /** This operator does vector addition by creating a copy of 'this'.
    *  @param P the vector to add
    *  @return the result of the addition */
   const Point operator+(const Point& P) const;

   /** Vector addition in short form.
    *  @param P the vector to add
    *  @return result of the vector addition */
   const Point& operator+=(const Point& P);

   /** This operator does vector subtraction by creating a copy of 'this'.
    *  @param P the vector to subtract
    *  @return the result of the subtraction */
   const Point operator-(const Point& P)const;

   /** Vector subtraction in short form.
    *  @param P the vector to subtract
    *  @return result of the vector subtraction */
   const Point& operator-=(const Point& P);

   /** Scalar subtraction.
    * @param s the scalar to subtract from each component
    * @return the result of the scalar subtraction */
   const Point operator-(const double s) const;

   /** Scalar multiplication.
    *  @param s the scalar to multiply
    *  @return result of the scalar multiplication */
   const Point operator*(const double s)const;

   /** Scalar multiplication.
    *  @param s the scalar to multiply
    *  @return result of the scalar multiplication */
   const Point operator*(const int s)const;

   /** Scalar multiplication in short form.
    *  @param s the scalar to multiply
    *  @return result of the scalar multiplication */
   const Point& operator*=(const double s);

   /** Every component is multiplied with the corresponding component of the other vector.
    * @param P the second vector
    * @return component product vector */
   const Point operator*(const Point& P)const;

   /** Scalar division.
    * @param s the scalar for division
    * @return  the result of the operation */
   const Point operator/(const double s)const;

   /** Scalar division in short form.
    * @param s the scalar for division
    * @return  the result of the operation */
   const Point& operator/=(const double s);

   /** Scalar division for every component.
    * @param P the vector for division
    * @return  the result of the operation */
   const Point operator/(const Point& P)const;

   /** Scalar division for every component.
    * @param P the vector for division
    * @return  the result of the operation */
   const Point& operator/=(const Point& P);

   /** Comparison to another point.
    * @param P the point for comparison
    * @return true if the condition is valid for every component */
   bool operator<=(const Point& P) const;

   /** Comparison to another point.
    * @param P the point for comparison
    * @return true if the condition is valid for every component */
   bool operator<(const Point& P) const;

   /** Comparison to another point.
    * @param P the point for comparison
    * @return true if the condition is valid for every component */
   bool operator>=(const Point& P) const;

   /** Comparison to another point.
    * @param P the point for comparison
    * @return true if the condition is valid for every component */
   bool operator>(const Point& P) const;

   /** Comparison to another point.
    * @param P     the point for comparison
    * @return true if the condition is valid for at least one component */
   bool operator!=(const Point& P) const;

   /** Comparison to another point.
    * @param P     the point for comparison
    * @return true if the condition is valid for all components */
   bool operator==(const Point& P) const;

   Point& operator=(const Point& P);

   /** Prints the object to standard out */
   void printObject() const;

   /// The three coordinates of the point.
   double x[3];                                   // coordinates
};


// Some of the actual class functions are defined here to be inlined at other places in the code as well
// as these basic operations are used quite frequently. (Check with gprof for further information)

// =====================================================================================================
//
// Constructors
//
// =====================================================================================================

inline Point::Point()
{
   x[0] = 0;
   x[1] = 0;
   x[2] = 0;
}

inline Point::Point(double x1, double x2, double x3)
{
   x[0] = x1;
   x[1] = x2;
   x[2] = x3;
}

inline Point::Point(const double px[])
{
   for (short i = 0; i < 3; i++)
   {
      x[i] = px[i];
   }
}

// =====================================================================================================
//
// get-Funktionen und Tools
//
// =====================================================================================================

inline double Point::getX() const
{
   return(x[0]);
}

inline double Point::getY() const
{
   return(x[1]);
}

inline double Point::getZ() const
{
   return(x[2]);
}

inline void Point::setX(double mx)
{
   x[0] = mx;
}

inline void Point::setY(double my)
{
   x[1] = my;
}

inline void Point::setZ(double mz)
{
   x[2] = mz;
}

inline double Point::getSquare() const
{
   return(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
}

inline double Point::getDistToOrigin() const
{
   return(sqrt(getSquare()));
}

inline double Point::getScalarProduct(const Point& p1, const Point& p2)
{
   double Result = 0;

   for (short i = 0; i < 3; i++)
   {
      Result += p1.x[i] * p2.x[i];
   }
   return(Result);
}

inline const Point Point::getCrossProduct(const Point& p1, const Point& p2)
{
   const Point Result(p1.x[1] * p2.x[2] - p1.x[2] * p2.x[1],
                      p1.x[2] * p2.x[0] - p1.x[0] * p2.x[2],
                      p1.x[0] * p2.x[1] - p1.x[1] * p2.x[0]);

   return(Result);
}

inline const Point Point::getInv() const
{
   Point Result(*this);

   for (short i = 0; i < 3; i++)
   {
      Result.x[i] = 1.0 / Result.x[i];
   }

   return(Result);
}

inline double Point::getLargestComp() const
{
   double res = -1e200;

   if (x[0] > res)
   {
      res = x[0];
   }
   if (x[1] > res)
   {
      res = x[1];
   }
   if (x[2] > res)
   {
      res = x[2];
   }
   return(res);
}

inline double Point::getSmallestComp() const
{
   double res = 1e200;

   if (x[0] < res)
   {
      res = x[0];
   }
   if (x[1] < res)
   {
      res = x[1];
   }
   if (x[2] < res)
   {
      res = x[2];
   }
   return(res);
}

inline const Point Point::normalize()
{
   Point Result(*this);

   return(Result /= getDistToOrigin());
}

inline void Point::mirror(const Point& V)
{
   const Point doubleOrigin(V + V);

   *this = doubleOrigin - *this;
}

inline void Point::mirror_XY(const Point& V)
{
   x[0] = 2.0 * V.x[0] - x[0];
   x[1] = 2.0 * V.x[1] - x[1];
}

inline void Point::mirror_Z(const double V)
{
   x[2] = 2.0 * V - x[2];
}

inline bool Point::isZero() const
{
   if (x[0] == 0 && x[1] == 0 && x[2] == 0)
   {
      return(true);
   }
   else
   {
      return(false);
   }
}

inline const Point Point::getOrthogonalComponent(const Point&p) const
{
   Point unity = p;

   unity = unity.normalize();
   Point total(*this);

   Point parallel   = unity * getScalarProduct(total, unity);
   Point orthogonal = total - parallel;

   return(orthogonal);
}

// =====================================================================================================
//
// Operatoren
//
// =====================================================================================================

inline Point& Point::operator=(const Point& P)
{
   if (&P != this)
   {
      x[0] = P.x[0];
      x[1] = P.x[1];
      x[2] = P.x[2];
   }
   return(*this);
}

inline const Point Point::operator+(const Point& V)const
{
   Point Result(*this);

   for (short i = 0; i < 3; i++)
   {
      Result.x[i] += V.x[i];
   }

   return(Result);
}

inline const Point& Point::operator+=(const Point& V)
{
   x[0] += V.x[0];
   x[1] += V.x[1];
   x[2] += V.x[2];
   return(*this);
}

inline const Point Point::operator-(const Point& V)const
{
   Point Result(*this);

   Result.x[0] -= V.x[0];
   Result.x[1] -= V.x[1];
   Result.x[2] -= V.x[2];

   return(Result);
}

inline const Point& Point::operator-=(const Point& V)
{
   for (short i = 0; i < 3; i++)
   {
      x[i] -= V.x[i];
   }
   return(*this);
}

inline const Point Point::operator-(const double f)const
{
   Point Result(*this);

   for (short i = 0; i < 3; i++)
   {
      Result.x[i] -= f;
   }

   return(Result);
}

inline const Point Point::operator*(const double f)const
{
   Point Result(*this);

   for (short i = 0; i < 3; i++)
   {
      Result.x[i] *= f;
   }
   return(Result);
}

inline const Point Point::operator*(const int f)const
{
   Point Result(*this);

   for (short i = 0; i < 3; i++)
   {
      Result.x[i] *= f;
   }
   return(Result);
}

inline const Point& Point::operator*=(const double f)
{
   for (short i = 0; i < 3; i++)
   {
      x[i] *= f;
   }
   return(*this);
}

inline const Point Point::operator*(const Point& V)const
{
   Point Result(*this);

   for (short i = 0; i < 3; i++)
   {
      Result.x[i] *= V.x[i];
   }
   return(Result);
}

inline const Point Point::operator/(const double f)const
{
   const double InvF = 1.0 / f;
   Point       Result(*this);

   for (short i = 0; i < 3; i++)
   {
      Result.x[i] *= InvF;
   }
   return(Result);
}

inline const Point& Point::operator/=(const double f)
{
   const double InvF = 1.0 / f;

   for (short i = 0; i < 3; i++)
   {
      x[i] *= InvF;
   }
   return(*this);
}

inline const Point Point::operator/(const Point& V)const
{
   Point Result(*this);

   for (short i = 0; i < 3; i++)
   {
      Result.x[i] /= V.x[i];
   }
   return(Result);
}

inline const Point& Point::operator/=(const Point& V)
{
   for (short i = 0; i < 3; i++)
   {
      x[i] /= V.x[i];
   }
   return(*this);
}

inline bool Point::operator<=(const Point& P) const
{
   bool equal = true;

   if (x[0] > P.x[0])
   {
      equal = false;
   }
   else
   {
      if (x[1] > P.x[1])
      {
         equal = false;
      }
      else
      {
         if (x[2] > P.x[2])
         {
            equal = false;
         }
      }
   }
   return(equal);
}

inline bool Point::operator<(const Point& P) const
{
   bool equal = true;

   if (x[0] >= P.x[0])
   {
      equal = false;
   }
   else
   {
      if (x[1] >= P.x[1])
      {
         equal = false;
      }
      else
      {
         if (x[2] >= P.x[2])
         {
            equal = false;
         }
      }
   }
   return(equal);
}

inline bool Point::operator>=(const Point& P) const
{
   bool equal = true;

   if (x[0] < P.x[0])
   {
      equal = false;
   }
   else
   {
      if (x[1] < P.x[1])
      {
         equal = false;
      }
      else
      {
         if (x[2] < P.x[2])
         {
            equal = false;
         }
      }
   }
   return(equal);
}

inline bool Point::operator>(const Point& P) const
{
   bool equal = true;

   if (x[0] <= P.x[0])
   {
      equal = false;
   }
   else
   {
      if (x[1] <= P.x[1])
      {
         equal = false;
      }
      else
      {
         if (x[2] <= P.x[2])
         {
            equal = false;
         }
      }
   }
   return(equal);
}

inline bool Point::operator!=(const Point& P) const
{
   bool equal = false;

   if (x[0] != P.x[0])
   {
      equal = true;
   }
   else
   {
      if (x[1] != P.x[1])
      {
         equal = true;
      }
      else
      {
         if (x[2] != P.x[2])
         {
            equal = true;
         }
      }
   }
   return(equal);
}

inline bool Point::operator==(const Point& P) const
{
   return(x[0] == P.x[0] && x[1] == P.x[1] && x[2] == P.x[2]);
}

// =============================================================================
//
// Output
//
// =============================================================================

inline void Point::printObject() const
{
   printf("# %f %f %f \n", x[0], x[1], x[2]);
}

#endif //POINT_H_

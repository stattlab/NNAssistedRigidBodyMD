#ifndef MONTECARLOUTILS_H_
#define MONTECARLOUTILS_H_


// #include "MPIErrorHandling.h"
#include "Point.h"
// #include "CompilerDirectives.h"
#include "Randomc.h"

#include <iostream>
#include <string>
#include <cmath>
#include <complex>
#include <vector>
#include <cstdlib>
#include <cstring>



/**
 * @file
 * @brief Random object generation
 */
/** This is an utility class for generating various types of random objects. This class contains only
 *  static methods. Right interval borders are always excluded.
 *
 * @brief Random object generation
 */
class MonteCarloUtils
{
public:

   static const unsigned int NoOfPregeneratedGaussians = 4096;
   static const unsigned int NoOfPregeneratedMersennes = 1;

   /// Multithread version of Mersenne twister, length of the given vector is the number of threads
   static void setSeed(const std::vector <int>& seeds);

   /// Set the seed of the random number generator, which is used here.
   static void setSeed(const long);

   /** Reseeds the Mersenne random number generators.
    * @param seeds Vector containing a new set of seeds.*/
   static void resetSeed(const std::vector <int>& seeds);

   /** Resets the Mersenne random number generators to the initial state before seeding */
   static void deleteMersenneSeeds();

   /// Generates a random number between 0 and 1. The default generation is based on drand48.
   static double ran();

   /// Returns a random number based on the thread ID
   static double ran(const int TID);

   /** Returns a random number 0<=x<1 using Mersenne twister, with an array of pregenerated random numbers
    * @bug This function did not work correctly. The array was filled AFTER the first pass. Fixed but not tested...
    */
   static double getPGMersenneRan(const int TID);

   /// Returns a random boolean, using getPGMersenneRan()
   static bool getPGMersenneRanBool(const int TID);

   /// Returns a random integer 0 <= i < To, using getPGMersenneRan()
   static int getPGMersenneRanInt(const int To, const int TID);


   /// Generate a random number from a Gaussian distribution with variance 1 and mean 0.
   static void gaussianRan(double *r1, double *r2);

   /** Generate a random number from a Gaussian distribution with variance 1 and mean 0.
    * Mersenne Twister based version. */
   static void gaussianRan(double *r1, double *r2, const int threadID);

   /** Generate a random number from a Gaussian distribution with variance v (=sigma^2) and mean m.
    * This method was tested for a resulting Gaussian histogram. */
   static void gaussianRan(double *r1, double *r2, const double m, const double v);

   /** Generate a random number from a Gaussian distribution with variance v (=simga^2) and mean m.
    * This method was tested for a resulting Gaussian histogram. This is the thread safe
    * Mersenne Twister based version. */
   static void gaussianRan(double *r1, double *r2, const double m, const double v, const int threadID);

   static double getNormalRan();

   /** Returns a Gaussian random number from the pregenerated array. */
   static double getPGGaussianRan();

   /** Returns a Gaussian random number from the pregenerated array.
    * Mersenne Twister based version. */
   static double getPGGaussianRan(const int threadID);

   static double getPGGaussianRan(const double m, const double v, const int threadID);

   /** Generate a Gamma-distributed random number. */
   static double gammaRan(const double shape, const double theta);
   static double gammaRan(const double shape, const double theta, const int TID);

   /// Generates a boolean variable with chances 50:50 to be true or false.
   static bool generateBool();

   /// Generates a boolean variable with chances 50:50 to be true or false. Mersenne version.
   static bool generateBool(const int ID);

   /// Generate an integer randomly in the given interval from 0 to To where To is EXCLUDED.
   static int generateInt(const int To);

   /// Generate an integer randomly in the given interval from 0 to To where To is EXCLUDED. Mersenne version.
   static int generateInt(const int To, const int TID);

   /** Generates a variable of type double randomly in the given interval. */
   static double generateDouble(const double From, const double To);

   /** Generates a variable of type double randomly in the given interval using the thread safe Mersenne Twister impl.*/
   static double generateDouble(const double From, const double To, const int threadID);

   /** Generates a variable of type double randomly in the given interval using the thread safe PG Mersenne Twister impl.*/
   static double generatePGDouble(const double From, const double To, const int TID);

//    /// Generates a variable of type double randomly in the given interval using the thread safe MT.
//    static double generateDouble(const double To, const int TID);

   /// Generates a variable of type double randomly in the given interval.
   static double generateDouble(const double To);

   /** Generates a random vector in 3d where each component has randomly a value inside the given interval.
    *  @param From left border
    *  @param To right border
    *  @return random 3d vector with respect to the interval */
   static Point genRandomCoordCube(const double From, const double To);

   /** Generates a random vector in 3d where each component has randomly a value inside the given interval.
    *  @param From left border
    *  @param To right border
    *  @param TID
    *  @return random 3d vector with respect to the interval */
   static Point genRandomCoordCube(const double From, const double To, const int TID);

   /** Generates a random vector in 3d where each component has randomly a value inside the given interval. Uses PG Mersenne.
    *  @param From left border
    *  @param To right border
    *  @param TID
    *  @return random 3d vector with respect to the interval */
   static Point genPGRandomCoordCube(const double From, const double To, const int TID);

   /** Generates a random vector in 3d where each component has randomly a value between 0 and the given parameter.
    *  @param To right border
    *  @return random 3d vector with respect to the interval */
   static Point genRandomCoordCube(const double To);

   /** Generates a random vector in 3d where the components have randomly values inside the given interval.
    * @param From coordinates of the left border
    * @param To coordinates of the right border
    * @return random 3d vector with respect to the interval  */
   static Point genRandomCoordCuboid(const Point& From, const Point& To);

   /** Generates a random vector in 3d where the components have randomly values inside the given interval.
    * @param From coordinates of the left border
    * @param To coordinates of the right border
    * @param TID
    * @return random 3d vector with respect to the interval  */
   static Point genRandomCoordCuboid(const Point& From, const Point& To, const int TID);

   /** Generates a random 3d vector inside the given sphere by generating coordinates in a cube and  omitting all
    * coordinates lying outside the sphere.
    * @param Radius the radius of the sphere
    * @return random 3d vector with respect to the sphere  */
   static Point genRandomCoordSphere(const double Radius);

   /** Generates a random 3d vector inside the given sphere by generating coordinates in a cube and  omitting all
    * coordinates lying outside the sphere.
    * @param Radius the radius of the sphere
    * @param TID
    * @return random 3d vector with respect to the sphere  */
   static Point genRandomCoordSphere(const double Radius, const int TID);

   /** Generates randomly oriented 3d vector with length 1.0
    * @return random 3d vector with length 1 */
   static Point genRandomUnitVector();

   /** Generates randomly oriented 3d vector with length 1.0
    * @return random 3d vector with length 1 */
   static Point genRandomUnitVector(const int threadID);

   /** Generates randomly oriented 3d vector with given length
    * @param Length length of the 3d vector
    * @return random 3d vector with fixed length */
   static Point genRandomVector(const double Length);

   /// Get the path of the file name. Memory have to be freed outside this method.
   static std::string getPathName(const char FileName[]);

   /** Prints the given message to std out and exits with exit code 8.
    * @param Message the error message */
   static void error(const char *Message);

   /** Prints the given message to std out and exits with exit code 8.
    * @param Message the error message */
   static void error(const std::string& Message);

   /** Prints the given message to std out as a warning.
    * @param Message the warning message */
   static void warning(const char *Message);

   /// Method which calculates the volume of the sphere with the given radius
   static double sphereVol(const double R);

   /// Method which calculates the surface of the sphere with the given radius
   static double sphereSurf(const double R);

   /// Method which calculates the volume of the cylinder
   static double cylinderVol(const double R, const double Length);

   /// Method which calculates the surface of the cylinder
   static double cylinderSurf(const double R, const double Length);

   /// Theta function
   static double Th(const double x);

   /// Checks if the given point lies in the circle around the origin with radius Rad.
   static bool inCircle(int Rad, Point Pkt);

   /// Numerical zero check (tunable)
   static bool fzero(double x)
   {
      return(fabs(x) < 1.0e-30);
   }

   /// Calculate a factorial (only up to n=7 fast) (tunable)
   static long factorial(int n);

   /** Calculates the radius of a particle species with respect to the given volume fraction and number density.
    * @param eta packing fraction
    * @param rho number density
    * @return the particle radius */
   static double getRadiusFromEta(const double eta, const double rho);

   /** Calculates the packing fraction based on the given radius and number density.
    * @param R the length scale
    * @param rho the number density
    * @return the packing fraction */
   static double getEtaFromRadius(const double R, const double rho);

   /** Rounds the given doubleing point number to the precision specified by p, see source to understand the simple technique. */
   static long roundToPrecision(const double d, const long p);

   /** Shifted sigmoid function which maps the x values [0,1] to [0,1]. a changes the strength.
    *  normalized version. Expensive!, useful to cache, if it's used often. */
   static double sfunction_norm(const double x, const double a);

   /** Shifted exponetial function which maps the x values [0,1] to [0,1]. a changes the strength.
    *  normalized version. Expensive!, useful to cache, if it's used often. */
   static double expfunction_norm(const double x, const double a);

   static bool compareInt(const int i, const int j)
   {
      return(i < j);
   };

   inline static unsigned int getMax3_UI(const unsigned int a[3])
   {
      if (a[0] > a[1])
      {
         if (a[0] > a[2])
         {
            return(a[0]);
         }
         else
         {
            return(a[2]);
         }
      }
      else
      {
         if (a[1] > a[2])
         {
            return(a[1]);
         }
         else
         {
            return(a[2]);
         }
      }
   }

   /** calculates the greatest common divisor of two numbers a and b
    * @param a first number
    * @param b second number
    * @return greatest common divisor
    */
   static int greatestCommonDivisor(int a, int b);

   /** calculates the greatest common divisor in an array of numbers
    *  numbers that are zero are ignored
    * @param numbers array with numbers
    * @param length length of the array numbers
    * @return greatest common divisor
    */
   static int greatestCommonDivisor(const int *numbers, const int length);


   // static members --------------------------------------

   static std::vector <CRandomMersenne> Mersennes;

   static std::vector <std::vector <double> > preGeneratedMersennes;

   static std::vector <unsigned int> currentMersenne;

   static std::vector <double> preGeneratedGaussians;

   static unsigned int currentGaussian;

   static std::vector <std::vector <double> > preGeneratedGaussians_TS;

   static std::vector <unsigned int> currentGaussian_TS;
};
#endif // MONTECARLOUTILS_H_

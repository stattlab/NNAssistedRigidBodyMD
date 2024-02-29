#include "MonteCarloUtils.h"

// Define the static members
std::vector <CRandomMersenne> MonteCarloUtils::Mersennes;

std::vector <std::vector <double> > MonteCarloUtils::preGeneratedMersennes;

std::vector <unsigned int> MonteCarloUtils::currentMersenne;

std::vector <double> MonteCarloUtils::preGeneratedGaussians;

unsigned int MonteCarloUtils::currentGaussian;

std::vector <std::vector <double> > MonteCarloUtils::preGeneratedGaussians_TS;

std::vector <unsigned int> MonteCarloUtils::currentGaussian_TS;


void MonteCarloUtils::setSeed(const std::vector <int>& seeds)
{
   if (MonteCarloUtils::Mersennes.size() > 0)
   {
      // MPIErrorHandling::error("Seeds are already set.");
      std::cout<<"Seeds are already set."<<std::endl;
   }
   for (unsigned int i = 0; i < seeds.size(); i++)
   {
      MonteCarloUtils::Mersennes.push_back(CRandomMersenne(seeds[i]));
   }
   preGeneratedGaussians_TS.resize(seeds.size());
   currentGaussian_TS.resize(seeds.size());
   for (unsigned int i = 0; i < currentGaussian_TS.size(); i++)
   {
      currentGaussian_TS[i] = 0;
   }
   currentGaussian = 0;

   preGeneratedMersennes.resize(seeds.size());
   currentMersenne.resize(seeds.size());
   for (unsigned int i = 0; i < currentMersenne.size(); i++)
   {
      currentMersenne[i] = NoOfPregeneratedMersennes;
      preGeneratedMersennes[i].resize(NoOfPregeneratedMersennes);
   }
}

void MonteCarloUtils::setSeed(const long pseed)
{
   srand48(pseed);
   currentGaussian = 0;
}

void MonteCarloUtils::resetSeed(const std::vector <int>& seeds)
{
   MonteCarloUtils::deleteMersenneSeeds();
   MonteCarloUtils::setSeed(seeds);
}

void MonteCarloUtils::deleteMersenneSeeds()
{
   Mersennes.clear();
   preGeneratedGaussians_TS.clear();
   currentGaussian_TS.clear();
   preGeneratedMersennes.clear();
   currentMersenne.clear();
}

double MonteCarloUtils::ran()
{
   //printf("RandomNumber ran\n");
   return(drand48());
}

double MonteCarloUtils::ran(const int TID)
{
   /* faster than with push_back, but slower than without precaching
    * if(MonteCarloUtils::currentMersenne[TID]>=preGeneratedMersennes[TID].size()) {
    *  MonteCarloUtils::currentMersenne[TID] = 0;
    *  //MonteCarloUtils::preGeneratedMersennes[TID].clear();
    *  for (int i = 0; i < NoOfPregeneratedMersennes; i++) {
    *      MonteCarloUtils::preGeneratedMersennes[TID][i]=Mersennes[TID].Random();
    *  }
    * }
    * return MonteCarloUtils::preGeneratedMersennes[TID][currentMersenne[TID]++];
    *
    * old version
    * if(MonteCarloUtils::currentMersenne[TID]>=preGeneratedMersennes[TID].size()) {
    *  // make new random numbers
    *  //printf("RandomNumber PG New Random Numbers!\n");
    *  MonteCarloUtils::currentMersenne[TID] = 0;
    *  MonteCarloUtils::preGeneratedMersennes[TID].clear();
    *  while(MonteCarloUtils::preGeneratedMersennes[TID].size() < NoOfPregeneratedMersennes) {
    *      MonteCarloUtils::preGeneratedMersennes[TID].push_back(Mersennes[TID].Random());
    *  }
    * }
    * //printf("RandomNumber PG %15.15f\n",preGeneratedMersennes[TID][currentMersenne[TID]]);
    * return MonteCarloUtils::preGeneratedMersennes[TID][currentMersenne[TID]++];
    */

   return(Mersennes[TID].Random());
}

double MonteCarloUtils::getPGMersenneRan(const int TID)
{
//    Covered by unittests now. Seems to work fine.
//    MonteCarloUtils::warning("In getPGMersenneRan, the array is not filled during the first run, resulting in 0 for the first random number. Afterwards it should work fine.");
//    MonteCarloUtils::warning("This should be fixed by changing the start value of currentMersenne (Not tested)");
   if (MonteCarloUtils::currentMersenne[TID] >= preGeneratedMersennes[TID].size())
   {
      // make new random numbers
      //printf("RandomNumber PG New Random Numbers!\n");
      MonteCarloUtils::currentMersenne[TID] = 0;
      MonteCarloUtils::preGeneratedMersennes[TID].clear();
      while (MonteCarloUtils::preGeneratedMersennes[TID].size() < NoOfPregeneratedMersennes)
      {
         MonteCarloUtils::preGeneratedMersennes[TID].push_back(Mersennes[TID].Random());
      }
   }
   //printf("RandomNumber PG %15.15f\n",preGeneratedMersennes[TID][currentMersenne[TID]]);
   return(MonteCarloUtils::preGeneratedMersennes[TID][currentMersenne[TID]++]);
}

bool MonteCarloUtils::getPGMersenneRanBool(const int TID)
{
   if (getPGMersenneRan(TID) < 0.5)
   {
      return(true);
   }
   else
   {
      return(false);
   }
}

int MonteCarloUtils::getPGMersenneRanInt(const int To, const int TID)
{
   return(int( getPGMersenneRan(TID) * (To))); // Dieser cast schliesst To aus! getestet mit gcc O0, O3, intel O3,aggressive
}

void MonteCarloUtils::gaussianRan(double *r1, double *r2)
{
   // Polar Methode
   double a1 = generateDouble(-1.0, 1.0);
   double a2 = generateDouble(-1.0, 1.0);
   double q  = a1 * a1 + a2 * a2;

   while (q <= 0.0 || q > 1.0)
   {
      a1 = generateDouble(-1.0, 1.0);
      a2 = generateDouble(-1.0, 1.0);
      q  = a1 * a1 + a2 * a2;
   }
   const double p = sqrt(-2.0 * log(q) / q);

   *r1 = a1 * p;
   *r2 = a2 * p;
}

double MonteCarloUtils::getNormalRan()
{
   static bool  saved  = false;
   static double r_save = 0;
   double        normalran;

   if (saved)
   {
      normalran = r_save;
      saved     = false;
   }
   else
   {
      double a1 = ran();
      double a2 = ran();
      normalran = sqrt(-2.0 * log(a1)) * cos(2.0 * M_PI * a2);
      r_save    = sqrt(-2.0 * log(a1)) * sin(2.0 * M_PI * a2);
      saved     = true;
   }
   return(normalran);
}

void MonteCarloUtils::gaussianRan(double *r1, double *r2, const int TID)
{
   // Polar Methode
   double a1 = generateDouble(-1.0, 1.0, TID);
   double a2 = generateDouble(-1.0, 1.0, TID);
   double q  = a1 * a1 + a2 * a2;

   while (q <= 0.0 || q > 1.0)
   {
      a1 = generateDouble(-1.0, 1.0, TID);
      a2 = generateDouble(-1.0, 1.0, TID);
      q  = a1 * a1 + a2 * a2;
   }
   const double p = sqrt(-2.0 * log(q) / q);

   //const double p = sqrt(-2.0*lut_log[int(q*4095)]/q);
   *r1 = a1 * p;
   *r2 = a2 * p;
}

void MonteCarloUtils::gaussianRan(double *r1, double *r2, const double m, const double v)
{
   // Polar Methode
   const double sqrt_v = sqrt(v);
   double       a1     = generateDouble(-1.0, 1.0);
   double       a2     = generateDouble(-1.0, 1.0);
   double       q      = a1 * a1 + a2 * a2;

   while (q <= 0.0 || q > 1.0)
   {
      a1 = generateDouble(-1.0, 1.0);
      a2 = generateDouble(-1.0, 1.0);
      q  = a1 * a1 + a2 * a2;
   }
   const double p = sqrt(-2.0 * log(q) / q);

   *r1 = a1 * p * sqrt_v + m;
   *r2 = a2 * p * sqrt_v + m;
}

void MonteCarloUtils::gaussianRan(double *r1, double *r2, const double m, const double v, const int TID)
{
   // Polar Methode
   const double sqrt_v = sqrt(v);
   double       a1     = generateDouble(-1.0, 1.0, TID);
   double       a2     = generateDouble(-1.0, 1.0, TID);
   double       q      = a1 * a1 + a2 * a2;

   while (q <= 0.0 || q > 1.0)
   {
      a1 = generateDouble(-1.0, 1.0, TID);
      a2 = generateDouble(-1.0, 1.0, TID);
      q  = a1 * a1 + a2 * a2;
   }
   const double p = sqrt(-2.0 * log(q) / q);

   *r1 = a1 * p * sqrt_v + m;
   *r2 = a2 * p * sqrt_v + m;
}

double MonteCarloUtils::getPGGaussianRan()
{
   if (MonteCarloUtils::currentGaussian < preGeneratedGaussians.size())
   {
      return(MonteCarloUtils::preGeneratedGaussians[currentGaussian++]);
   }
   else
   {
      MonteCarloUtils::currentGaussian = 0;
      MonteCarloUtils::preGeneratedGaussians.clear();
      while (MonteCarloUtils::preGeneratedGaussians.size() < NoOfPregeneratedGaussians)
      {
         double x, y;
         gaussianRan(&x, &y);
         MonteCarloUtils::preGeneratedGaussians.push_back(x);
         MonteCarloUtils::preGeneratedGaussians.push_back(y);
      }
      return(MonteCarloUtils::preGeneratedGaussians[currentGaussian++]);
   }
}

double MonteCarloUtils::getPGGaussianRan(const int TID)
{
   if (MonteCarloUtils::currentGaussian_TS[TID] < preGeneratedGaussians_TS[TID].size())
   {
      return(MonteCarloUtils::preGeneratedGaussians_TS[TID][currentGaussian_TS[TID]++]);
   }
   else
   {
      MonteCarloUtils::currentGaussian_TS[TID] = 0;
      MonteCarloUtils::preGeneratedGaussians_TS[TID].clear();
      while (MonteCarloUtils::preGeneratedGaussians_TS[TID].size() < NoOfPregeneratedGaussians)
      {
         double x, y;
         gaussianRan(&x, &y, TID);
         MonteCarloUtils::preGeneratedGaussians_TS[TID].push_back(x);
         MonteCarloUtils::preGeneratedGaussians_TS[TID].push_back(y);
      }
      return(MonteCarloUtils::preGeneratedGaussians_TS[TID][currentGaussian_TS[TID]++]);
   }
}

double MonteCarloUtils::getPGGaussianRan(const double m, const double v, const int threadID)
{
   return(getPGGaussianRan(threadID) * sqrt(v) + m);
}

double MonteCarloUtils::gammaRan(const double shape, const double theta)
{
   const double d = shape - 1.0 / 3.0;
   const double c = 1.0 / sqrt(9.0 * d);

   while (true)
   {
      double v, x;
      do
      {
         x = getPGGaussianRan();
         v = pow(1.0 + c * x, 3);
      }while(v <= 0.0);

      const double u = ran();
      if (u < 1.0 - 0.0331 * pow(x, 4))
      {
         return(d * v * theta);
      }
      else
      {
         if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v)))
         {
            return(d * v * theta);
         }
      }
   }
}

double MonteCarloUtils::gammaRan(const double shape, const double theta, const int TID)
{
   const double d = shape - 1.0 / 3.0;
   const double c = 1.0 / sqrt(9.0 * d);

   while (true)
   {
      double v, x;
      do
      {
         x = getPGGaussianRan(TID);
         v = pow(1.0 + c * x, 3);
      }while(v <= 0.0);

      const double u = ran(TID);
      if (u < 1.0 - 0.0331 * pow(x, 4))
      {
         return(d * v * theta);
      }
      else
      {
         if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v)))
         {
            return(d * v * theta);
         }
      }
   }
}

bool MonteCarloUtils::generateBool()
{
   if (ran() < 0.5)
   {
      return(true);
   }
   else
   {
      return(false);
   }
}

bool MonteCarloUtils::generateBool(const int TID)
{
   if (ran(TID) < 0.5)
   {
      return(true);
   }
   else
   {
      return(false);
   }
}

int MonteCarloUtils::generateInt(const int To)
{
   return(int( ran() * (To))); // Dieser cast schliesst To aus! getestet mit gcc O0, O3, intel O3,aggressive
}

int MonteCarloUtils::generateInt(const int To, const int TID)
{
   return(int( ran(TID) * (To))); // Dieser cast schliesst To aus! getestet mit gcc O0, O3, intel O3,aggressive
}

double MonteCarloUtils::generateDouble(const double From, const double To)
{
   return(ran() * (To - From) + From);
}

double MonteCarloUtils::generateDouble(const double From, const double To, const int TID)
{
   return(ran(TID) * (To - From) + From);
}

double MonteCarloUtils::generatePGDouble(const double From, const double To, const int TID)
{
   return(getPGMersenneRan(TID) * (To - From) + From);
}

/**
 * double MonteCarloUtils::generateDouble(const double To, const int TID){
 * return  ran(TID)*To;
 * }
 **/

double MonteCarloUtils::generateDouble(const double To)
{
   return(ran() * To);
}

Point MonteCarloUtils::genRandomCoordCube(const double From, const double To)
{
   double p[3];

   for (short i = 0; i < 3; i++)
   {
      p[i] = MonteCarloUtils::generateDouble(From, To);
   }
   return(Point(p));
}

Point MonteCarloUtils::genRandomCoordCube(const double From, const double To, const int TID)
{
   double p[3];

   for (short i = 0; i < 3; i++)
   {
      p[i] = MonteCarloUtils::generateDouble(From, To, TID);
   }
   return(Point(p));
}

Point MonteCarloUtils::genRandomCoordCube(const double To)
{
   double p[3];

   for (short i = 0; i < 3; i++)
   {
      p[i] = MonteCarloUtils::generateDouble(To);
   }

   Point P(p[0], p[1], p[2]);

   return(P);
}

Point MonteCarloUtils::genRandomCoordCuboid(const Point& From, const Point& To)
{
   double p[3];

   for (short i = 2; i >= 0; i--)
   {
      p[i] = MonteCarloUtils::generateDouble(From.x[i], To.x[i]);
   }

   Point P(p[0], p[1], p[2]);

   return(P);
}

Point MonteCarloUtils::genRandomCoordCuboid(const Point& From, const Point& To, const int TID)
{
   double p[3];

   for (short i = 2; i >= 0; i--)
   {
      p[i] = MonteCarloUtils::generateDouble(From.x[i], To.x[i], TID);
   }
   return(Point(p));
}

Point MonteCarloUtils::genRandomCoordSphere(const double Radius)
{
   Point P(2.0 * ran() - 1.0, 2.0 * ran() - 1.0, 2.0 * ran() - 1.0);

   while (P.getSquare() >= 1.0)
   {
      P = Point(2.0 * ran() - 1.0, 2.0 * ran() - 1.0, 2.0 * ran() - 1.0);
   }
   P *= Radius;
   return(P);
}

Point MonteCarloUtils::genRandomCoordSphere(const double Radius, const int TID)
{
   Point P(2.0 * ran(TID) - 1.0, 2.0 * ran(TID) - 1.0, 2.0 * ran(TID) - 1.0);

   while (P.getSquare() >= 1.0)
   {
      P = Point(2.0 * ran(TID) - 1.0, 2.0 * ran(TID) - 1.0, 2.0 * ran(TID) - 1.0);
   }
   P *= Radius;
   return(P);
}

Point MonteCarloUtils::genRandomUnitVector()
{
   double ran1, ran2, ransq, ranh;

   ransq = 2.0;
   while (ransq >= 1)
   {
      ran1  = 1.0 - 2.0 * ran();
      ran2  = 1.0 - 2.0 * ran();
      ransq = ran1 * ran1 + ran2 * ran2;
   }
   ranh = 2.0 * sqrt(1.0 - ransq);

   Point Result(ran1 * ranh, ran2 * ranh, 1 - 2.0 * ransq);

   return(Result);
}

Point MonteCarloUtils::genRandomUnitVector(const int threadID)
{
   double ran1, ran2, ransq, ranh;

   ransq = 2.0;
   while (ransq >= 1)
   {
      ran1  = 1.0 - 2.0 * ran(threadID);
      ran2  = 1.0 - 2.0 * ran(threadID);
      ransq = ran1 * ran1 + ran2 * ran2;
   }
   ranh = 2.0 * sqrt(1.0 - ransq);

   Point Result(ran1 * ranh, ran2 * ranh, 1 - 2.0 * ransq);

   return(Result);
}

Point MonteCarloUtils::genRandomVector(const double Length)
{
   Point Result = genRandomUnitVector() * Length;

   return(Result);
}

std::string MonteCarloUtils::getPathName(const char DataFileName[])
{
   int i = std::strlen(DataFileName) - 1;

   while (DataFileName[i] != '/' && i >= 0)
   {
      i--;
   }

   char *path = new char[512];

   if (i != 0)
   {
      for (int j = 0; j < i; j++)
      {
         path[j] = DataFileName[j];
      }
      path[i] = 0x00;
   }
   std::string Result(path);

   delete[] path;

   return(Result);
}

void MonteCarloUtils::error(const char *text)
{
   std::cout << "# ERROR: " << text << std::endl;
   std::cout << "# ***************************************************************************\n";
   std::cerr << "# ERROR: " << text << std::endl;
   std::cerr << "# ***************************************************************************\n";
   exit(8);
}

void MonteCarloUtils::error(const std::string& text)
{
   std::cout << "# ERROR: " << text << std::endl;
   std::cout << "# ***************************************************************************\n";
   std::cerr << "# ERROR: " << text << std::endl;
   std::cerr << "# ***************************************************************************\n";
   exit(8);
}

void MonteCarloUtils::warning(const char *text)
{
   std::cout << "# WARNING: " << text << std::endl;
   std::cerr << "# WARNING: " << text << std::endl;
}

double MonteCarloUtils::sphereVol(const double R)
{
   return(4.0 * M_PI * R * R * R / 3.0);
}

double MonteCarloUtils::sphereSurf(const double R)
{
   return(4.0 * M_PI * R * R);
}

double MonteCarloUtils::cylinderVol(const double R, const double Length)
{
   return(M_PI * R * R * Length);
}

double MonteCarloUtils::cylinderSurf(const double R, const double Length)
{
   return(2.0 * M_PI * R * Length);
}

double MonteCarloUtils::sfunction_norm(const double x, const double a)
{
   return(1.0 / (1.0 - cosh(a) + sinh(a) / tanh(a * x)));
}

double MonteCarloUtils::expfunction_norm(const double x, const double a)
{
   // expm1(x) -> exp(x) - 1.0 without loss of precision!
   return((expm1(a * x)) / (expm1(a)));
}

double MonteCarloUtils::Th(const double x)
{
   if (x >= 0)
   {
      return(1.0);
   }
   else
   {
      return(0.0);
   }
}

double MonteCarloUtils::getRadiusFromEta(const double eta, const double rho)
{
   return(pow(1.9098593171 * eta / rho, 0.333333333333));
}

double MonteCarloUtils::getEtaFromRadius(const double R, const double rho)
{
   return(0.5235987756 * rho * pow(R, 3.0));
}

long MonteCarloUtils::roundToPrecision(const double d, const long p)
{
   return(long(d * double(p)));
}

int MonteCarloUtils::greatestCommonDivisor(int a, int b)
{
   int gcd = b;

   if (b == 0)
   {
      return(0);
   }
   while (a != 0)
   {
      gcd = a;
      a   = b % a;
      b   = gcd;
   }
   return(gcd);
}

int MonteCarloUtils::greatestCommonDivisor(const int *numbers, const int length)
{
   int i = 0, gcd = 0;

   // find first nonzero number
   while (gcd == 0)
   {
      gcd = numbers[i++];
   }

   // go on from ith number
   for (; i < length; i++)
   {
      gcd = greatestCommonDivisor(gcd, numbers[i]);
   }
   return(gcd);
}

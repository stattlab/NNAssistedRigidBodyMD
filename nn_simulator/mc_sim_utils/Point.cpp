#include "Point.h"

// =====================================================================================================
//
// get-Funktionen und Tools
//
// =====================================================================================================

void Point::mirror(const Point& A, const Point& V)
{
   // Über Drehung um 180°
   // Drehmatrix-Elemente vorbereiten
   *this -= A;
   const double absV           = V.getDistToOrigin(); // Wurzel!
   const Point VUnit          = V / absV;
   const double two_v1_sq_min1 = VUnit.x[0] * VUnit.x[0] + VUnit.x[0] * VUnit.x[0] - 1.0f;
   const double two_v12        = VUnit.x[0] * VUnit.x[1] + VUnit.x[0] * VUnit.x[1];
   const double two_v13        = VUnit.x[0] * VUnit.x[2] + VUnit.x[0] * VUnit.x[2];
   const double two_v2_sq_min1 = VUnit.x[1] * VUnit.x[1] + VUnit.x[1] * VUnit.x[1] - 1.0f;
   const double two_v23        = VUnit.x[1] * VUnit.x[2] + VUnit.x[1] * VUnit.x[2];
   const double two_v3_sq_min1 = VUnit.x[2] * VUnit.x[2] + VUnit.x[2] * VUnit.x[2] - 1.0f;

   // Multiplikation durchführen
   Point Result(this->x[0] * two_v1_sq_min1 + this->x[1] * two_v12 + this->x[2] * two_v13,
                this->x[0] * two_v12 + this->x[1] * two_v2_sq_min1 + this->x[2] * two_v23,
                this->x[0] * two_v13 + this->x[1] * two_v23 + this->x[2] * two_v3_sq_min1);

   Result += A;
   *this   = Result;
}

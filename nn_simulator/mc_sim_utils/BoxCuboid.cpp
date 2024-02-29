#include "BoxCuboid.h"

using namespace linalg::aliases;

BoxCuboid::BoxCuboid(const Point& pSysDim) : Box(pSysDim)
{
   m_periodic = {true,true,true};
}


BoxCuboid::~BoxCuboid()
{
}

void BoxCuboid::usePBConditions(Point *P)
{
   for (short i = 2; i >= 0; i--)
   {
     double dx = P->x[i] + HalfDim.x[i];
     dx = dx - floor(dx*SysDimInv.x[i])*SysDim.x[i];
     P->x[i] = dx - HalfDim.x[i];
   }
}

// void BoxCuboid::usePBC(linalg::aliases::double3 *P)
// {
//    for (short i = 2; i >= 0; i--)
//    {
//      // double dx = P->x[i] + HalfDim.x[i];
//      // dx = dx - floor(dx*SysDimInv.x[i])*SysDim.x[i];
//      // P->x[i] = dx - HalfDim.x[i];
//      double dx = P[i] + HalfDim.x[i];
//      dx = dx - floor(dx*SysDimInv.x[i])*SysDim.x[i];
//      P[i] = dx - HalfDim.x[i];
//    }
// }

bool BoxCuboid::checkIfInBox(const Particle&) const
{
   return(true);
}

bool BoxCuboid::checkIfInBox(const Point&) const
{
   return(true);
}

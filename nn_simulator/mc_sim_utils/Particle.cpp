#include "Particle.h"



Particle::Particle(const Point& pPos, unsigned int Type) :
   Origin(pPos), m_Box(0), marked(false), m_Type(Type)
{
}

Particle::Particle(const Point& pPos, Box *pBox, unsigned int Type) :
    Origin(pPos), marked(false), m_Type(Type)
{
   m_Box = pBox;
}


Box *Particle::getBoxPointer() const
{
   return(m_Box);
}

void Particle::setBoxPointer(Box *newBox)
{
   m_Box = newBox;
}

double Particle::getX(const short i) const
{
   return(Origin.x[i]);
}

void Particle::setX(const short c, const double v)
{
   Origin.x[c] = v;
}

unsigned int Particle::getType() const

{
   return(m_Type);
}


void Particle::setType(unsigned int newType)
{
   m_Type = newType;
}

const Point& Particle::getOrigin() const
{
   return(Origin);
}

void Particle::setOrigin(const Point& newOrigin)
{
   Origin = newOrigin;
}

void Particle::translate(const double amount)
{
    std::cout<<"Needs random number generator class don't use"<<std::endl;
    exit(0);
   // Point Trans       = MonteCarloUtils::genRandomCoordCube(-amount, amount, 0);
   // Point NewPosition = Origin + Trans;
   //
   // m_Box->usePBConditions(&NewPosition);
   // Origin = NewPosition;
}

void Particle::translate(const Point& Trans)
{
   Point NewPosition = Origin + Trans;

   m_Box->usePBConditions(&NewPosition);
   Origin = NewPosition;
}

void Particle::rotate(double **rotation_matrix, const Point& center)
{
   Point RelPos = Origin - center;
   m_Box->usePBConditions(&RelPos);
   Point NewRelPos;
   NewRelPos.setX(rotation_matrix[0][0]*RelPos.getX() + rotation_matrix[0][1]*RelPos.getY() + rotation_matrix[0][2]*RelPos.getZ());
   NewRelPos.setY(rotation_matrix[1][0]*RelPos.getX() + rotation_matrix[1][1]*RelPos.getY() + rotation_matrix[1][2]*RelPos.getZ());
   NewRelPos.setZ(rotation_matrix[2][0]*RelPos.getX() + rotation_matrix[2][1]*RelPos.getY() + rotation_matrix[2][2]*RelPos.getZ());

   Point NewPos = NewRelPos + center;
   m_Box->usePBConditions(&NewPos);
   Origin = NewPos;
}

void Particle::rotate_WOpbc(double **rotation_matrix, const Point& center)
{
   Point RelPos = Origin - center;
   Point NewRelPos;
   NewRelPos.setX(rotation_matrix[0][0]*RelPos.getX() + rotation_matrix[0][1]*RelPos.getY() + rotation_matrix[0][2]*RelPos.getZ());
   NewRelPos.setY(rotation_matrix[1][0]*RelPos.getX() + rotation_matrix[1][1]*RelPos.getY() + rotation_matrix[1][2]*RelPos.getZ());
   NewRelPos.setZ(rotation_matrix[2][0]*RelPos.getX() + rotation_matrix[2][1]*RelPos.getY() + rotation_matrix[2][2]*RelPos.getZ());

   Point NewPos = NewRelPos + center;
   Origin = NewPos;
}

void Particle::mirror(const Point& V)
{
   Origin.mirror(V);
   m_Box->usePBConditions(&Origin);
}

void Particle::mirror_XY(const Point& V)
{
   Origin.mirror_XY(V);
   m_Box->usePBConditions(&Origin);
}

void Particle::mirror_Z(const double V)
{
   Origin.mirror_Z(V);
   m_Box->usePBConditions(&Origin);
}

bool Particle::isMarked() const
{
   return(marked);
}

void Particle::mark()
{
   marked = true;
}

void Particle::unMark()
{
   marked = false;
}

Point Particle::getDist(const Particle& Particle) const
{
   Point d = Particle.getOrigin() - Origin;

   m_Box->usePBConditions(&d);
   return(d);
}



void Particle::printObject(const int InfLevel) const
{
   switch (InfLevel)
   {
   case 1: {
      Origin.printObject();
      break;
   }

   case 2: {
      std::cout << "# Origin: ";
      Origin.printObject();
      break;
   }

   case 4: {
    //  std::cout << "# next=" << next << std::endl;
  //    std::cout << "# prev=" << prev << std::endl;
    //  std::cout << "# cell=" << cell << std::endl;
      std::cout << "# Origin: ";
      Origin.printObject();
      break;
   }
   }
}

void Particle::printObject(FILE *File, const int Style, const int Type) const
{
   switch (Style)
   {
   case 0: {  // VMD-Style
      fprintf(File, "%f %f %f \n", Origin.x[0], Origin.x[1], Origin.x[2]);
      break;
   }

   case 1: {  // Qmga-Style
      fprintf(File, "%f %f %f 0 0 0 1 %d %d %d %d %d %d %d\n", Origin.x[0], Origin.x[1], Origin.x[2], -Type, Type, Type, Type, Type, Type, Type);
      break;
   }
   }
}

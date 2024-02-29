#include "CommandLineArgumentParser.h"

CommandLineArgumentParser::CommandLineArgumentParser(int&argc, char **argv)
{
   for (int i = 1; i < argc; ++i)
   {
      this->tokens.push_back(std::string(argv[i]));
   }
}

const std::string& CommandLineArgumentParser::getCmdOption(const std::string&option) const
{
   std::vector <std::string>::const_iterator itr;

   itr = std::find(this->tokens.begin(), this->tokens.end(), option);
   if (itr != this->tokens.end() && ++itr != this->tokens.end())
   {
      return(*itr);
   }
   static const std::string empty_string("");

   return(empty_string);
}

bool CommandLineArgumentParser::cmdOptionExists(const std::string&option) const
{
   return(std::find(this->tokens.begin(), this->tokens.end(), option)
          != this->tokens.end());
}

bool CommandLineArgumentParser::setInitFileName(const std::string InitFileName)
{
   std::ifstream infile(InitFileName);

   if (infile.good())
   {
      m_initfilename = InitFileName;
      return(true);
   }
   else
   {
      return(false);
   }
   return(false);
}

bool CommandLineArgumentParser::setInputKt(const std::string temp_str)
{
  float num_float = std::stof(temp_str);
  m_input_temp = num_float;
  return(true);
}

bool CommandLineArgumentParser::setInputDt(const std::string temp_str)
{
  float num_float = std::stof(temp_str);
  m_input_timestepsize = num_float;
  return(true);
}

bool CommandLineArgumentParser::setInputDf(const std::string temp_str)
{
  int num_float = std::stoi(temp_str);
  m_input_dump_frequency = num_float;
  return(true);
}

bool CommandLineArgumentParser::setInputTs(const std::string temp_str)
{
  int num_float = std::stoi(temp_str);
  m_input_timesteps = num_float;
  return(true);
}

bool CommandLineArgumentParser::setInputmN(const std::string temp_str)
{
  int num_float = std::stoi(temp_str);
  m_model_number = num_float;
  return(true);
}

void CommandLineArgumentParser::printHelp(void) const
{
   std::cout<<"Authors:  Antonia Statt "<<std::endl;    // authors in alphabetical order
   std::cout<<"Mandatory options: ---------------------------------------------------------"<<std::endl;
   std::cout<<"-o <OutputBaseFileName>"<<std::endl;
   std::cout<<"   The base file name (e.g. the start of the name) of all files created during this run."<<std::endl;
   std::cout<<"-kT <Temperature>"<<std::endl;
   std::cout<<"   The path to the parameter file."<<std::endl;
   std::cout<<"-ts <Timesteps>"<<std::endl;
   std::cout<<"   The total number of timesteps to run the simulation"<<std::endl;
   std::cout<<"-df <DumpFrequency>"<<std::endl;
   std::cout<<"   Every df dump a gsd file."<<std::endl;
   std::cout<<"-dt <Timestepsize>"<<std::endl;
   std::cout<<"   Time step size."<<std::endl;
   std::cout<<"-i <InitFile>"<<std::endl;
   std::cout<<"   The gsd file that contains the initial configuration for the simulation."<<std::endl;
   std::cout<<"-mN <Model>"<<std::endl;
   std::cout<<"   the neural net model number that will be used ."<<std::endl;
   std::cout<<"Additional options: --------------------------------------------------------"<<std::endl;
   std::cout<<"-h"<<std::endl;
   std::cout<<"   Show this help"<<std::endl;
}

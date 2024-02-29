#ifndef COMMAND_LINE_ARGUMENT_PARSER_H
#define COMMAND_LINE_ARGUMENT_PARSER_H

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

/**
 *  @brief Parses the arguments from the command line
 *
 *
 */
class CommandLineArgumentParser {
public:
   CommandLineArgumentParser(int&argc, char **argv);

   const std::string& getCmdOption(const std::string&option) const;

   bool cmdOptionExists(const std::string&option) const;

   void printHelp() const;


   std::string getInitFileName()
   {
      return(m_initfilename);
   }

   std::string getOutputBaseFileName()
   {
      return(m_outputfilename);
   }

   float getInputTemp()
   {
      return(m_input_temp);
   }

   float getInputDt()
   {
      return(m_input_timestepsize);
   }

   int getInputTs()
   {
      return(m_input_timesteps);
   }

   int getInputDf()
   {
      return(m_input_dump_frequency);
   }

   int getInputmN()
   {
      return(m_model_number);
   }

   bool setOutputBaseFileName(const std::string OutputBaseFileName)
   {
      m_outputfilename = OutputBaseFileName;
      return(true);
   }

   bool setInitFileName(const std::string InitFileName);
   bool setInputKt(const std::string InitFileName);
   bool setInputDt(const std::string InitFileName);
   bool setInputDf(const std::string InitFileName);
   bool setInputTs(const std::string InitFileName);
   bool setInputmN(const std::string InitFileName);

private:
   std::vector <std::string> tokens;
   std::string m_initfilename;
   std::string m_outputfilename;
   float m_input_temp;
   float m_input_timestepsize;
   int m_input_timesteps;
   int m_input_dump_frequency;
   int m_model_number;
};


#endif /* COMMAND_LINE_ARGUMENT_PARSER_H */

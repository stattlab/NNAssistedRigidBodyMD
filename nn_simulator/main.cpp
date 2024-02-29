#pragma GCC diagnostic ignored "-Wdeprecated-declarations" // just to ignore deprecated warnings

#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0


#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <cmath>
#include <vector>
#include <ctime>
#include <algorithm>
#include <sys/stat.h>

#include "gsd_utils/GSDReader.h"
#include "gsd_utils/GSDWriter.h"
#include "gsd_utils/gsd.h"
#include "CommandLineArgumentParser.h"
#include "Sim.h"



#include <chrono>
// using namespace nvinfer1;
// const std::string gSampleName = "TensorRT.sample_onnx_mnist";
// Basic filenames
std::string CBON;
std::string CBCN;

// using namespace linalg::aliases;

// command line argument parsing - all paramters from command line
CommandLineArgumentParser *CommandLineInput;
void CommandLineInput_Handler(int argc, char *argv[]);
time_t startSimulationTime = time(0);
time_t endSimulationTime = time(0);

class InputParser{
    public:
        InputParser (int &argc, char **argv){
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }

        const std::string& getCmdOption(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }

        bool cmdOptionExists(const std::string &option) const{
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }
    private:
        std::vector <std::string> tokens;
};

int main(int argc, char* argv[])
{
    /// command line input reading stuff, added after vacation
    CommandLineInput_Handler(argc, argv);
    mkdir("out", 0755);
    std::string BaseOutName = CommandLineInput->getOutputBaseFileName();
    std::string init_filename = CommandLineInput->getInitFileName();
    float kT = CommandLineInput->getInputTemp();
    float dt = CommandLineInput->getInputDt();
    int timesteps = CommandLineInput->getInputTs();
    int dump_frequency = CommandLineInput->getInputDf();
    int model_number = CommandLineInput->getInputmN();
    int         line_size   = BaseOutName.size();
    // CBON = "out/" + BaseOutName;
    std::string output_filename = "out/" + BaseOutName + ".gsd";
    std::cout<<"Init gsd file: "<<init_filename<<std::endl;
    std::cout<<"Temperature: "<<kT<<std::endl;
    std::cout<<"Timestep size: "<<dt<<std::endl;
    std::cout<<"Total timesteps: "<<timesteps<<std::endl;
    std::cout<<"Dump frequency: "<<dump_frequency<<std::endl;
    std::cout<<"Output filename: "<<output_filename<<std::endl;
    std::cout<<"Model Number: "<<model_number<<std::endl;


    int t = 0;
    auto reader2 = std::make_shared<GSDReader>(init_filename, 0, false);
    std::shared_ptr< SnapshotSystemData<float>> snap = reader2->getSnapshot();



    Sim simulator;
    simulator.setInitConfig(snap);
    simulator.setNlistCutoff(6.30 + 2.0);
    simulator.setPotentialPairs();
    simulator.setCutoff(6.30);
    simulator.setDt(dt);
    simulator.setkT(kT);
    simulator.setModelNumber(model_number);
    simulator.setOutname(output_filename);
    simulator.buildNNs();

    unsigned int N_timesteps = timesteps;
    unsigned int refresh_Nlist_every = 100;
    unsigned int dump_every = dump_frequency;
    unsigned int N_dump = N_timesteps/dump_every;
    unsigned int count = 0;

    time(&startSimulationTime);
    std::cout << "# Time SimulationBegin " << ctime(&startSimulationTime);


    for (unsigned int ts = 0; ts < N_timesteps; ts++)
    {
      if(ts%refresh_Nlist_every==0)
      {
        simulator.refreshNList();
      }
      if(ts%dump_every==0)
      {
        simulator.dumpGsd();
        std::cout<<"**************   "<<count<<" / "<<N_dump<<" *****************************"<<std::endl;
        time_t current_time = time(0);
        time(&current_time);
        // TODO set up chrono to show TPS instead 
        std::cout << "# Current Time " << ctime(&current_time);
        count++;
      }
      simulator.nvt_integrate();
    }
    std::cout<<"Simulation Completed"<<std::endl;
    time(&endSimulationTime);
    std::cout << "# Time SimulationEnds " << ctime(&endSimulationTime);

    return 0;
}



void CommandLineInput_Handler(int argc, char *argv[])
{
   CommandLineInput = new CommandLineArgumentParser(argc, argv);

   //parse help option
   if (CommandLineInput->cmdOptionExists("-h"))
   {
      CommandLineInput->printHelp();

      exit(EXIT_SUCCESS);
   }

   //check for existence of mandatory command line arguments

   if (!CommandLineInput->cmdOptionExists("-i"))
   {
      std::cout<<"Missing mandatory command line argument -i <InitFile>"<<std::endl;
      CommandLineInput->printHelp();
      exit(EXIT_FAILURE);
   }

   if (!CommandLineInput->cmdOptionExists("-o"))
   {
      std::cout<<"Missing mandatory command line argument -o <Output>"<<std::endl;
      CommandLineInput->printHelp();
      exit(EXIT_FAILURE);
   }

   bool success = false;
   const std::string&initfilename = CommandLineInput->getCmdOption("-i");

   if (!initfilename.empty())
   {
      success = CommandLineInput->setInitFileName(initfilename);
   }
   if (!success)
   {
      std::cout<<"Bad filename for -i argument. Specify init file name/path."<<std::endl;
      CommandLineInput->printHelp();
      exit(EXIT_FAILURE);
   }

   success = false;
   const std::string&model_number = CommandLineInput->getCmdOption("-mN");

   if (!model_number.empty())
   {
      success = CommandLineInput->setInputmN(model_number);
   }
   if (!success)
   {
      std::cout<<"Use an integer -mN argument."<<std::endl;
      CommandLineInput->printHelp();
      exit(EXIT_FAILURE);
   }

   success = false;
   const std::string&outputfilename = CommandLineInput->getCmdOption("-o");

   if (!outputfilename.empty())
   {
      success = CommandLineInput->setOutputBaseFileName(outputfilename);
   }
   if (!success)
   {
      std::cout<<"Bad filename for -o argument. Specify file name/path relative to ./out or ./cont ."<<std::endl;
      CommandLineInput->printHelp();
      exit(EXIT_FAILURE);
   }

   success = false;
   const std::string&temp_str = CommandLineInput->getCmdOption("-kT");

   if (!temp_str.empty())
   {
      success = CommandLineInput->setInputKt(temp_str);
   }
   if (!success)
   {
      std::cout<<"Use a float to specify the temperature with -kT argument"<<std::endl;
      CommandLineInput->printHelp();
      exit(EXIT_FAILURE);
   }
   const std::string&dt_str = CommandLineInput->getCmdOption("-dt");
   success = CommandLineInput->setInputDt(dt_str);

   const std::string&ts_str = CommandLineInput->getCmdOption("-df");
   success = CommandLineInput->setInputDf(ts_str);

   const std::string&df_str = CommandLineInput->getCmdOption("-ts");
   success = CommandLineInput->setInputTs(df_str);



}

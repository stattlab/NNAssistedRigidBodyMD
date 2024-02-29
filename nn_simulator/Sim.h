#ifndef SIM_H_
#define SIM_H_

#include <cstring>
#include <numeric>
#include <iomanip>      // for std::setprecision
#include <vector>
#include <set>
#include <iostream>
#include <memory>
#include <cmath>
#include <torch/script.h> // One-stop header.

#include "gsd_utils/GSDReader.h"
#include "gsd_utils/GSDWriter.h"
#include "gsd_utils/gsd.h"
#include "gsd_utils/VectorMath.h"
#include "gsd_utils/SnapshotSystemData.h"

#include "mc_sim_utils/Point.h"
#include "mc_sim_utils/Particle.h"
#include "mc_sim_utils/Box.h"
#include "mc_sim_utils/BoxCuboid.h"
#include "mc_sim_utils/NeighborListTree.h"



/**
 * @brief Simulation class
 *
 * This class is the base class for the Sim classes, which execute the actual simulation.
 **/
class Sim {
public:

    /** Constructor */
    Sim()
    {
      std::cout<<"Simulation will be constructed"<<std::endl;
    };

    /** Destructor */
    ~Sim();
    int m_public = 0;

    void buildNNs();
    void setInitConfig(std::shared_ptr< SnapshotSystemData<float>>);
    void setPotentialPairs();
    void refreshNList();
    void dumpGsd();
    void nvt_integrate();
    void integrate_step_one();
    void integrate_step_two();
    void integrate_mid_step();

    // setter/getters
    void setNlistCutoff(double val);
    void setCutoff(double val);
    void setDt(double val);
    void setkT(double val);
    void setModelNumber(int val);
    void setIntegrator();
    void setOutname(std::string& str);


    float get_Kinetic_Trans();
    float get_Kinetic_Rot();


    unsigned int getNTypes() const
    {
      return m_NumParticleTypes;
    }

    std::vector <Particle*> getParticles() const
    {
      return m_Particles;
    }

    unsigned int getMaxN() const
    {
      return m_Particles.size();
    }


protected:
/// VARIABLES ///
  int m_protected = 0;
  int m_ts = 0;
  std::string m_outname;

  std::shared_ptr< SnapshotSystemData<float>> m_init_snap;
  std::vector<linalg::aliases::double_3> m_pos;
  std::vector<linalg::aliases::double_3> m_vel;
  std::vector<linalg::aliases::double_3> m_net_forces;
  std::vector<linalg::aliases::double_3> m_net_torks;
  std::vector<linalg::aliases::double_4> m_quat;
  std::vector<linalg::aliases::double_4> m_ang_mom;
  std::vector<float> m_mass;
  std::vector<linalg::aliases::double_3> m_moi;
  std::vector<linalg::aliases::double_3> m_accel;
  std::vector<unsigned int> m_type;

  std::vector <Particle *> m_Particles;

  unsigned int m_Nparticles;
  float m_Lx;
  Box *m_Box;
  NeighborListTree *m_tree;
  double m_NlistCutoff;
  double m_Cutoff;
  double m_Cutoffsq;

  float m_kT;
  float m_tau = 0.1;
  float m_dt;
  int m_N_dof;
  int m_RN_dof;
  int m_model_number;

  // NVT-Integrator variables
  float m_exp_thermo_fac = 1.00011;
  std::vector<float> m_integrator_vars{-0.0317711,0.108609,0.00814227,-0.014238};
  std::vector<float> m_dxs{0.006,0.01,0.03,0.03,0.01,0.03};

  std::vector<unsigned int> m_pair0;
  std::vector<unsigned int> m_pair1;

  // neural net input/output normalization stuff
  // GLJ-Cube set
  // all input is normalized (scaled to [0,1]) before going to Neural-Net
  // same normalization as training must be applied when we do inference in the
  // simulation code so copy them here
  std::vector<float> min_inputs{0.98,0.0,0.0,0.0,-3.142,0.0};
  std::vector<float> max_inputs{6.31,5.43,3.65,2.04, 3.142,1.572};

  // time data for performance, bottleneck detection etc. 
  float tt1 = 0.0;
  float tt2 = 0.0;
  float tt3 = 0.0;
  float tt4 = 0.0;
  float tt5 = 0.0;
  float tt6 = 0.0;

  float* p_quats=NULL;


  // Constant Variables //
  unsigned int m_NumParticleTypes = 1;


torch::jit::script::Module m_energy;
torch::jit::script::Module m_selector;


};

#endif /* SIM_H_ */

#include "Sim.h"
#include <chrono>
#include <cmath>
#include <math.h>
#include <memory>


using namespace linalg::aliases;
using namespace torch::indexing;
Sim::~Sim()
{
  for (unsigned int i = 0; i < m_Particles.size(); i++)
  {
     delete m_Particles[i];
  }
  m_Particles.clear();
}

void Sim::setDt(double val)
{
  m_dt = val;
}

void Sim::setkT(double val)
{
  m_kT = val;
}

void Sim::setModelNumber(int val)
{
  m_model_number = val;
}



void Sim::setIntegrator()
{

}

void Sim::setOutname(std::string& str)
{
  m_outname = str;

}

void Sim::dumpGsd()
{
  // std::cout<<"Dumping gsd ..."<<std::endl;
  auto writer = std::make_shared<GSDDumpWriter>(m_outname,false,false);
  writer->setWriteMomentum(true);
  writer->setWriteProperty(true);
  writer->setWriteAttribute(true);

  SnapshotSystemData<float> temp_snapshot;
  temp_snapshot.dimensions = m_init_snap->dimensions;
  temp_snapshot.global_box = m_init_snap->global_box;
  temp_snapshot.type_mapping = m_init_snap->type_mapping;

  for (unsigned int i = 0; i < m_pos.size(); i++)
  {
    temp_snapshot.pos.push_back(linalg::aliases::float3(m_pos[i]));
    temp_snapshot.type.push_back(m_type[i]);
    temp_snapshot.quat.push_back(linalg::aliases::float4(m_quat[i]));
    // std::cout<<m_quat[i][0]<<" "<<m_quat[i][1]<<" "<<m_quat[i][2]<<" "<<m_quat[i][3]<<std::endl;
    temp_snapshot.ang_mom.push_back(linalg::aliases::float4(m_ang_mom[i]));
    temp_snapshot.moi.push_back(linalg::aliases::float3(m_moi[i]));
    temp_snapshot.mass.push_back(m_mass[i]);
    temp_snapshot.vel.push_back(linalg::aliases::float3(m_vel[i]));
  }


  writer->analyze(0,temp_snapshot);



}


float Sim::get_Kinetic_Trans()
{
  float en = 0.0;
  for(int i = 0 ; i < m_Nparticles ; i++)
  {
    en = en + (0.5)*m_mass[i]*dot(m_vel[i],m_vel[i]);
  }
  return en;
}

float Sim::get_Kinetic_Rot()
{
  float en = 0.0;
  // soranlara sormayanlara
  double_3 moi = m_moi[0];
  for(int i = 0 ; i < m_Nparticles ; i++)
  {
    double_4 q1 = -m_quat[i];
    q1[0] = -q1[0];
    double_3 q1v = double_3(q1[1],q1[2],q1[3]);
    double_4 q2 = m_ang_mom[i];
    double_3 q2v = double_3(q2[1],q2[2],q2[3]);
    // quaternion multiplication
    // float sca = (q1[0]*q2[0] - dot(q1v,q2v))*0.5; Not needed
    double_3 vect = (cross(q1v,q2v) + q1[0]*q2v + q2[0]*q1v)*0.5;

    en = en + vect[0]*vect[0]/moi[0] + vect[1]*vect[1]/moi[1] + vect[2]*vect[2]/moi[2];
  }

  en = en*0.5;

  return en;
}



void Sim::buildNNs()
{
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    // Energy neural net is always named as 'traced_energyWidthDepth.pt'
    // TODO set Neural net name as an input argument
    module = torch::jit::load("./models/traced_energy606.pt");
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    exit(0);
  }
  // TODO : Uses GPU 0 by default set this as an input argument
  torch::Device device(torch::kCUDA, 0);

  // module.to(at::kCUDA);
  module.eval();
  module.to(device);
  m_energy = module;

  torch::jit::script::Module module2;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    // Selector neural net is always named as 'traced_selector.pt'
    // TODO set Neural net name as an input argument
    module2 = torch::jit::load("./models/traced_selector.pt");
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    exit(0);
  }
  module2.to(at::kCUDA);
  m_selector = module2;
  m_energy.eval();
  m_selector.eval();

}


void Sim::setInitConfig(std::shared_ptr< SnapshotSystemData<float>> init_snap)
{

  m_init_snap = init_snap;

  // here get the positions, quats, velocities, typeids and box size from the snapshot provided
  // ang mom is a quaternion, moi is a 3-vector

  std::vector<linalg::aliases::double_3> pos((m_init_snap->pos).begin(), (m_init_snap->pos).end());
  std::vector<linalg::aliases::double_4> quat((m_init_snap->quat).begin(), (m_init_snap->quat).end());
  std::vector<unsigned int> typeids((m_init_snap->type).begin(), (m_init_snap->type).end());
  std::vector<linalg::aliases::double_3> velocity((m_init_snap->vel).begin(), (m_init_snap->vel).end());
  std::vector<linalg::aliases::double_4> ang_mom((m_init_snap->ang_mom).begin(), (m_init_snap->ang_mom).end());
  std::vector<linalg::aliases::double_3> moi((m_init_snap->moi).begin(), (m_init_snap->moi).end());
  std::vector<float> mass((m_init_snap->mass).begin(), (m_init_snap->mass).end());

  // set the box
  linalg::aliases::double_3 box_dims = m_init_snap->global_box;
  m_Lx = box_dims[0];
  Point SysDim(m_Lx,m_Lx,m_Lx);
  std::cout<<"Setting the box ..."<<std::endl;
  m_Box = new BoxCuboid(SysDim);

  // set the neighbor list
  std::cout<<"Setting the tree ..."<<std::endl;
  m_tree = new NeighborListTree(this,m_Box);
  std::cout<<"Tree is set ..."<<std::endl;


  for (unsigned int i = 0; i < quat.size(); i++)
  {
      // std::cout<<i<<" "<<velocity[i][0]<<" "<<velocity[i][1]<<" "<<velocity[i][2]<<std::endl;
      if(typeids[i]==0)
      {
        m_pos.push_back(pos[i]);
        m_quat.push_back(quat[i]);
        m_vel.push_back(velocity[i]);
        m_ang_mom.push_back(ang_mom[i]);
        m_mass.push_back(mass[i]);
        m_moi.push_back(moi[i]);
      }
  }

  m_Nparticles = m_pos.size();
  m_type.resize(m_Nparticles);
  std::fill(m_type.begin(),m_type.end(),0);

  std::cout<<m_Nparticles<<" particles will be simulated."<<std::endl;

  // Here set the m_Particles -> This is needed at least for NeighborList to work
  // m_pos that is declared above might be useless
  std::cout<<"Setting the particles ..."<<std::endl;
  m_Particles.resize(m_Nparticles);
  for (unsigned int i = 0; i < m_Nparticles; i++)
  {
    // const Point P = MonteCarloUtils::genRandomCoordCuboid(From, To);
    Point P(m_pos[i][0],m_pos[i][1],m_pos[i][2]);
    m_Particles[i] = new Particle(P, m_Box, 0);
    // m_Particles[all]->setOrigin(P_new);
  }

  // Accelerations are initialized as zeros
  m_accel.resize(m_Nparticles);
  std::fill(m_accel.begin(), m_accel.end(), double_3(0.0,0.0,0.0));

  m_N_dof = (m_Nparticles - 1)*3;
  m_RN_dof = (m_Nparticles)*3;

}

void Sim::setNlistCutoff(double in_val)
{
  std::vector<std::vector <double>> val;
  val.push_back({in_val});
  m_tree->setParticleCutoffs(val);

  m_NlistCutoff = in_val;
}

void Sim::setCutoff(double in_val)
{
  m_Cutoff = in_val;
  m_Cutoffsq = in_val*in_val;
}

void Sim::refreshNList()
{
  for (unsigned int i = 0; i < m_Nparticles; i++)
  {
    Point P(m_pos[i][0],m_pos[i][1],m_pos[i][2]);
    m_Particles[i]->setOrigin(P);
  }

  m_tree->slotMaxNumChanged();
  m_tree->slotRemapParticles();
  m_tree->slotBoxChanged();
  m_tree->buildTrees();

  m_pair0.clear();
  m_pair1.clear();

  for (unsigned int i = 0; i <  m_Particles.size(); i++)
  {
    std::vector<unsigned int > neigh = m_tree->findNeighborsParticle(i,m_NlistCutoff);
    for(unsigned int j=0; j < neigh.size(); j++)
    {
      m_pair0.push_back(i);
      m_pair1.push_back(neigh[j]);
    }
  }


}


void Sim::setPotentialPairs()
{
  std::cout<<"Setting pairs from Nlist"<<std::endl;
  // this updates the tree neighbor list
  // also updates the pair0 and pair1 accordingly
  m_tree->slotMaxNumChanged();
  m_tree->slotRemapParticles();
  m_tree->slotBoxChanged();
  m_tree->buildTrees();

  m_pair0.clear();
  m_pair1.clear();

  for (unsigned int i = 0; i <  m_Particles.size(); i++)
  {
    std::vector<unsigned int > neigh = m_tree->findNeighborsParticle(i,m_NlistCutoff);
    for(unsigned int j=0; j < neigh.size(); j++)
    {
      m_pair0.push_back(i);
      m_pair1.push_back(neigh[j]);
    }
  }

}
std::vector<float> linearize(const std::vector<std::vector<float>>& vec_vec) {
    std::vector<float> vec;
    for (const auto& v : vec_vec) {
        for (auto d : v) {
            vec.push_back(d);
        }
    }
    return vec;
}
std::vector<float> linearize3(const std::vector<double_3>& vec_vec) {
    std::vector<float> vec;
    for (const auto& v : vec_vec) {
        for (auto d : v) {
            vec.push_back(d);
        }
    }
    return vec;
}
std::vector<float> linearize4(const std::vector<double_4>& vec_vec) {
    std::vector<float> vec;
    for (const auto& v : vec_vec) {
        for (auto d : v) {
            vec.push_back(d);
        }
    }
    return vec;
}
void Sim::nvt_integrate()
{
  if(m_ts==0)
  {
    integrate_mid_step();
    integrate_step_two();
  }
  else
  {
    auto start_one = std::chrono::high_resolution_clock::now();
    integrate_step_one();
    auto end_one = std::chrono::high_resolution_clock::now();
    auto start_mid = std::chrono::high_resolution_clock::now();
    integrate_mid_step();
    auto end_mid = std::chrono::high_resolution_clock::now();
    auto start_two = std::chrono::high_resolution_clock::now();
    integrate_step_two();
    auto end_two = std::chrono::high_resolution_clock::now();
    // float t_mid = std::chrono::duration<float, std::milli>(end_mid-start_mid).count();
    // float t_one = std::chrono::duration<float, std::milli>(end_one-start_one).count();
    // float t_two = std::chrono::duration<float, std::milli>(end_two-start_two).count();
    // tt1 += t_one;
    // tt2 += t_mid;
    // tt3 += t_two;
    //
    // if(m_ts==1000)
    // {
    //   std::cout<<"Timestep one: "<<tt1<<std::endl;
    //   std::cout<<"Timestep mid: "<<tt2<<std::endl;
    //   std::cout<<"Timestep end: "<<tt3<<std::endl;
    // }
  }

}

void Sim::integrate_step_one()
{
  // INTEGRATE STEP 1 - TRANSLATION //
  for(int i = 0; i < m_Nparticles; i++)
  {
    m_vel[i] = m_vel[i] + (0.5)*m_accel[i]*m_dt;
    m_vel[i] = m_vel[i]*m_exp_thermo_fac;
    double_3 np = m_pos[i] + m_vel[i]*m_dt;
    Point p(np[0],np[1],np[2]);
    m_Box->usePBConditions(&p);
    m_pos[i] = double_3(p.x[0],p.x[1],p.x[2]);
  }


  // INTEGRATE STEP 2 - ROTATION //
  float xi_rot = m_integrator_vars[2];
  float exp_fac = exp((-m_dt/2.0)*xi_rot);

  double_3 moi = m_moi[0];
  // std::cout<<"880"<<std::endl;

  for(int i = 0; i < m_Nparticles; i++)
  {
    // get_dp //
    double_4 q = m_quat[i];
    double_3 qv = double_3(q[1],q[2],q[3]);
    double_3 v = m_net_torks[i];
    // std::cout<<i<<" nettork "<<v[0]<<" "<<v[1]<<" "<<v[2]<<std::endl;
    double_4 q_res;
    q_res[0] = -dot(qv,v);
    double_3 res_v = cross(qv,v) + q[0]*v;
    q_res[1] = res_v[0];
    q_res[2] = res_v[1];
    q_res[3] = res_v[2];
    q_res = q_res * m_dt;
    // get_dp //

    m_ang_mom[i] = m_ang_mom[i] + q_res;
    m_ang_mom[i] = m_ang_mom[i] * exp_fac;

    // permutation 1 //
    double_4 p = m_ang_mom[i];

    double_4 p3; // can be written shorter
    p3[0] = -p[3];
    p3[1] = p[2];
    p3[2] = -p[1];
    p3[3] = p[0];

    double_4 q3; // can be written shorter
    q3[0] = -q[3];
    q3[1] = q[2];
    q3[2] = -q[1];
    q3[3] = q[0];

    float cphi3 = cos( ((1.0/4.0)/moi[2])*dot(p,q3)*0.5*m_dt );
    float sphi3 = sin( ((1.0/4.0)/moi[2])*dot(p,q3)*0.5*m_dt );

    p = cphi3*p + sphi3*p3;
    q = cphi3*q + sphi3*q3;
    // permutation 1 //
    // permutation 2 //
    double_4 p2 = double_4(-p[2],-p[3],p[0],p[1]);
    double_4 q2 = double_4(-q[2],-q[3],q[0],q[1]);

    float cphi2 = cos( ((1.0/4.0)/moi[1])*dot(p,q2)*0.5*m_dt );
    float sphi2 = sin( ((1.0/4.0)/moi[1])*dot(p,q2)*0.5*m_dt );

    p = cphi2*p + sphi2*p2;
    q = cphi2*q + sphi2*q2;
    // permutation 2 //


    // permutation 3 //
    double_4 p1 = double_4(-p[1],p[0],p[3],-p[2]);
    double_4 q1 = double_4(-q[1],q[0],q[3],-q[2]);

    float cphi1 = cos( ((1.0/4.0)/moi[0])*dot(p,q1)*m_dt );
    float sphi1 = sin( ((1.0/4.0)/moi[0])*dot(p,q1)*m_dt );

    p = cphi1*p + sphi1*p1;
    q = cphi1*q + sphi1*q1;
    // permutation 3 //

    // std::cout<<i<<" "<<p[0]<<" "<<p[1]<<" "<<p[2]<<" "<<p[3]<<std::endl;

    // permutation 2 //
    double_4 p22 = double_4(-p[2],-p[3],p[0],p[1]);
    double_4 q22 = double_4(-q[2],-q[3],q[0],q[1]);

    float cphi22 = cos( ((1.0/4.0)/moi[1])*dot(p,q22)*0.5*m_dt );
    float sphi22 = sin( ((1.0/4.0)/moi[1])*dot(p,q22)*0.5*m_dt );

    p = cphi22*p + sphi22*p22;
    q = cphi22*q + sphi22*q22;
    // permutation 2 //

    // permutation 1 //
    double_4 p32; // can be written shorter
    p32[0] = -p[3];
    p32[1] = p[2];
    p32[2] = -p[1];
    p32[3] = p[0];

    double_4 q32; // can be written shorter
    q32[0] = -q[3];
    q32[1] = q[2];
    q32[2] = -q[1];
    q32[3] = q[0];

    float cphi32 = cos( ((1.0/4.0)/moi[2])*dot(p,q32)*0.5*m_dt );
    float sphi32 = sin( ((1.0/4.0)/moi[2])*dot(p,q32)*0.5*m_dt );

    p = cphi32*p + sphi32*p32;
    q = cphi32*q + sphi32*q32;
    // permutation 1 //
    // renormalize orientations



    double_4 q_n = normalize(q);
    m_quat[i] = q_n;
    m_ang_mom[i] = p;
  }
  // ADVANCE THERMOSTAT - TRANSLATION //

  float trans_kin_en = get_Kinetic_Trans();
  float trans_kT = (2.0/m_N_dof)*trans_kin_en;
  float xi_prime = m_integrator_vars[0] + (0.5)*((m_dt*m_tau)/m_tau)*(trans_kT/m_kT - 1.0);
  m_integrator_vars[0] = xi_prime + (0.5)*((m_dt*m_tau)/m_tau)*(trans_kT/m_kT - 1.0);
  m_integrator_vars[1] = m_integrator_vars[1] + xi_prime*m_dt;
  m_exp_thermo_fac = exp(-0.5*m_integrator_vars[0]*m_dt);

  // ADVANCE THERMOSTAT - ROTATION //

  float xi_rot2 = m_integrator_vars[2];
  float eta_rot = m_integrator_vars[3];
  float rot_kin_en = get_Kinetic_Rot();

  float xi_prime_rot = xi_rot2 + (0.5)*((m_dt/m_tau)/m_tau)*( ((2.0*rot_kin_en)/m_RN_dof)/m_kT - 1.0);
  xi_rot2 =  xi_prime_rot + (0.5)*((m_dt/m_tau)/m_tau)*( ((2.0*rot_kin_en)/m_RN_dof)/m_kT - 1.0);

  // std::cout<<"1001"<<std::endl;
  eta_rot = eta_rot + xi_prime_rot*m_dt;
  m_integrator_vars[2] = xi_rot2;
  m_integrator_vars[3] = eta_rot;
}

void Sim::integrate_step_two()
{
  // INTEGRATE STEP 2 - TRANS   &&& INTEGRATE STEP 2 - ROT  - SAME LOOP//
  float mass = m_mass[0];
  float exp_fac2 = exp((-m_dt/2.0)*m_integrator_vars[2]);
  for(int i = 0; i < m_Nparticles; i++)
  {
    m_accel[i] = m_net_forces[i]/mass;
    m_vel[i] = m_vel[i]*m_exp_thermo_fac;
    m_vel[i] = m_vel[i] + (0.5)*m_dt*m_accel[i];

    m_ang_mom[i] = m_ang_mom[i]*exp_fac2;
    // get_dp //
    double_4 q = m_quat[i];
    double_3 qv = double_3(q[1],q[2],q[3]);
    double_3 v = m_net_torks[i];
    double_4 q_res;
    q_res[0] = -dot(qv,v);
    double_3 res_v = cross(qv,v) + q[0]*v;
    q_res[1] = res_v[0];
    q_res[2] = res_v[1];
    q_res[3] = res_v[2];
    q_res = q_res * m_dt;
    // get_dp //
    m_ang_mom[i] = m_ang_mom[i] + q_res;

  }


  m_ts = m_ts + 1;

}

void Sim::integrate_mid_step()
{

  auto s1 = std::chrono::high_resolution_clock::now();
  int N_potpair = m_pair0.size();

  // std::cout<<"# potential pairs: "<<N_potpair<<std::endl;

  std::vector<unsigned int> pair0;
  std::vector<unsigned int> pair1;
  std::vector<Point> delta;

  // calculate distances between potential pairs and eliminate those with larger cutoff
  for (unsigned int i = 0; i < N_potpair; i++)
  {
    // Point dist_vect = (m_Particles[m_pair0[i]]->getOrigin() - m_Particles[m_pair1[i]]->getOrigin());
    double_3 disto = m_pos[m_pair0[i]]-m_pos[m_pair1[i]] ;
    Point dist_vect(disto[0],disto[1],disto[2]);
    m_Box->usePBConditions(&dist_vect);
    double dist_sq = dist_vect.getSquare();
    if(dist_sq<m_Cutoffsq)
    {
      pair0.push_back(m_pair0[i]);
      pair1.push_back(m_pair1[i]);
      delta.push_back(dist_vect);
    }
  }
  int N_pair = pair0.size();
  // std::cout<<"# pairs: "<<N_pair<<std::endl;
  // std::cout<<"# of particles "<<m_Nparticles<<std::endl;
  auto t1 = std::chrono::high_resolution_clock::now();

  auto s2 = std::chrono::high_resolution_clock::now();
  // DO THE GEOMETRIC OPERATIONS BY HAND USING VECTORS of double_3 and double_4

  std::vector<double_3> true_faces {
          double_3(0.0, 0.0, 0.0),
          double_3(1.0, 0.0, 0.0),
          double_3(-1.0, 0.0, 0.0),
          double_3(0.0, 1.0, 0.0),
          double_3(0.0, -1.0, 0.0),
          double_3(0.0, 0.0, 1.0),
          double_3(0.0, 0.0, -1.0)
      };

  // FIND INTERACTING FACE OF THE CUBE 1 - LIBTORCH GPU //
  std::vector<float> true_faces1_vec2 = linearize3(true_faces);
  std::vector<float> m_quats_lin = linearize4(m_quat);
  std::vector<float> m_pos_lin = linearize3(m_pos);
  torch::Tensor true_faces_tensor = torch::from_blob(true_faces1_vec2.data(), {7,3}, torch::kFloat32).to(at::kCUDA);;
  torch::Tensor m_quats_tensor = torch::from_blob(m_quats_lin.data(), {m_Nparticles,4}, torch::kFloat32).to(at::kCUDA);
  torch::Tensor m_pos_tensor = torch::from_blob(m_pos_lin.data(), {m_Nparticles,3}, torch::kFloat32).to(at::kCUDA);
  torch::Tensor pair0_tensor = torch::from_blob(pair0.data(), {pair0.size()},torch::kInt32).to(at::kCUDA);
  torch::Tensor pair1_tensor = torch::from_blob(pair1.data(), {pair1.size()},torch::kInt32).to(at::kCUDA);


  torch::Tensor q1_faces_tensor = torch::index_select(m_quats_tensor, 0, pair0_tensor);
  torch::Tensor q2_faces_tensor = torch::index_select(m_quats_tensor, 0, pair1_tensor);
  torch::Tensor com1_tensor = torch::index_select(m_pos_tensor, 0, pair0_tensor);
  torch::Tensor com2_tensor = torch::index_select(m_pos_tensor, 0, pair1_tensor);

  torch::Tensor true_faces_tiled = true_faces_tensor.tile({N_pair,1});
  torch::Tensor q1_faces_tiled = q1_faces_tensor.repeat_interleave(7, 0);
  torch::Tensor q2_faces_tiled = q2_faces_tensor.repeat_interleave(7, 0);
  torch::Tensor com1_faces_tiled = com1_tensor.repeat_interleave(7, 0);
  torch::Tensor com2_tensor_repeated = com2_tensor.repeat_interleave(7, 0);

  torch::Tensor coef1 = q1_faces_tiled.index({Slice(),Slice(None,1)}) * q1_faces_tiled.index({Slice(),Slice(None,1)}) - torch::sum(q1_faces_tiled.index({Slice(),Slice(1,None)})*q1_faces_tiled.index({Slice(),Slice(1,None)}),1    ).view({-1, 1});
  torch::Tensor term1 = coef1 * true_faces_tiled;
  torch::Tensor term2 = 2.0 * q1_faces_tiled.index({Slice(),Slice(None,1)}) * torch::cross(q1_faces_tiled.index({Slice(),Slice(1,None)}), true_faces_tiled);
  torch::Tensor term3 = 2.0 * torch::sum(q1_faces_tiled.index({Slice(),Slice(1,None)}) * true_faces_tiled, 1).view({-1, 1}) *q1_faces_tiled.index({Slice(),Slice(1,None)});
  torch::Tensor faces1_tensor = term1 + term2 + term3;
  torch::Tensor faces1_abs = faces1_tensor + com1_faces_tiled;
  torch::Tensor to_pbc = com2_tensor_repeated - faces1_abs;

  torch::Tensor delta2 = torch::where(to_pbc > 0.5 * m_Lx, to_pbc - m_Lx, to_pbc); //TODO fix box length
  torch::Tensor com2_faces1_rel = torch::where(delta2 < -0.5 * m_Lx, m_Lx + delta2, delta2);
  torch::Tensor dist2faces1 = torch::linalg_norm(com2_faces1_rel,None,1).view({-1,7});
  torch::Tensor face1_index_tensor = torch::argmin(dist2faces1, 1);
  ////////// FIND INTERACTING FACE 2 - LIBTORCH ///////////////
  torch::Tensor coef1_2 = q2_faces_tiled.index({Slice(),Slice(None,1)}) * q2_faces_tiled.index({Slice(),Slice(None,1)}) - torch::sum(q2_faces_tiled.index({Slice(),Slice(1,None)})*q2_faces_tiled.index({Slice(),Slice(1,None)}),1    ).view({-1, 1});
  torch::Tensor term1_2 = coef1_2 * true_faces_tiled;

  torch::Tensor term2_2 = 2.0 * q2_faces_tiled.index({Slice(),Slice(None,1)}) * torch::cross(q2_faces_tiled.index({Slice(),Slice(1,None)}), true_faces_tiled);
  torch::Tensor term3_2 = 2.0 * torch::sum(q2_faces_tiled.index({Slice(),Slice(1,None)}) * true_faces_tiled, 1).view({-1, 1}) *q2_faces_tiled.index({Slice(),Slice(1,None)});

  torch::Tensor faces2_tensor = term1_2 + term2_2 + term3_2;
  torch::Tensor faces2_abs = faces2_tensor + com2_tensor_repeated;
  torch::Tensor to_pbc2 = faces2_abs - com1_faces_tiled;
  torch::Tensor delta3 = torch::where(to_pbc2 > 0.5 * m_Lx, to_pbc2 - m_Lx, to_pbc2);
  torch::Tensor com1_faces2_rel_tensor = torch::where(delta3 < -0.5 * m_Lx, m_Lx + delta3, delta3);
  torch::Tensor dist2faces2 = torch::linalg_norm(com1_faces2_rel_tensor,None,1).view({-1,7});
  torch::Tensor face2_index_tensor = torch::argmin(dist2faces2, 1);
  ////////// FIND INTERACTING FACE 2 - LIBTORCH ///////////////



    ////////////////////// ROTATE EVERYTHING SUCH THAT INTERACTING FACE 1 IS [1.0,0.0,0.0] - LIBTORCH //////////////////////
    std::vector<float> true_right {1.0,0.0,0.0};
    torch::Tensor right_true_tensor = torch::from_blob(true_right.data(), 3, torch::kFloat32).to(at::kCUDA);
    torch::Tensor right_true_tiled = right_true_tensor.tile({N_pair,1});
    torch::Tensor faces1i_tensor = faces1_tensor.view({-1,7,3});
    torch::Tensor indices0 = torch::arange(0, N_pair).to(at::kCUDA);
    torch::Tensor faces1_inter_tensor = faces1i_tensor.index({indices0,face1_index_tensor});

    ///quat_from_two_vectors(faces1_inter,right_true)///
    torch::Tensor vect0 = torch::cross(faces1_inter_tensor,right_true_tiled);
    torch::Tensor scalar0 = 1.0 + torch::sum(faces1_inter_tensor*right_true_tiled,1);
    torch::Tensor quat0 = torch::zeros({N_pair,4},torch::dtype(torch::kFloat32)).to(at::kCUDA);
    quat0.index_put_({Slice(),Slice(None,1)},scalar0.view({-1,1}));// = scalar0;
    quat0.index_put_({Slice(),Slice(1,None)},vect0);// = vect0;
    ///renormalize_quat()//
    torch::Tensor q0_norm = torch::sqrt(torch::sum(quat0*quat0,1));
    torch::Tensor q_rot1u_tensor = quat0/q0_norm.view({-1,1});

    torch::Tensor q_rot1_tensor = q_rot1u_tensor.repeat_interleave(7, 0);
    /// faces1_r1 = rotate(q_rot1,faces1) ////
    torch::Tensor coef1_3 = q_rot1_tensor.index({Slice(),Slice(None,1)}) * q_rot1_tensor.index({Slice(),Slice(None,1)}) - torch::sum(q_rot1_tensor.index({Slice(),Slice(1,None)})*q_rot1_tensor.index({Slice(),Slice(1,None)}),1    ).view({-1, 1});
    torch::Tensor term1_3 = coef1_3 * faces1_tensor;

    torch::Tensor term2_3 = 2.0 * q_rot1_tensor.index({Slice(),Slice(None,1)}) * torch::cross(q_rot1_tensor.index({Slice(),Slice(1,None)}), faces1_tensor);
    torch::Tensor term3_3 = 2.0 * torch::sum(q_rot1_tensor.index({Slice(),Slice(1,None)}) * faces1_tensor, 1).view({-1, 1}) *q_rot1_tensor.index({Slice(),Slice(1,None)});

    torch::Tensor faces1_r1_tensor = term1_3 + term2_3 + term3_3;


    //// faces2_r1 = rotate(q_rot1,com1_faces2_rel) ////
    torch::Tensor coef1_4 = q_rot1_tensor.index({Slice(),Slice(None,1)}) * q_rot1_tensor.index({Slice(),Slice(None,1)}) - torch::sum(q_rot1_tensor.index({Slice(),Slice(1,None)})*q_rot1_tensor.index({Slice(),Slice(1,None)}),1    ).view({-1, 1});
    torch::Tensor term1_4 = coef1_4 * com1_faces2_rel_tensor;

    torch::Tensor term2_4 = 2.0 * q_rot1_tensor.index({Slice(),Slice(None,1)}) * torch::cross(q_rot1_tensor.index({Slice(),Slice(1,None)}), com1_faces2_rel_tensor);
    torch::Tensor term3_4 = 2.0 * torch::sum(q_rot1_tensor.index({Slice(),Slice(1,None)}) * com1_faces2_rel_tensor, 1).view({-1, 1}) *q_rot1_tensor.index({Slice(),Slice(1,None)});

    torch::Tensor faces2_r1_tensor = term1_4 + term2_4 + term3_4;
    //face1p_index = (face1_index + 2)%7//
    torch::Tensor face1p_index_tensor = torch::remainder(face1_index_tensor+2,7);

    //face1p_index[face1p_index==0] = 1//
    face1p_index_tensor.masked_fill_(face1p_index_tensor.eq(0), 1);

    //forward_true = np.array([0.0,1.0,0.0])//
    std::vector<float> true_forward {0.0,1.0,0.0};
    torch::Tensor forward_true_tensor = torch::from_blob(true_forward.data(), 3, torch::kFloat32).to(at::kCUDA);

    //forward_true = np.tile(forward_true,(self.N_pair,1))//
    torch::Tensor forward_true_tiled = forward_true_tensor.tile({N_pair,1});

    /*
    faces1p_inter = np.zeros((self.N_pair,3))
    faces1ri = faces1_r1.reshape(-1,7,3)

    for i in range(self.N_pair):
        faces1p_inter[i] = np.copy(faces1ri[i,face1p_index[i],:])
    */
    torch::Tensor faces1ri_tensor = faces1_r1_tensor.view({-1,7,3});
    torch::Tensor faces1p_inter_tensor = faces1ri_tensor.index({indices0,face1p_index_tensor});
    //q_rot2u = quat_from_two_vectors(faces1p_inter,forward_true)//
    torch::Tensor vect1 = torch::cross(faces1p_inter_tensor,forward_true_tiled);
    torch::Tensor scalar1 = 1.0 + torch::sum(faces1p_inter_tensor*forward_true_tiled,1);
    torch::Tensor quat1 = torch::zeros({N_pair,4},torch::dtype(torch::kFloat32)).to(at::kCUDA);
    quat1.index_put_({Slice(),Slice(None,1)},scalar1.view({-1,1}));// = scalar1;
    quat1.index_put_({Slice(),Slice(1,None)},vect1);// = vect1;
    ///renormalize_quat()//
    torch::Tensor q1_norm = torch::sqrt(torch::sum(quat1*quat1,1));
    torch::Tensor q_rot2u_tensor = quat1/q1_norm.view({-1,1});

    //q_rot2 = np.repeat(q_rot2u,7,axis=0)//
    torch::Tensor q_rot2_tensor = q_rot2u_tensor.repeat_interleave(7, 0);

    //faces1_r2 = rotate(q_rot2,faces1_r1)//
    torch::Tensor coef1_6 = q_rot2_tensor.index({Slice(),Slice(None,1)}) * q_rot2_tensor.index({Slice(),Slice(None,1)}) - torch::sum(q_rot2_tensor.index({Slice(),Slice(1,None)})*q_rot2_tensor.index({Slice(),Slice(1,None)}),1    ).view({-1, 1});
    torch::Tensor term1_6 = coef1_6 * faces1_r1_tensor;

    torch::Tensor term2_6 = 2.0 * q_rot2_tensor.index({Slice(),Slice(None,1)}) * torch::cross(q_rot2_tensor.index({Slice(),Slice(1,None)}), faces1_r1_tensor);
    torch::Tensor term3_6 = 2.0 * torch::sum(q_rot2_tensor.index({Slice(),Slice(1,None)}) * faces1_r1_tensor, 1).view({-1, 1}) *q_rot2_tensor.index({Slice(),Slice(1,None)});

    torch::Tensor faces1_r2_tensor = term1_6 + term2_6 + term3_6;

    //faces2_r2 = rotate(q_rot2,faces2_r1)//
    torch::Tensor coef1_5 = q_rot2_tensor.index({Slice(),Slice(None,1)}) * q_rot2_tensor.index({Slice(),Slice(None,1)}) - torch::sum(q_rot2_tensor.index({Slice(),Slice(1,None)})*q_rot2_tensor.index({Slice(),Slice(1,None)}),1    ).view({-1, 1});
    torch::Tensor term1_5 = coef1_5 * faces2_r1_tensor;
    torch::Tensor term2_5 = 2.0 * q_rot2_tensor.index({Slice(),Slice(None,1)}) * torch::cross(q_rot2_tensor.index({Slice(),Slice(1,None)}), faces2_r1_tensor);
    torch::Tensor term3_5 = 2.0 * torch::sum(q_rot2_tensor.index({Slice(),Slice(1,None)}) * faces2_r1_tensor, 1).view({-1, 1}) *q_rot2_tensor.index({Slice(),Slice(1,None)});

    torch::Tensor faces2_r2_tensor = term1_5 + term2_5 + term3_5;

    // std::cout<<faces2_r2_tensor<<std::endl; //CORRECT

    //faces2ri = faces2_r2.reshape(-1,7,3)//
    torch::Tensor faces2ri_tensor = faces2_r2_tensor.view({-1,7,3});
    //faces2_r2_com = faces2ri[:,0,:]//
    torch::Tensor faces2_r2_com_tensor = faces2ri_tensor.index({Slice(),Slice(0,1),Slice()}).view({N_pair,3});

    // multiplier = np.ones_like(faces2_r2) //
    torch::Tensor multiplier_tensor = torch::ones_like(faces2_r2_tensor);

    // y_signs = np.sign(faces2_r2_com[:,1]) //
    torch::Tensor y_signs_tensor = torch::sign(faces2_r2_com_tensor.index({Slice(),Slice(1,2)}));

    // z_signs = np.sign(faces2_r2_com[:,2]) //
    torch::Tensor z_signs_tensor = torch::sign(faces2_r2_com_tensor.index({Slice(),Slice(2,None)}));

    // multiplier_force = np.ones((self.N_pair,3)) //
    torch::Tensor multiplier_force_tensor = torch::ones({N_pair,3},torch::dtype(torch::kFloat32)).to(at::kCUDA);

    // multiplier_force[:,1] = y_signs //
    multiplier_force_tensor.index_put_({Slice(),Slice(1,2)},y_signs_tensor);

    // multiplier_force[:,2] = z_signs //
    multiplier_force_tensor.index_put_({Slice(),Slice(2,None)},z_signs_tensor);
    torch::Tensor multiplier_torque_tensor = torch::ones({N_pair,3},torch::dtype(torch::kFloat32)).to(at::kCUDA);
    multiplier_torque_tensor.index_put_({Slice(),Slice(0,1)},y_signs_tensor*z_signs_tensor);
    multiplier_torque_tensor.index_put_({Slice(),Slice(1,2)},z_signs_tensor);
    multiplier_torque_tensor.index_put_({Slice(),Slice(2,None)},y_signs_tensor);

    // y_signs = np.repeat(y_signs,7) //
    // z_signs = np.repeat(z_signs,7) //
    y_signs_tensor = y_signs_tensor.repeat_interleave(7);
    z_signs_tensor = z_signs_tensor.repeat_interleave(7);

    // multiplier[:,1] = y_signs //
    // multiplier[:,2] = z_signs //
    multiplier_tensor.index_put_({Slice(),Slice(1,2)},y_signs_tensor.view({-1,1}));
    multiplier_tensor.index_put_({Slice(),Slice(2,None)},z_signs_tensor.view({-1,1}));

    faces2_r2_tensor = faces2_r2_tensor*multiplier_tensor;

    //faces2ri = faces2_r2.reshape(-1,7,3)//
    faces2ri_tensor = faces2_r2_tensor.view({-1,7,3});
    //faces2_r2_com = faces2ri[:,0,:]//
    faces2_r2_com_tensor = faces2ri_tensor.index({Slice(),Slice(0,1),Slice()}).view({N_pair,3});


    //switch_index = np.where(faces2_r2_com[:,2]>faces2_r2_com[:,1])//
    //switch_index = switch_index[0]//


    torch::Tensor will_switch_tensor = torch::where(faces2_r2_com_tensor.index({Slice(),Slice(2,None)}).view({-1})>faces2_r2_com_tensor.index({Slice(),Slice(1,2)}).view({-1}),torch::ones({N_pair},torch::dtype(torch::kInt32)).to(at::kCUDA),torch::zeros({N_pair},torch::dtype(torch::kInt32)).to(at::kCUDA));
    torch::Tensor will_switch_booltensor = will_switch_tensor.to(torch::kBool);


    // faces2ri[mask, :, [1, 2]] = faces2ri[mask, :, [2, 1]] // this is not in the python code make sure it works as intended
    /*
    for i in range(self.N_pair):
        if(i in switch_index):
            faces2ri[i,:,[1,2]] = faces2ri[i,:,[2,1]]
    */

    // std::cout<<faces2ri_tensor<<std::endl;
    // exit(0);
    faces2ri_tensor.index_put_({will_switch_booltensor,Slice(),Slice(1,3)},faces2ri_tensor.index({will_switch_booltensor,Slice(),Slice()}).index({Slice(),Slice(),torch::tensor({2,1})})); // TODO : fix this

    //faces2_r2 = faces2ri.reshape(-1,3)//
    faces2_r2_tensor = faces2ri_tensor.view({-1,3});

    /*
    faces2_inter = np.zeros((self.N_pair,3))
    for i in range(self.N_pair):
        faces2_inter[i] = np.copy(faces2ri[i,face2_index[i],:])
    */
    faces1p_inter_tensor = faces2ri_tensor.index({indices0,face2_index_tensor});

    //faces2_r2_com = faces2ri[:,0,:]//
    faces2_r2_com_tensor = faces2ri_tensor.index({Slice(),Slice(0,1),Slice()}).view({N_pair,3});

    // std::cout<<faces2_r2_com_tensor<<std::endl;
    // exit(0);


    //faces2_r2_com7 = np.repeat(faces2_r2_com,7,axis=0)//
    //faces2_r2_relcom2 = faces2_r2 - faces2_r2_com7//
    torch::Tensor faces2_r2_com7_tensor = faces2_r2_com_tensor.repeat_interleave(7, 0);
    torch::Tensor faces2_r2_relcom2_tensor = faces2_r2_tensor - faces2_r2_com7_tensor;

    /*
    faces2_r2_relcom2_3d = faces2_r2_relcom2.reshape(-1,7,3)
    for i in range(self.N_pair):
      faces2_inter_relcom2[i] = np.copy(faces2_r2_relcom2_3d[i,face2_index[i],:])
    */
    torch::Tensor faces2_r2_relcom2_3d_tensor = faces2_r2_relcom2_tensor.view({-1,7,3});
    torch::Tensor faces2_inter_relcom2_tensor = faces2_r2_relcom2_3d_tensor.index({indices0,face2_index_tensor});

    //faces2_inter_relcom2[faces2_inter_relcom2[:,0]<-1.0,0] = -1.0//
    faces2_inter_relcom2_tensor.index_put_({faces2_inter_relcom2_tensor.index({Slice(),Slice(0,1)}).view({-1})<-1.0,Slice(0,1)},-1.0);

    //faces2_inter_relcom2[faces2_inter_relcom2[:,0]>1.0,0] = 1.0//
    faces2_inter_relcom2_tensor.index_put_({faces2_inter_relcom2_tensor.index({Slice(),Slice(0,1)}).view({-1})>1.0,Slice(0,1)},1.0);

    torch::Tensor reduced_configs = torch::zeros({N_pair,6},torch::dtype(torch::kFloat32)).to(at::kCUDA);
    // xcos_angle2 = np.arccos(-faces2_inter_relcom2[:,0]) //
    reduced_configs.index_put_({Slice(),Slice(3,4)},torch::arccos(-faces2_inter_relcom2_tensor.index({Slice(),Slice(0,1)})));

    // std::cout<<faces2_inter_relcom2_tensor.index({12707,0})<<std::endl;
    // exit(0);

    // yztan_angle2 = np.arctan2(-faces2_inter_relcom2[:,2],-faces2_inter_relcom2[:,1]) //
    reduced_configs.index_put_({Slice(),Slice(4,5)},torch::arctan2(-faces2_inter_relcom2_tensor.index({Slice(),Slice(2,None)}),-faces2_inter_relcom2_tensor.index({Slice(),Slice(1,2)}) ));

    // reduced_configs[:,:3] = faces2_r2_com //
    reduced_configs.index_put_({Slice(),Slice(0,3)},faces2_r2_com_tensor);
    // true_left = np.array([-1.0,0.0,0.0]) //
    // true_left = np.tile(true_left,(len(faces2_inter_relcom2),1)) //
    std::vector<float> left_true {-1.0,0.0,0.0};
    torch::Tensor left_true_tensor = torch::from_blob(left_true.data(), 3, torch::kFloat32).to(at::kCUDA);
    torch::Tensor left_true_tiled = left_true_tensor.tile({N_pair,1});

    //q21 = quat_from_two_vectors(faces2_inter_relcom2,true_left)//
    torch::Tensor vect2 = torch::cross(faces2_inter_relcom2_tensor,left_true_tiled);
    torch::Tensor scalar2 = 1.0 + torch::sum(faces2_inter_relcom2_tensor*left_true_tiled,1);
    torch::Tensor quat2 = torch::zeros({N_pair,4},torch::dtype(torch::kFloat32)).to(at::kCUDA);
    quat2.index_put_({Slice(),Slice(None,1)},scalar2.view({-1,1}));// = scalar1;
    quat2.index_put_({Slice(),Slice(1,None)},vect2);// = vect1;
    ///renormalize_quat()//
    torch::Tensor q2_norm = torch::sqrt(torch::sum(quat2*quat2,1));
    torch::Tensor q21_tensor = quat2/q2_norm.view({-1,1});

    // q21 = np.repeat(q21,7,axis=0) //
    q21_tensor = q21_tensor.repeat_interleave(7, 0);

    // faces2_r2_r21_relcom2 = rotate(q21,faces2_r2_relcom2) //
    torch::Tensor coef1_7 = q21_tensor.index({Slice(),Slice(None,1)}) * q21_tensor.index({Slice(),Slice(None,1)}) - torch::sum(q21_tensor.index({Slice(),Slice(1,None)})*q21_tensor.index({Slice(),Slice(1,None)}),1).view({-1, 1});
    torch::Tensor term1_7 = coef1_7 * faces2_r2_relcom2_tensor;

    torch::Tensor term2_7 = 2.0 * q21_tensor.index({Slice(),Slice(None,1)}) * torch::cross(q21_tensor.index({Slice(),Slice(1,None)}), faces2_r2_relcom2_tensor);
    torch::Tensor term3_7 = 2.0 * torch::sum(q21_tensor.index({Slice(),Slice(1,None)}) * faces2_r2_relcom2_tensor, 1).view({-1, 1}) *q21_tensor.index({Slice(),Slice(1,None)});

    torch::Tensor faces2_r2_r21_relcom2_tensor = term1_7 + term2_7 + term3_7;

    torch::Tensor faces2r21_3d = faces2_r2_r21_relcom2_tensor.view({-1,7,3});

    torch::Tensor ff = faces2r21_3d.index({Slice(),Slice(1,None)});

    auto condition1 = torch::abs(ff.index({Slice(),Slice(),Slice(0,1)})) < 0.001;
    auto condition2 = ff.index({Slice(),Slice(),Slice(2,None)}) >= 0.0;
    auto condition3 = ff.index({Slice(),Slice(),Slice(1,2)}) > 0.0;

    auto condition_all = condition1.view({N_pair,6}) & condition2.view({N_pair,6}) & condition3.view({N_pair,6});

    torch::Tensor sumoc = torch::sum(condition_all,1);

    torch::Tensor wrong_indexes = torch::where(torch::abs(sumoc-1)>0.3)[0];
    if(wrong_indexes.sizes()[0]>0)
    {
      for (int i=0;i<wrong_indexes.sizes()[0];i++)
      {
        condition_all.index_put_({wrong_indexes.index({i})},torch::tensor({1,0,0,0,0,0}));
      }
    }


    torch::Tensor faces2p_inter_tensor = ff.index({condition_all});

    reduced_configs.index_put_({Slice(),Slice(5,None)},torch::arctan2(faces2p_inter_tensor.index({Slice(),Slice(2,None)}),faces2p_inter_tensor.index({Slice(),Slice(1,2)}) ));
    // std::cout<<"1061"<<std::endl;
    if(wrong_indexes.sizes()[0]>0)
    {
      reduced_configs.index_put_({wrong_indexes,Slice(5,None)},0.0);
    }

    /*
    self.rotQ1 = -q_rot1u
    self.rotQ1[:,0] = -self.rotQ1[:,0]
    self.rotQ2 = -q_rot2u
    self.rotQ2[:,0] = -self.rotQ2[:,0]
    */

    torch::Tensor rotQ1_tensor = -q_rot1u_tensor;
    rotQ1_tensor.index_put_({Slice(),Slice(0,1)},-rotQ1_tensor.index({Slice(),Slice(0,1)}));
    torch::Tensor rotQ2_tensor = -q_rot2u_tensor;
    rotQ2_tensor.index_put_({Slice(),Slice(0,1)},-rotQ2_tensor.index({Slice(),Slice(0,1)}));

    /*
    for i in range(6):
        x_data[:,i] = (reduced_configs_torch[:,i] - mins_[i]) / (maxs_[i] - mins_[i])
    */
    std::vector<float> mins_ {0.98,0.0,0.0,0.0,-3.142,0.0};
    std::vector<float> maxs_ {6.31,5.43,3.65,2.04, 3.142,1.572};
    torch::Tensor indices6 = torch::arange(0, 6).to(at::kCUDA);
    torch::Tensor mins_tensor = torch::from_blob(mins_.data(), 6, torch::kFloat32).to(at::kCUDA);
    torch::Tensor maxs_tensor = torch::from_blob(maxs_.data(), 6, torch::kFloat32).to(at::kCUDA);

    auto t2 = std::chrono::high_resolution_clock::now();
    auto s3 = std::chrono::high_resolution_clock::now();


    torch::Tensor x_data_tensor = (reduced_configs.index({Slice(),indices6}) - mins_tensor.index({indices6}))/(maxs_tensor.index({indices6}) - mins_tensor.index({indices6}));
    x_data_tensor = x_data_tensor.to(torch::kFloat32);
    torch::jit::IValue x_data_ival(x_data_tensor);
    std::vector<torch::jit::IValue> inputs_selector = {x_data_ival};

    at::Tensor selector_output_tensor = m_selector.forward(inputs_selector).toTensor();

    // selector_output = np.argmax(selector_output,axis=1) //
    // interacting_index_nn = np.where(selector_output==1)[0] //
    selector_output_tensor = torch::argmax(selector_output_tensor,1);
    std::vector<torch::Tensor> interacting_index_nn_vector = torch::where(selector_output_tensor==1);
    torch::Tensor interacting_index_nn_tensor = interacting_index_nn_vector[0];

    /*
    interacting_reduced_configs = reduced_configs[interacting_index_nn]
    pair0 = pair0[interacting_index_nn]
    pair1 = pair1[interacting_index_nn]
    self.rotQ1 = self.rotQ1[interacting_index_nn]
    self.rotQ2 = self.rotQ2[interacting_index_nn]
    multiplier_tork = multiplier_tork[interacting_index_nn]
    multiplier_force = multiplier_force[interacting_index_nn]
    will_switch = will_switch[interacting_index_nn]
    self.NpairInter = len(pair0)
    x_true = interacting_reduced_configs
    x_data = x_data[interacting_index_nn]
    */
    torch::Tensor interacting_reduced_configs_tensor = reduced_configs.index({interacting_index_nn_tensor});
    torch::Tensor xtt = interacting_reduced_configs_tensor;


    x_data_tensor = x_data_tensor.index({interacting_index_nn_tensor});
    pair0_tensor = pair0_tensor.index({interacting_index_nn_tensor});
    pair1_tensor = pair1_tensor.index({interacting_index_nn_tensor});

    rotQ1_tensor = rotQ1_tensor.index({interacting_index_nn_tensor});
    rotQ2_tensor = rotQ2_tensor.index({interacting_index_nn_tensor});

    multiplier_torque_tensor = multiplier_torque_tensor.index({interacting_index_nn_tensor});
    multiplier_force_tensor = multiplier_force_tensor.index({interacting_index_nn_tensor});
    will_switch_booltensor = will_switch_booltensor.index({interacting_index_nn_tensor});

    int N_inter_tensor = interacting_index_nn_tensor.sizes()[0];

    /*
    FORWARD DERIVATION
    xx_ender = torch.tile(xx_ender,(7,1))
    dxx = torch.zeros_like(xx_ender)
    for i in range(6):
        dxx[(i+1)*mm:(i+2)*mm,i] = dx
    xx_ender = xx_ender + dxx
    e_xx = self.force.energy_net(xx_ender)
    */
    auto t3 = std::chrono::high_resolution_clock::now();
    auto s4 = std::chrono::high_resolution_clock::now();
    torch::Tensor xx_ender_tensor = x_data_tensor.tile({7,1});
    std::vector<float> dxs_v {0.006,0.01,0.03,0.03,0.01,0.03};
    torch::Tensor dxs_tensor = torch::from_blob(dxs_v.data(), 6, torch::kFloat32).to(at::kCUDA);
    torch::Tensor dxx_tensor = torch::zeros_like(xx_ender_tensor);
    for(int i =0; i < 6 ;i++)
    {
      dxx_tensor.index_put_({Slice((i+1)*N_inter_tensor,(i+2)*N_inter_tensor),i},dxs_tensor.index({i}));
    }
    xx_ender_tensor = xx_ender_tensor + dxx_tensor;
    xx_ender_tensor = xx_ender_tensor.to(torch::kFloat32);

    // std::cout<<"Energy Input size: "<<xx_ender_tensor.sizes()<<std::endl;

    torch::jit::IValue xx_ender_ival(xx_ender_tensor);
    std::vector<torch::jit::IValue> inputs_energy = {xx_ender_ival};
    at::Tensor e_xx_tensor = m_energy.forward(inputs_energy).toTensor();

    // std::cout<<"Energy Output size: "<<e_xx_tensor.sizes()<<std::endl;



    /*
    e_x = e_xx[:mm]
    e_xx = torch.flatten(e_xx)
    e_xdx = torch.zeros((mm,6))
    for i in range(1,7):
        e_xdx[:,i-1] = e_xx[i*mm:(i+1)*mm]
    gradient = (e_xdx - e_x)/dx
    */
    torch::Tensor e_x_tensor = e_xx_tensor.index({Slice(0,N_inter_tensor)});
    torch::Tensor e_xdx_tensor = torch::zeros({N_inter_tensor,6},torch::dtype(torch::kFloat32)).to(at::kCUDA);
    for(int i =1; i < 7 ;i++)
    {
      e_xdx_tensor.index_put_({Slice(),i-1},e_xx_tensor.index({Slice(i*N_inter_tensor,(i+1)*N_inter_tensor)}).view({-1}));
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    auto s5 = std::chrono::high_resolution_clock::now();


    torch::Tensor gt = (e_xdx_tensor - e_x_tensor)/(dxs_tensor);

    // grados *= (en_max - en_min) / (np.array(maxs_) - np.array(mins_)) //

    float en_max_tensor = 17.0;
    float en_min_tensor = -5.1;
    gt *= (en_max_tensor - en_min_tensor)/(maxs_tensor - mins_tensor);

    /*
    a1 = x_true[:,3]
    a2 = x_true[:,4]

    tz = gradient[:,0]*(-x_true[:,1]) + gradient[:,1]*(x_true[:,0]) + gradient[:,3]*np.cos(x_true[:,4])
    tz = tz + gradient[:,4]*(-np.cos(a1)*np.sin(a2)/np.sin(a1)) + gradient[:,5]*np.sin(a2)*(1-np.cos(a1))/np.sin(a1)

    ty = gradient[:,0]*(x_true[:,2]) + gradient[:,1]*0 + gradient[:,2]*(-x_true[:,0])
    ty = ty + gradient[:,3]*-np.sin(x_true[:,4]) + gradient[:,4]*(-np.cos(a2)*np.cos(a1)/np.sin(a1)) +  gradient[:,5]*(np.cos(a2)*(1-np.cos(a1))/np.sin(a1))

    tx = gradient[:,0]*0 + gradient[:,1]*(-x_true[:,2]) + gradient[:,2]*(x_true[:,1])
    tx = tx + gradient[:,3]*0.0 + gradient[:,4]*1.0 +  gradient[:,5]*1.0

    */
    torch::Tensor a1t = xtt.index({Slice(),3});
    torch::Tensor a2t = xtt.index({Slice(),4});

    // not needed for python code, used only here since float32 is used here and if a1t=0.0 /sin(a1t) is nan //
    a1t.index_put_({a1t==0.0},0.999);


    torch::Tensor tz = gt.index({Slice(),0})*-xtt.index({Slice(),1}) + gt.index({Slice(),1})*xtt.index({Slice(),0}) + gt.index({Slice(),3})*torch::cos(xtt.index({Slice(),4}));
    tz += gt.index({Slice(),4})*(-torch::cos(a1t)*torch::sin(a2t)/torch::sin(a1t)) + gt.index({Slice(),5})*torch::sin(a2t)*(1.0-torch::cos(a1t))/torch::sin(a1t);


    torch::Tensor ty =  gt.index({Slice(),0})*xtt.index({Slice(),2}) + gt.index({Slice(),2})*-xtt.index({Slice(),0}) + gt.index({Slice(),3})*-torch::sin(a2t);
    ty +=  gt.index({Slice(),4})*(-torch::cos(a2t)*torch::cos(a1t)/torch::sin(a1t)) + gt.index({Slice(),5})*(torch::cos(a2t)*(1.0-torch::cos(a1t))/torch::sin(a1t));

    torch::Tensor tx =  gt.index({Slice(),1})*-xtt.index({Slice(),2}) +  gt.index({Slice(),2})*xtt.index({Slice(),1}) + gt.index({Slice(),4}) + gt.index({Slice(),5});

    // std::cout<<tx.index({7960})<<" "<<ty.index({7960})<<" "<<tz.index({7960})<<std::endl;
    // exit(0);

/*
forces_inter[:,0] = gradient[:,0]
forces_inter[:,1] = gradient[:,1]
forces_inter[:,2] = gradient[:,2]

torks_inter[:,0] = tx
torks_inter[:,1] = ty
torks_inter[:,2] = tz

forces_inter[forces_inter>100.0] = 100.0
torks_inter[torks_inter>100.0] = 100.0

forces_inter[forces_inter<-100.0] = -100.0
torks_inter[torks_inter<-100.0] = -100.0
*/
    torch::Tensor forces_inter_tensor = gt.index({Slice(),Slice(0,3)});
    torch::Tensor torks_inter_tensor = torch::zeros({N_inter_tensor,3},torch::dtype(torch::kFloat32)).to(at::kCUDA);
    torks_inter_tensor.index_put_({Slice(),0},tx);
    torks_inter_tensor.index_put_({Slice(),1},ty);
    torks_inter_tensor.index_put_({Slice(),2},tz);

    forces_inter_tensor.index_put_({forces_inter_tensor>100.0},100.0);
    forces_inter_tensor.index_put_({forces_inter_tensor<-100.0},-100.0);

    torks_inter_tensor.index_put_({torks_inter_tensor>100.0},100.0);
    torks_inter_tensor.index_put_({torks_inter_tensor<-100.0},-100.0);

/*
for i in range(self.NpairInter):
    if(will_switch[i]==1):
        forces_inter[i,[1,2]] = forces_inter[i,[2,1]]
        torks_inter[i,[1,2]] = -torks_inter[i,[2,1]]
        torks_inter[i,0] = -torks_inter[i,0]
    forces_inter = forces_inter*multiplier_force
    torks_inter = torks_inter*multiplier_tork
*/
    forces_inter_tensor.index_put_({will_switch_booltensor,Slice(1,3)},forces_inter_tensor.index({will_switch_booltensor,Slice()}).index({Slice(),torch::tensor({2,1})}));
    torks_inter_tensor.index_put_({will_switch_booltensor,Slice(1,3)},-torks_inter_tensor.index({will_switch_booltensor,Slice()}).index({Slice(),torch::tensor({2,1})}));
    torks_inter_tensor.index_put_({will_switch_booltensor,0},-torks_inter_tensor.index({will_switch_booltensor,0}));

    forces_inter_tensor *= multiplier_force_tensor;
    torks_inter_tensor *= multiplier_torque_tensor;
    // std::cout<<rotQ2_tensor<<std::endl;
    // exit(0);
/*
    forces_inter_box = rotate(self.rotQ2,forces_inter)
    forces_inter_box = rotate(self.rotQ1,forces_inter_box)

    torks_inter_box = rotate(self.rotQ2,torks_inter)
    torks_inter_box = rotate(self.rotQ1,torks_inter_box)
*/

  // forces_inter_box = rotate(self.rotQ2,forces_inter) //
  torch::Tensor coef1_8 = rotQ2_tensor.index({Slice(),Slice(None,1)}) * rotQ2_tensor.index({Slice(),Slice(None,1)}) - torch::sum(rotQ2_tensor.index({Slice(),Slice(1,None)})*rotQ2_tensor.index({Slice(),Slice(1,None)}),1).view({-1, 1});
  torch::Tensor term1_8 = coef1_8 * forces_inter_tensor;

  torch::Tensor term2_8 = 2.0 * rotQ2_tensor.index({Slice(),Slice(None,1)}) * torch::cross(rotQ2_tensor.index({Slice(),Slice(1,None)}), forces_inter_tensor);
  torch::Tensor term3_8 = 2.0 * torch::sum(rotQ2_tensor.index({Slice(),Slice(1,None)}) * forces_inter_tensor, 1).view({-1, 1}) *rotQ2_tensor.index({Slice(),Slice(1,None)});

  torch::Tensor forces_inter_box_tensor = term1_8 + term2_8 + term3_8;

  torch::Tensor coef1_9 = rotQ1_tensor.index({Slice(),Slice(None,1)}) * rotQ1_tensor.index({Slice(),Slice(None,1)}) - torch::sum(rotQ1_tensor.index({Slice(),Slice(1,None)})*rotQ1_tensor.index({Slice(),Slice(1,None)}),1).view({-1, 1});
  torch::Tensor term1_9 = coef1_9 * forces_inter_box_tensor;

  torch::Tensor term2_9 = 2.0 * rotQ1_tensor.index({Slice(),Slice(None,1)}) * torch::cross(rotQ1_tensor.index({Slice(),Slice(1,None)}), forces_inter_box_tensor);
  torch::Tensor term3_9 = 2.0 * torch::sum(rotQ1_tensor.index({Slice(),Slice(1,None)}) * forces_inter_box_tensor, 1).view({-1, 1}) *rotQ1_tensor.index({Slice(),Slice(1,None)});

  forces_inter_box_tensor = term1_9 + term2_9 + term3_9;
  // std::cout<<torks_inter_tensor.index({7960})<<std::endl;
  // std::cout<<rotQ2_tensor.index({7960})<<std::endl;

  torch::Tensor coef1_81 = rotQ2_tensor.index({Slice(),Slice(None,1)}) * rotQ2_tensor.index({Slice(),Slice(None,1)}) - torch::sum(rotQ2_tensor.index({Slice(),Slice(1,None)})*rotQ2_tensor.index({Slice(),Slice(1,None)}),1).view({-1, 1});
  torch::Tensor term1_81 = coef1_81 * torks_inter_tensor;

  torch::Tensor term2_81 = 2.0 * rotQ2_tensor.index({Slice(),Slice(None,1)}) * torch::cross(rotQ2_tensor.index({Slice(),Slice(1,None)}), torks_inter_tensor);
  torch::Tensor term3_81 = 2.0 * torch::sum(rotQ2_tensor.index({Slice(),Slice(1,None)}) * torks_inter_tensor, 1).view({-1, 1}) *rotQ2_tensor.index({Slice(),Slice(1,None)});

  torch::Tensor torks_inter_box_tensor = term1_81 + term2_81 + term3_81;

  torch::Tensor coef1_91 = rotQ1_tensor.index({Slice(),Slice(None,1)}) * rotQ1_tensor.index({Slice(),Slice(None,1)}) - torch::sum(rotQ1_tensor.index({Slice(),Slice(1,None)})*rotQ1_tensor.index({Slice(),Slice(1,None)}),1).view({-1, 1});
  torch::Tensor term1_91 = coef1_91 * torks_inter_box_tensor;

  torch::Tensor term2_91 = 2.0 * rotQ1_tensor.index({Slice(),Slice(None,1)}) * torch::cross(rotQ1_tensor.index({Slice(),Slice(1,None)}), torks_inter_box_tensor);
  torch::Tensor term3_91 = 2.0 * torch::sum(rotQ1_tensor.index({Slice(),Slice(1,None)}) * torks_inter_box_tensor, 1).view({-1, 1}) *rotQ1_tensor.index({Slice(),Slice(1,None)});

  torks_inter_box_tensor = term1_91 + term2_91 + term3_91;

  // std::cout<<torch::where(torch::isnan(torks_inter_box_tensor.index({Slice(),0}))==true)[0]<<std::endl;
  // std::cout<<torks_inter_box_tensor.index({7960})<<std::endl;
  // exit(0);
/*
forces_net = np.zeros((self.Nparticles,3))
torks_net = np.zeros((self.Nparticles,3))
for i in range(self.NpairInter):
    forces_net[pair0[i]] += forces_inter_box[i]
    torks_net[pair0[i]] += torks_inter_box[i]
self.torks = torks_net
self.forces = forces_net
or
forces_net.index_add_(0, torch.from_numpy(pair0), torch.from_numpy(forces_inter_box))

*/

torch::Tensor forces_net_tensor = torch::zeros({m_Nparticles,3},torch::dtype(torch::kFloat32)).to(at::kCUDA);
torch::Tensor torks_net_tensor = torch::zeros({m_Nparticles,3},torch::dtype(torch::kFloat32)).to(at::kCUDA);


forces_net_tensor.index_add_(0,pair0_tensor,forces_inter_box_tensor);
torks_net_tensor.index_add_(0,pair0_tensor,torks_inter_box_tensor);

// std::cout<<torks_net_tensor.index({1259})<<std::endl;

forces_net_tensor = forces_net_tensor.to(torch::kCPU);
torks_net_tensor = torks_net_tensor.to(torch::kCPU);

std::vector<float> forces_net(forces_net_tensor.data<float>(), forces_net_tensor.data<float>() + forces_net_tensor.numel());
std::vector<float> torks_net(torks_net_tensor.data<float>(), torks_net_tensor.data<float>() + torks_net_tensor.numel());

m_net_forces.clear();
m_net_torks.clear();

m_net_forces.resize(m_Nparticles);
m_net_torks.resize(m_Nparticles);

std::fill(m_net_forces.begin(),m_net_forces.end(),double_3(0.0,0.0,0.0));
std::fill(m_net_torks.begin(),m_net_torks.end(),double_3(0.0,0.0,0.0));
auto t5 = std::chrono::high_resolution_clock::now();
auto s6 = std::chrono::high_resolution_clock::now();
for(int i = 0; i < m_Nparticles; i++)
{
  m_net_torks[i] = double_3(torks_net[i*3],torks_net[i*3+1],torks_net[i*3+2]);
  m_net_forces[i] = double_3(forces_net[i*3],forces_net[i*3+1],forces_net[i*3+2]);

}

// std::cout<<m_net_torks[1259][0]<<" "<<m_net_torks[1259][1]<<" "<<m_net_torks[1259][2]<<std::endl;

for(int i = 0; i < m_Nparticles; i++)
{
  // just a rotation with conjugate
  double_4 q = -m_quat[i];
  q[0] = -q[0];
  double_3 qv = double_3(q[1],q[2],q[3]);
  double_3 v = m_net_torks[i];

  double_3 r = 2.0*q[0]*cross(qv,v) + (q[0]*q[0] - q[1]*q[1] - q[2]*q[2] - q[3]*q[3])*v + 2.0*dot(qv,v)*qv;
  m_net_torks[i] = r;
}

auto t6 = std::chrono::high_resolution_clock::now();

// float t_t1 = std::chrono::duration<float, std::milli>(t1-s1).count();
// float t_t2 = std::chrono::duration<float, std::milli>(t2-s2).count();
// float t_t3 = std::chrono::duration<float, std::milli>(t3-s3).count();
// float t_t4 = std::chrono::duration<float, std::milli>(t4-s4).count();
// float t_t5 = std::chrono::duration<float, std::milli>(t5-s5).count();
// float t_t6 = std::chrono::duration<float, std::milli>(t6-s6).count();

// tt1 += t_one;
// tt2 += t_mid;
// tt3 += t_two;
//
// if(m_ts==1000)
// {
//   std::cout<<"Timestep one: "<<tt1<<std::endl;
//   std::cout<<"Timestep mid: "<<tt2<<std::endl;
//   std::cout<<"Timestep end: "<<tt3<<std::endl;
// }


}

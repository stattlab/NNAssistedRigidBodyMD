// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: mphoward


#include "NeighborListTree.h"


NeighborListTree::NeighborListTree(Sim* sim, Box* box)
    : m_Sim(sim),m_Box(box),m_box_changed(true), m_max_num_changed(true), m_remap_particles(true),
      m_type_changed(true), m_n_images(0)
    {

      m_r_cut.resize(m_Sim->getNTypes());
      for (unsigned int i = 0; i < m_Sim->getNTypes(); ++i)
      {
         m_r_cut[i].resize(m_Sim->getNTypes());
      }
    }


NeighborListTree::NeighborListTree(Sim* sim, Box* box, double cutoff) // cutoff not used
    : m_Sim(sim),m_Box(box),m_box_changed(true), m_max_num_changed(true), m_remap_particles(true),
      m_type_changed(true), m_n_images(0)
    {
      m_r_cut.resize(m_Sim->getNTypes());
      for (unsigned int i = 0; i < m_Sim->getNTypes(); ++i)
      {
         m_r_cut[i].resize(m_Sim->getNTypes());
      }
    }


// instead of having two versions of each function for molecule and particle
// trees
// 1) Create a separate class for molecule tree
// 2) Make maxnumber of particles-molecules (things you make tree with) a member
// of the class instead of calling them from the simulation class from the beginning
// and update them when necessary after the moves than you would only have duplicate
// functions where you need the m_Particles or m_Molecules
// 3) In addition to 2) have the pointer vector of particles or molecules as a member
// of the tree this way you dont need to duplicate any function similarly update them
// when necessary


NeighborListTree::~NeighborListTree()
{

    }

void NeighborListTree::buildNlist()
{
    // allocate the memory as needed and sort particles
    std::cout<<"buildN list"<<std::endl;
    setupTree();
   buildHeadList();
   // rebuild the list until there is no overflow

    bool overflowed = false;
    do
        {
          // allocate the memory as needed and sort particles
          setupTree();
          // build the trees
          buildTree();
          // now walk the trees
          traverseTree();
        overflowed = checkConditions();
        // if we overflowed, need to reallocate memory and reset the conditions
        if (overflowed)
            {
            // always rebuild the head list after an overflow
            buildHeadList();

            // zero out the conditions for the next build
            resetConditions();
            }
        } while (overflowed);

}

// void NeighborListTree::buildNlistMolecules()
// {
//
//     // allocate the memory as needed and sort particles
//     setupTreeMolecules();
//
//    buildHeadListMolecules();
//
//     // rebuild the list until there is no overflow
//     bool overflowed = false;
//     do
//         {
//           // allocate the memory as needed and sort particles
//           setupTreeMolecules();
//           // build the trees
//           buildTreeMolecules();
//           // now walk the trees
//           traverseTreeMolecules();
//
//         overflowed = checkConditionsMolecules();
//         // if we overflowed, need to reallocate memory and reset the conditions
//         if (overflowed)
//             {
//             // always rebuild the head list after an overflow
//             buildHeadListMolecules();
//
//             // zero out the conditions for the next build
//             resetConditions();
//             }
//         } while (overflowed);
//
// }

void NeighborListTree::buildTrees()
{

    // allocate the memory as needed and sort particles
    setupTree();
    buildHeadList();
    // build the trees
    buildTree();

}

// void NeighborListTree::buildTreesMolecules()
// {
//
//     // allocate the memory as needed and sort particles
//     setupTreeMolecules();
//     buildHeadListMolecules();
//     // build the trees
//     buildTreeMolecules();
//
// }


void NeighborListTree::setupTree()
{
  // std::cout<<"setup tree"<<std::endl;
    if (m_max_num_changed)
        {
        reallocate();
        m_aabbs.resize(m_Sim->getMaxN());
        //for ( unsigned int i=0; i < m_Sim->getMaxN();i++) m_aabbs[i] = AABB(Point(0,0,0),0.0);
        m_map_pid_tree.resize(m_Sim->getMaxN());
        m_max_num_changed = false;
        }

    if (m_type_changed)
        {
        // double corruption happens if we just resize due to the way the AABBNodes are allocated
        // so first destroy all of the trees from the vector and then resize. could probably be fixed using scoped
        // pointers as well
        m_aabb_trees.clear();
        m_aabb_trees.resize(m_Sim->getNTypes());
        m_num_per_type.resize(m_Sim->getNTypes(), 0);
        m_type_head.resize(m_Sim->getNTypes(), 0);
        reallocateTypes();
        slotRemapParticles();
        m_type_changed = false;
        }

    if (m_remap_particles)
        {
        mapParticlesByType();
        m_remap_particles = false;
        }

    if (m_box_changed)
        {
        updateImageVectors();
        m_box_changed = false;
        }
}

// void NeighborListTree::setupTreeMolecules()
// {
//     if (m_max_num_changed)
//         {
//         reallocateMolecules();
//         m_aabbs.resize(m_Sim->getMaxM());
//         //for ( unsigned int i=0; i < m_Sim->getMaxN();i++) m_aabbs[i] = AABB(Point(0,0,0),0.0);
//
//         m_map_pid_tree.resize(m_Sim->getMaxM());
//
//         m_max_num_changed = false;
//         }
//
//     if (m_type_changed)
//         {
//         // double corruption happens if we just resize due to the way the AABBNodes are allocated
//         // so first destroy all of the trees from the vector and then resize. could probably be fixed using scoped
//         // pointers as well
//         m_aabb_trees.clear();
//         m_aabb_trees.resize(m_Sim->getMTypes());
//
//         m_num_per_type.resize(m_Sim->getMTypes(), 0);
//         m_type_head.resize(m_Sim->getMTypes(), 0);
//
//         reallocateTypesMolecules();
//         slotRemapParticles();
//
//         m_type_changed = false;
//         }
//
//
//     if (m_remap_particles)
//         {
//         mapParticlesByTypeMolecule();
//         m_remap_particles = false;
//         }
//
//
//     if (m_box_changed)
//         {
//         updateImageVectors();
//         m_box_changed = false;
//         }
//
// }

void NeighborListTree::mapParticlesByType()
{

    // clear out counters
    unsigned int n_types = m_Sim->getNTypes();

    for (unsigned int i=0; i < n_types; ++i)
        {
        m_num_per_type[i] = 0;
        }

    // histogram all particles on this rank, and accumulate their positions within the tree
    unsigned int n_local = m_Sim->getMaxN();
    std::vector <Particle*> particles = m_Sim->getParticles();

    for (unsigned int i=0; i < n_local; ++i)
        {
        unsigned int my_type = particles[i]->getType();
        m_map_pid_tree[i] = m_num_per_type[my_type]; // global id i is particle num_per_type after head of my_type
        ++m_num_per_type[my_type];
        }
    // set the head for each type in m_aabbs by looping back over the types
    unsigned int local_head = 0;
    for (unsigned int i=0; i < n_types; ++i)
        {
        m_type_head[i] = local_head;
        local_head += m_num_per_type[i];
        }

}

// void NeighborListTree::mapParticlesByTypeMolecule()
// {
//     // clear out counters
//     unsigned int n_types = m_Sim->getMTypes();
//     for (unsigned int i=0; i < n_types; ++i)
//         {
//         m_num_per_type[i] = 0;
//         }
//
//     // histogram all particles on this rank, and accumulate their positions within the tree
//     unsigned int n_local = m_Sim->getMaxM();
//     std::vector <Molecule*> molecules = m_Sim->getMolecules();
//
//     for (unsigned int i=0; i < n_local; ++i)
//         {
//         unsigned int my_type = molecules[i]->getType();
//         m_map_pid_tree[i] = m_num_per_type[my_type]; // global id i is particle num_per_type after head of my_type
//         ++m_num_per_type[my_type];
//         }
//
//     // set the head for each type in m_aabbs by looping back over the types
//     unsigned int local_head = 0;
//     for (unsigned int i=0; i < n_types; ++i)
//         {
//         m_type_head[i] = local_head;
//         local_head += m_num_per_type[i];
//         }
// }


void NeighborListTree::reallocate()
{
    // resize the head list and number of neighbors per particle
    m_head_list.resize(m_Sim->getMaxN());
    m_n_neigh.resize(m_Sim->getMaxN());

}

// void NeighborListTree::reallocateMolecules()
// {
//     // resize the head list and number of neighbors per particle
//     m_head_list.resize(m_Sim->getMaxM());
//     m_n_neigh.resize(m_Sim->getMaxM());
//
// }


void NeighborListTree::reallocateTypes()
{
    m_Nmax.resize(m_Sim->getNTypes());

    // flood Nmax with 4s initially
        {
        for (unsigned int i=0; i < m_Sim->getNTypes(); ++i)
            {
            m_Nmax[i] = 4;
            }
        }

    m_conditions.resize(m_Sim->getNTypes());
    resetConditions();
}

// void NeighborListTree::reallocateTypesMolecules()
// {
//     m_Nmax.resize(m_Sim->getMTypes());
//
//     // flood Nmax with 4s initially
//         {
//         for (unsigned int i=0; i < m_Sim->getMTypes(); ++i)
//             {
//             m_Nmax[i] = 4;
//             }
//         }
//
//     m_conditions.resize(m_Sim->getMTypes());
//     resetConditions();
// }


void NeighborListTree::buildHeadList()
{

    unsigned int headAddress = 0;
        {

        std::vector <Particle*> particles = m_Sim->getParticles();

        for (unsigned int i=0; i < m_Sim->getMaxN(); ++i)
            {
            m_head_list[i] = headAddress;

            // move the head address along
            unsigned int myType = particles[i]->getType();
            headAddress += m_Nmax[myType];
            }
        }

    resizeNlist(headAddress);
}

// void NeighborListTree::buildHeadListMolecules()
// {
//
//     unsigned int headAddress = 0;
//         {
//
//         std::vector <Molecule*> molecules = m_Sim->getMolecules();
//
//         for (unsigned int i=0; i < m_Sim->getMaxM(); ++i)
//             {
//             m_head_list[i] = headAddress;
//
//             // move the head address along
//             unsigned int myType = molecules[i]->getType();
//             headAddress += m_Nmax[myType];
//             }
//         }
//
//     resizeNlist(headAddress);
// }

void NeighborListTree::resizeNlist(unsigned int size)
{
    if (size > m_nlist.size())
        {

        unsigned int alloc_size = m_nlist.size() ? m_nlist.size() : 1;

        while (size > alloc_size)
            {
            alloc_size = ((unsigned int) (((float) alloc_size) * 1.125f)) + 1 ;
            }

        // round up to nearest multiple of 4
        alloc_size = (alloc_size > 4) ? (alloc_size + 3) & ~3 : 4;

        m_nlist.resize(alloc_size);
        }
}


bool NeighborListTree::checkConditions()
{
    bool result = false;

    for (unsigned int i=0; i < m_Sim->getNTypes(); ++i)
        {
        if (m_conditions[i] > m_Nmax[i])
            {
            m_Nmax[i] = (m_conditions[i] > 4) ? (m_conditions[i] + 3) & ~3 : 4;
            result = true;
            }
        }

    return result;
}

// bool NeighborListTree::checkConditionsMolecules()
// {
//     bool result = false;
//
//     for (unsigned int i=0; i < m_Sim->getMTypes(); ++i)
//         {
//         if (m_conditions[i] > m_Nmax[i])
//             {
//             m_Nmax[i] = (m_conditions[i] > 4) ? (m_conditions[i] + 3) & ~3 : 4;
//             result = true;
//             }
//         }
//
//     return result;
// }


void NeighborListTree::resetConditions()
{
    std::fill(m_conditions.begin(), m_conditions.end(), 0);
}

// void NeighborListTree::removePbcforPmf()
// {
//   std::cout<<"remove pbc"<<std::endl;
//   m_n_images = 1;
//   m_image_list.resize(1);
//   m_image_list[0] = Point(0.0, 0.0, 0.0);
// }

void NeighborListTree::updateImageVectors()
    {
      // std::cout<<"updating image vectors"<<std::endl;

    const Point box = m_Box->getDim();
    std::vector<bool>  periodic = m_Box->getPeriodic();

    // now compute the image vectors
    // each dimension increases by one power of 3
    m_n_images = 1;
    for (unsigned int dim = 0; dim < 3; ++dim)
        {
        m_n_images *= 3;
        }

    // reallocate memory if necessary
    if (m_n_images > m_image_list.size())
        {
        m_image_list.resize(m_n_images);
        }

    Point latt_a = Point(box.getX(),0,0);
    Point latt_b = Point(0,box.getY(),0);
    Point latt_c = Point(0,0,box.getZ());

    // there is always at least 1 image, which we put as our first thing to look at
    m_image_list[0] = Point(0.0, 0.0, 0.0);

    // iterate over all other combinations of images, skipping those that are
    unsigned int n_images = 1;
    for (int i=-1; i <= 1 && n_images < m_n_images; ++i)
        {
        for (int j=-1; j <= 1 && n_images < m_n_images; ++j)
            {
            for (int k=-1; k <= 1 && n_images < m_n_images; ++k)
                {
                if (!(i == 0 && j == 0 && k == 0))
                    {
                    // skip any periodic images if we don't have periodicity
                    if (i != 0 && !periodic[0]) continue;
                    if (j != 0 && !periodic[1]) continue;
                    if (k != 0 && !periodic[2]) continue;

                    m_image_list[n_images] = latt_a*i + latt_b*j + latt_c*k;
                    ++n_images;
                    }
                }
            }
        }

    }


void NeighborListTree::buildTree()
    {

    std::vector <Particle*> particles = m_Sim->getParticles();

    // construct a point AABB for each particle and push it into the right spot in the AABB list
    for (unsigned int i=0; i < m_Sim->getMaxN(); ++i)
        {
        // make a point particle AABB
        Point my_pos(particles[i]->getOrigin());

        unsigned int my_type = particles[i]->getType();
        unsigned int my_aabb_idx = m_type_head[my_type] + m_map_pid_tree[i];
        m_aabbs[my_aabb_idx] = hpmc::detail::AABB(my_pos,i);
        }

    // call the tree build routine, one tree per type
    for (unsigned int i=0; i < m_Sim->getNTypes(); ++i)
        {
        if (m_num_per_type[i] > 0)
            {
            m_aabb_trees[i].buildTree(&(m_aabbs[0]) + m_type_head[i], m_num_per_type[i]);
            }
        }

    }



// void NeighborListTree::buildTreeMolecules()
//     {
//
//     std::vector <Molecule*> molecules = m_Sim->getMolecules();
//
//     // construct a point AABB for each particle and push it into the right spot in the AABB list
//     for (unsigned int i=0; i < m_Sim->getMaxM(); ++i)
//         {
//         // make a point particle AABB
//         Point my_pos(molecules[i]->getCofM());
//
//         unsigned int my_type = molecules[i]->getType();
//         unsigned int my_aabb_idx = m_type_head[my_type] + m_map_pid_tree[i];
//         m_aabbs[my_aabb_idx] = hpmc::detail::AABB(my_pos,i);
//         }
//
//     // call the tree build routine, one tree per type
//     for (unsigned int i=0; i < m_Sim->getMTypes(); ++i)
//         {
//         if (m_num_per_type[i] > 0)
//             {
//             m_aabb_trees[i].buildTree(&(m_aabbs[0]) + m_type_head[i], m_num_per_type[i]);
//             }
//         }
//
//     }




void NeighborListTree::updateParticleTree(unsigned int index)
{
  std::vector <Particle*> particles = m_Sim->getParticles();
  // make a point particle AABB
  Point my_pos(particles[index]->getOrigin());
  unsigned int my_type = particles[index]->getType();
  unsigned int my_aabb_idx = m_type_head[my_type] + m_map_pid_tree[index];
  //std::cout << " my_aabb_idx "<< my_aabb_idx <<std::endl;
  //std::cout << "len m_aabbs "<< m_aabbs.size()<<std::endl;
 // for (unsigned int i=0; i< m_aabbs.size();i++)
//  {
//      std::cout<<i << " "<< m_aabbs[i].getPosition().x[0]<<std::endl;
// }
 // std::cout<<" end print tree"<<std::endl;
  hpmc::detail::AABB aabb_new = hpmc::detail::AABB(my_pos,index);
 // std::cout<<" new node "<< aabb_new.getPosition().x[0]<<std::endl;
  m_aabbs[my_aabb_idx] = aabb_new; //AABB(my_pos,index);
 // std::cout << " index "<< index <<std::endl;
  m_aabb_trees[my_type].update(m_map_pid_tree[index], m_aabbs[my_aabb_idx]);
 // std::cout << " type "<< my_type <<std::endl;
 // exit(0);
}

// void NeighborListTree::updateMoleculeTree(unsigned int index)
// {
//   std::vector <Molecule*> molecules = m_Sim->getMolecules();
//   // make a point particle AABB
//   Point my_pos(molecules[index]->getCM());
//   unsigned int my_type = molecules[index]->getType();
//   unsigned int my_aabb_idx = m_type_head[my_type] + m_map_pid_tree[index];
//   //std::cout << " my_aabb_idx "<< my_aabb_idx <<std::endl;
//   //std::cout << "len m_aabbs "<< m_aabbs.size()<<std::endl;
//  // for (unsigned int i=0; i< m_aabbs.size();i++)
// //  {
// //      std::cout<<i << " "<< m_aabbs[i].getPosition().x[0]<<std::endl;
// // }
//  // std::cout<<" end print tree"<<std::endl;
//   hpmc::detail::AABB aabb_new = hpmc::detail::AABB(my_pos,index);
//  // std::cout<<" new node "<< aabb_new.getPosition().x[0]<<std::endl;
//   m_aabbs[my_aabb_idx] = aabb_new; //AABB(my_pos,index);
//  // std::cout << " index "<< index <<std::endl;
//   m_aabb_trees[my_type].update(m_map_pid_tree[index], m_aabbs[my_aabb_idx]);
//  // std::cout << " type "<< my_type <<std::endl;
//  // exit(0);
// }
//

std::vector<unsigned int> NeighborListTree::findNeighborsParticleType(unsigned int index, unsigned int type, double cutoff)
{

   // acquire particle data
   std::vector <Particle*> particles = m_Sim->getParticles();
   std::vector<unsigned int> neigh;

    // read in the current position and orientation
    const Point pos_i = particles[index]->getOrigin();
    const unsigned int type_i = particles[index]->getType();

    //const unsigned int Nmax_i = m_Nmax[type_i];
    //const unsigned int nlist_head_i = m_head_list[index];

    //unsigned int n_neigh_i = 0;

    unsigned int cur_pair_type = type;

    // pass on empty types
    if (!m_num_per_type[cur_pair_type])
        return neigh;

    double r_cut = 0;
    if (cutoff != -1)
    {
      r_cut = cutoff;
    }
    else
    {
      // Check if this tree type should be excluded by r_cut(i,j) <= 0.0
     r_cut = m_r_cut[type_i][cur_pair_type];
    }

    if (r_cut <= 0.0)
        return neigh;

    // Determine the minimum r_cut_i for this particle pair
    double r_cut_i = r_cut;
    double r_cutsq_i = r_cut_i*r_cut_i;
    //std::cout<<r_cut_i<<std::endl;
    hpmc::detail::AABBTree *cur_aabb_tree = &m_aabb_trees[cur_pair_type];

    for (unsigned int cur_image = 0; cur_image < m_n_images; ++cur_image) // for each image vector
        {
        // make an AABB for the image of this particle
        Point pos_i_image = pos_i + m_image_list[cur_image];
        hpmc::detail::AABB aabb = hpmc::detail::AABB(pos_i_image, r_cut_i);

        // stackless traversal of the tree
        for (unsigned int cur_node_idx = 0; cur_node_idx < cur_aabb_tree->getNumNodes(); ++cur_node_idx)
            {
            if (overlap(cur_aabb_tree->getNodeAABB(cur_node_idx), aabb))
                {
                if (cur_aabb_tree->isNodeLeaf(cur_node_idx))
                    {
                    for (unsigned int cur_p = 0; cur_p < cur_aabb_tree->getNodeNumParticles(cur_node_idx); ++cur_p)
                        {
                        // neighbor j
                        unsigned int j = cur_aabb_tree->getNodeParticleTag(cur_node_idx, cur_p);

                        // skip self-interaction always
                        bool excluded = (index == j);

                        if (!excluded)
                            {

                            // compute distance
                            Point pos_j = particles[j]->getOrigin();

                            Point drij = pos_j - pos_i_image;
                            double dr_sq = drij.getSquare();

                            if (dr_sq <= r_cutsq_i)
                                {
                                    neigh.push_back(j);
                                }
                            }
                        }
                    }
                }
            else
                {
                // skip ahead
                cur_node_idx += cur_aabb_tree->getNodeSkip(cur_node_idx);
                }
            } // end stackless search
        } // end loop over images

 return neigh;

}

std::vector<unsigned int> NeighborListTree::findNeighborsParticle(unsigned int index, double cutoff)
{
   // acquire particle data
   std::vector <Particle*> particles = m_Sim->getParticles();
   std::vector<unsigned int> neigh;

    // read in the current position and orientation
    const Point pos_i = particles[index]->getOrigin();
    const unsigned int type_i = particles[index]->getType();

    //const unsigned int Nmax_i = m_Nmax[type_i];
    //const unsigned int nlist_head_i = m_head_list[index];

    //unsigned int n_neigh_i = 0;
    for (unsigned int cur_pair_type=0; cur_pair_type < m_Sim->getNTypes(); ++cur_pair_type) // loop on pair types
        {
        // pass on empty types
        if (!m_num_per_type[cur_pair_type])
            continue;

        double r_cut = 0;
        if (cutoff != -1)
        {
          r_cut = cutoff;
        }
        else
        {
          // Check if this tree type should be excluded by r_cut(i,j) <= 0.0
         r_cut = m_r_cut[type_i][cur_pair_type];
        }

        if (r_cut <= 0.0)
            continue;

        // Determine the minimum r_cut_i for this particle pair
        double r_cut_i = r_cut;
        double r_cutsq_i = r_cut_i*r_cut_i;

        hpmc::detail::AABBTree *cur_aabb_tree = &m_aabb_trees[cur_pair_type];

        for (unsigned int cur_image = 0; cur_image < m_n_images; ++cur_image) // for each image vector
            {
            // make an AABB for the image of this particle
            Point pos_i_image = pos_i + m_image_list[cur_image];
            hpmc::detail::AABB aabb = hpmc::detail::AABB(pos_i_image, r_cut_i);

            // stackless traversal of the tree
            for (unsigned int cur_node_idx = 0; cur_node_idx < cur_aabb_tree->getNumNodes(); ++cur_node_idx)
                {
                if (hpmc::detail::overlap(cur_aabb_tree->getNodeAABB(cur_node_idx), aabb))
                    {
                    if (cur_aabb_tree->isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < cur_aabb_tree->getNodeNumParticles(cur_node_idx); ++cur_p)
                            {
                            // neighbor j
                            unsigned int j = cur_aabb_tree->getNodeParticleTag(cur_node_idx, cur_p);

                            // skip self-interaction always
                            bool excluded = (index == j);

                            if (!excluded)
                                {

                                // compute distance
                                Point pos_j = particles[j]->getOrigin();

                                Point drij = pos_j - pos_i_image;
                                double dr_sq = drij.getSquare();

                                if (dr_sq <= r_cutsq_i)
                                    {
                                        neigh.push_back(j);
                                    }
                                }
                            }
                        }
                    }
                else
                    {
                    // skip ahead
                    cur_node_idx += cur_aabb_tree->getNodeSkip(cur_node_idx);
                    }
                } // end stackless search
            } // end loop over images
        } // end loop over pair types

 return neigh;

}

// std::vector<unsigned int> NeighborListTree::findNeighborsMolecule(unsigned int index, double cutoff)
// {
//    // acquire particle data
//    std::vector <Molecule*> molecules = m_Sim->getMolecules();
//    std::vector<unsigned int> neigh;
//
//     // read in the current position and orientation
//     const Point pos_i = molecules[index]->getCM();
//     const unsigned int type_i = molecules[index]->getType();
//
//     //const unsigned int Nmax_i = m_Nmax[type_i];
//     //const unsigned int nlist_head_i = m_head_list[index];
//
//     //unsigned int n_neigh_i = 0;
//     for (unsigned int cur_pair_type=0; cur_pair_type < m_Sim->getMTypes(); ++cur_pair_type) // loop on pair types
//         {
//         // pass on empty types
//         // std::cout<<m_Sim->getMTypes()<<std::endl;
//         // exit(0);
//         if (!m_num_per_type[cur_pair_type])
//             continue;
//
//         double r_cut = 0;
//         if (cutoff != -1)
//         {
//           r_cut = cutoff;
//         }
//         else
//         {
//           // Check if this tree type should be excluded by r_cut(i,j) <= 0.0
//           // std::cout<<"findneighbors takes cutoff"<<std::endl;
//           r_cut = m_r_cut[type_i][cur_pair_type];
//           // std::cout<<"type_i : "<<type_i<<std::endl;
//           // std::cout<<"cur_pair_type : "<<cur_pair_type<<std::endl;
//           // std::cout<<"m_r_cut : "<<m_r_cut[type_i][cur_pair_type]<<std::endl;
//           // for(unsigned int k = 0 ; k < m_r_cut.size() ; k ++)
//           // {
//           //   for(unsigned int m = 0 ; m < m_r_cut[k].size() ; m++)
//           //   {
//           //     std::cout<<m_r_cut[k][m]<<" ";
//           //   }
//           //   std::cout<<" "<<std::endl;
//           // }
//           // exit(0);
//         }
//
//         if (r_cut <= 0.0)
//             continue;
//
//         // Determine the minimum r_cut_i for this particle pair
//         double r_cut_i = r_cut;
//         double r_cutsq_i = r_cut_i*r_cut_i;
//
//         hpmc::detail::AABBTree *cur_aabb_tree = &m_aabb_trees[cur_pair_type];
//
//         for (unsigned int cur_image = 0; cur_image < m_n_images; ++cur_image) // for each image vector
//             {
//             // make an AABB for the image of this particle
//             Point pos_i_image = pos_i + m_image_list[cur_image];
//             hpmc::detail::AABB aabb = hpmc::detail::AABB(pos_i_image, r_cut_i);
//
//             // stackless traversal of the tree
//             for (unsigned int cur_node_idx = 0; cur_node_idx < cur_aabb_tree->getNumNodes(); ++cur_node_idx)
//                 {
//                 if (hpmc::detail::overlap(cur_aabb_tree->getNodeAABB(cur_node_idx), aabb))
//                     {
//                     if (cur_aabb_tree->isNodeLeaf(cur_node_idx))
//                         {
//                         for (unsigned int cur_p = 0; cur_p < cur_aabb_tree->getNodeNumParticles(cur_node_idx); ++cur_p)
//                             {
//                             // neighbor j
//                             unsigned int j = cur_aabb_tree->getNodeParticleTag(cur_node_idx, cur_p);
//
//                             // skip self-interaction always
//                             bool excluded = (index == j);
//
//                             if (!excluded)
//                                 {
//
//                                 // compute distance
//                                 Point pos_j = molecules[j]->getCM();
//
//                                 Point drij = pos_j - pos_i_image;
//                                 double dr_sq = drij.getSquare();
//
//                                 if (dr_sq <= r_cutsq_i)
//                                     {
//                                         neigh.push_back(j);
//                                     }
//                                 }
//                             }
//                         }
//                     }
//                 else
//                     {
//                     // skip ahead
//                     cur_node_idx += cur_aabb_tree->getNodeSkip(cur_node_idx);
//                     }
//                 } // end stackless search
//             } // end loop over images
//         } // end loop over pair types
//
//  return neigh;
//
// }


void NeighborListTree::traverseTree()
    {
    // acquire particle data
    std::vector <Particle*> particles = m_Sim->getParticles();


    // Loop over all particles
    for (unsigned int i=0; i < m_Sim->getMaxN(); ++i)
        {

        // read in the current position and orientation
        const Point pos_i = particles[i]->getOrigin();
        const unsigned int type_i = particles[i]->getType();

        const unsigned int Nmax_i = m_Nmax[type_i];
        const unsigned int nlist_head_i = m_head_list[i];

        unsigned int n_neigh_i = 0;
        for (unsigned int cur_pair_type=0; cur_pair_type < m_Sim->getNTypes(); ++cur_pair_type) // loop on pair types
            {
            // pass on empty types
            if (!m_num_per_type[cur_pair_type])
                continue;

            // Check if this tree type should be excluded by r_cut(i,j) <= 0.0
            double r_cut = m_r_cut[type_i][cur_pair_type];
            if (r_cut <= 0.0)
                continue;

            // Determine the minimum r_cut_i for this particle pair
            double r_cut_i = r_cut;
            double r_cutsq_i = r_cut_i*r_cut_i;

            hpmc::detail::AABBTree *cur_aabb_tree = &m_aabb_trees[cur_pair_type];

            for (unsigned int cur_image = 0; cur_image < m_n_images; ++cur_image) // for each image vector
                {
                // make an AABB for the image of this particle
                Point pos_i_image = pos_i + m_image_list[cur_image];
                hpmc::detail::AABB aabb = hpmc::detail::AABB(pos_i_image, r_cut_i);

                // stackless traversal of the tree
                for (unsigned int cur_node_idx = 0; cur_node_idx < cur_aabb_tree->getNumNodes(); ++cur_node_idx)
                    {
                    if (hpmc::detail::overlap(cur_aabb_tree->getNodeAABB(cur_node_idx), aabb))
                        {
                        if (cur_aabb_tree->isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < cur_aabb_tree->getNodeNumParticles(cur_node_idx); ++cur_p)
                                {
                                // neighbor j
                                unsigned int j = cur_aabb_tree->getNodeParticleTag(cur_node_idx, cur_p);

                                // skip self-interaction always
                                bool excluded = (i == j);

                                if (!excluded)
                                    {

                                    // compute distance
                                    Point pos_j = particles[j]->getOrigin();

                                    Point drij = pos_j - pos_i_image;
                                    double dr_sq = drij.getSquare();

                                    if (dr_sq <= r_cutsq_i)
                                        {
                                        if (i < j)
                                            {
                                            if (n_neigh_i < Nmax_i)
                                                m_nlist[nlist_head_i + n_neigh_i] = j;

                                            else
                                                m_conditions[type_i] = std::max(m_conditions[type_i], n_neigh_i+1);

                                            ++n_neigh_i;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    else
                        {
                        // skip ahead
                        cur_node_idx += cur_aabb_tree->getNodeSkip(cur_node_idx);
                        }
                    } // end stackless search
                } // end loop over images
            } // end loop over pair types
            m_n_neigh[i] = n_neigh_i;
        } // end loop over particles


    }


// void NeighborListTree::traverseTreeMolecules()
//     {
//     // acquire particle data
//     std::vector <Molecule*> molecules = m_Sim->getMolecules();
//
//
//     // Loop over all particles
//     for (unsigned int i=0; i < m_Sim->getMaxM(); ++i)
//         {
//
//         // read in the current position and orientation
//         const Point pos_i = molecules[i]->getCM();
//         const unsigned int type_i = molecules[i]->getType();
//
//         const unsigned int Nmax_i = m_Nmax[type_i];
//         const unsigned int nlist_head_i = m_head_list[i];
//
//         unsigned int n_neigh_i = 0;
//         for (unsigned int cur_pair_type=0; cur_pair_type < m_Sim->getMTypes(); ++cur_pair_type) // loop on pair types
//             {
//             // pass on empty types
//             if (!m_num_per_type[cur_pair_type])
//                 continue;
//
//             // Check if this tree type should be excluded by r_cut(i,j) <= 0.0
//             double r_cut = m_r_cut[type_i][cur_pair_type];
//             if (r_cut <= 0.0)
//                 continue;
//
//             // Determine the minimum r_cut_i for this particle pair
//             double r_cut_i = r_cut;
//             double r_cutsq_i = r_cut_i*r_cut_i;
//
//             hpmc::detail::AABBTree *cur_aabb_tree = &m_aabb_trees[cur_pair_type];
//
//             for (unsigned int cur_image = 0; cur_image < m_n_images; ++cur_image) // for each image vector
//                 {
//                 // make an AABB for the image of this particle
//                 Point pos_i_image = pos_i + m_image_list[cur_image];
//                 hpmc::detail::AABB aabb = hpmc::detail::AABB(pos_i_image, r_cut_i);
//
//                 // stackless traversal of the tree
//                 for (unsigned int cur_node_idx = 0; cur_node_idx < cur_aabb_tree->getNumNodes(); ++cur_node_idx)
//                     {
//                     if (hpmc::detail::overlap(cur_aabb_tree->getNodeAABB(cur_node_idx), aabb))
//                         {
//                         if (cur_aabb_tree->isNodeLeaf(cur_node_idx))
//                             {
//                             for (unsigned int cur_p = 0; cur_p < cur_aabb_tree->getNodeNumParticles(cur_node_idx); ++cur_p)
//                                 {
//                                 // neighbor j
//                                 unsigned int j = cur_aabb_tree->getNodeParticleTag(cur_node_idx, cur_p);
//
//                                 // skip self-interaction always
//                                 bool excluded = (i == j);
//
//                                 if (!excluded)
//                                     {
//
//                                     // compute distance
//                                     Point pos_j = molecules[j]->getCM();
//
//                                     Point drij = pos_j - pos_i_image;
//                                     double dr_sq = drij.getSquare();
//
//                                     if (dr_sq <= r_cutsq_i)
//                                         {
//                                         if (i < j)
//                                             {
//                                             if (n_neigh_i < Nmax_i)
//                                                 m_nlist[nlist_head_i + n_neigh_i] = j;
//
//                                             else
//                                                 m_conditions[type_i] = std::max(m_conditions[type_i], n_neigh_i+1);
//
//                                             ++n_neigh_i;
//                                             }
//                                         }
//                                     }
//                                 }
//                             }
//                         }
//                     else
//                         {
//                         // skip ahead
//                         cur_node_idx += cur_aabb_tree->getNodeSkip(cur_node_idx);
//                         }
//                     } // end stackless search
//                 } // end loop over images
//             } // end loop over pair types
//             m_n_neigh[i] = n_neigh_i;
//         } // end loop over particles
//
//
//     }

void NeighborListTree::getNlist(std::vector<unsigned int> *nlist, std::vector<unsigned int> *n_neigh,std::vector<unsigned int> *head_list)
{
  *nlist = m_nlist;
  *n_neigh = m_n_neigh;
  *head_list = m_head_list;
}

void NeighborListTree::printNlist()
{

  // acquire particle data
  std::vector <Particle*> particles = m_Sim->getParticles();
  // for each particle
  for (unsigned int i = 0; i <  m_Sim->getMaxN(); i++)
      {

      // access the particle's position and type (MEM TRANSFER: 4 scalars)
      Point pi = particles[i]->getOrigin();
      unsigned int typei  = particles[i]->getType();

      // loop over all of the neighbors of this particle
      const unsigned int myHead = m_head_list[i];
      const unsigned int size = (unsigned int)m_n_neigh[i];

      for (unsigned int k = 0; k < size; k++)
          {
          // access the index of this neighbor (MEM TRANSFER: 1 scalar)
          unsigned int j = m_nlist[myHead + k];

          Point pj = particles[j]->getOrigin();
          unsigned int typej  = particles[j]->getType();

          Point dx = pi - pj;
          double dr = dx.getDistToOrigin();
          std::cout<< "neighbors "<< i << " "<< j << " types " << typei << " " << typej << " dist " << dr<<std::endl;
        }
      }
}

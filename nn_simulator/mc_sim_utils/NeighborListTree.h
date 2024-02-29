// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: mphoward



#include "AABBTree.h"
#include "Point.h"
#include "Box.h"
#include "../Sim.h"
#include <vector>
#include <memory>

#ifndef __NEIGHBORLISTTREE_H__
#define __NEIGHBORLISTTREE_H__

class Sim;


/*! @file NeighborListTree.h
    @brief Declares the NeighborListTree class
*/
/*! @brief Efficient neighbor list build on the CPU using BVH trees
 *
 * A bounding volume hierarchy (BVH) tree is a binary search tree. It is constructed from axis-aligned bounding boxes
 * (AABBs). The AABB for a node in the tree encloses all child AABBs. A leaf AABB holds multiple particles. The tree
 * is constructed in a balanced way using a heuristic to minimize AABB volume. We build one tree per particle type,
 * and use point AABBs for the particles. The neighbor list is built by traversing down the tree with an AABB
 * that encloses the pairwise cutoff for the particle. Periodic boundaries are treated by translating the query AABB
 * by all possible image vectors, many of which are trivially rejected for not intersecting the root node.
 *
 * Because one tree is built per type, complications can arise if particles change type "on the fly" during a
 * a simulation. At present, there is no signal for the types of particles changing (only the total number of types).
 * Any class directly modifying the types of particles \b must signal this change to NeighborListTree using
 * notifyParticleSort().
 *
 */
class  NeighborListTree
    {
    public:
        //! Constructor
        NeighborListTree(Sim* sim, Box* box);
        //! Constructor for molecule nlist
        NeighborListTree(Sim* sim, Box* box, double cutoff);

        //! Destructor
        virtual ~NeighborListTree();

        void setParticleCutoffs(const std::vector <std::vector<double>> in)
        {
            m_r_cut = in;
        }

        void buildNlist();
        // void buildNlistMolecules();
        void buildTrees();
        // void buildTreesMolecules();
        void updateParticleTree(unsigned int index);
        // void updateMoleculeTree(unsigned int index);
        std::vector<unsigned int> findNeighborsParticle(unsigned int index, double cutoff = -1);
        // std::vector<unsigned int> findNeighborsMolecule(unsigned int index, double cutoff = -1);
        std::vector<unsigned int> findNeighborsParticleType(unsigned int index, unsigned int type, double cutoff = -1);

        void printNlist();
        // void removePbcforPmf();
        void getNlist(std::vector<unsigned int> *nlist, std::vector<unsigned int> *n_neigh,std::vector<unsigned int> *head_list);

        //! Notification of a box size change
        void slotBoxChanged()
            {
            m_box_changed = true;
            }

        //! Notification of a max number of particle change
        void slotMaxNumChanged()
            {
            m_max_num_changed = true;
            }

        //! Notification of a particle sort - types changed
        void slotRemapParticles()
            {
            m_remap_particles = true;
            }

        //! Notification of a number of types change
        void slotNumTypesChanged()
            {
            m_type_changed = true;
            }

    private:


        const Sim* m_Sim;
        const Box* m_Box;

        bool m_box_changed;                                 //!< Flag if box size has changed
        bool m_max_num_changed;                             //!< Flag if the particle arrays need to be resized
        bool m_remap_particles;                             //!< Flag if the particles need to remapped (triggered by sort)
        bool m_type_changed;                                //!< Flag if the number of types has changed

        std::vector<unsigned int> m_nlist;         //!< Neighbor list data
        std::vector<unsigned int> m_n_neigh;       //!< Number of neighbors for each particle
        std::vector<unsigned int> m_head_list;     //!< Indexes for particles to read from the neighbor list
        std::vector<unsigned int> m_Nmax;          //!< Holds the maximum number of neighbors for each particle type
        std::vector<unsigned int> m_conditions;    //!< Holds the max number of computed particles by type for resizing

        std::vector<hpmc::detail::AABBTree> m_aabb_trees;     //!< Flat array of AABB trees of all types
        std::vector<hpmc::detail::AABB>     m_aabbs;          //!< Flat array of AABBs of all types
        std::vector<unsigned int>  m_num_per_type;   //!< Total number of particles per type
        std::vector<unsigned int>  m_type_head;      //!< Index of first particle of each type, after sorting
        std::vector<unsigned int>  m_map_pid_tree;   //!< Maps the particle id to its tag in tree for sorting

        std::vector< Point > m_image_list;      //!< List of translation vectors
        unsigned int m_n_images;                //!< The number of image vectors to check

        std::vector <std::vector <double> > m_r_cut;

        //! Driver for tree configuration
        void setupTree();
        // void setupTreeMolecules();

        //! Maps particles by local id to their id within their type trees
        /*!
         * Efficiently "sorts" particles by type into trees by generating a map from the local particle id to the
         * id within a flat array of AABBs sorted by type.
         */
        void mapParticlesByType();
        // void mapParticlesByTypeMolecule();

        //! Computes the image vectors to query for
        /*!
         * (Re-)computes the translation vectors for traversing the BVH tree. At most, there are 27 translation vectors
         * when the simulation box is 3D periodic.
         */
        void updateImageVectors();

        //! Driver to build AABB trees
        /*!
         * \note AABBTree implements its own build routine, so this is a wrapper to call this for multiple tree types.
         */
        void buildTree();
        // void buildTreeMolecules();

        //! Traverses AABB trees to compute neighbors
        /*!
         * Each AABBTree is traversed in a stackless fashion. One traversal is performed (per particle)-(per tree)-(per image).
         * The stackless traversal is a variation on left descent, where each node knows how far ahead to advance in the list
         * of nodes if there is no intersection between the current node AABB and the query AABB. Otherwise, the search advances
         * by one to the next node in the list.
         */
        void traverseTree();
        // void traverseTreeMolecules();

        void reallocate();
        void reallocateTypes();
        // void reallocateMolecules();
        // void reallocateTypesMolecules();


        /*!
        * Iterates through each particle, and calculates a running sum of the starting index for that particle
        * in the flat array of neighbors.
        *
        * \note The neighbor list is also resized when it requires more memory than is currently allocated.
        */
        void buildHeadList();
        // void buildHeadListMolecules();

        /*!
         * \param size the requested number of elements in the neighbor list
         *
         * Increases the size of the neighbor list memory using amortized resizing (growth factor: 9/8)
         * only when needed.
         * @param size for the resize
         */
        void resizeNlist(unsigned int size);

        /*!
         * \returns true if an overflow is detected for any particle type
         * \returns false if all particle types have enough memory for their neighbors
         *
         * The maximum number of neighbors per particle (rounded up to the nearest 4, min of 4) is recomputed when
         * an overflow happens.
         */
        bool checkConditions();
        // bool checkConditionsMolecules();

        void resetConditions();



    };


#endif // __NEIGHBORLISTTREE_H__

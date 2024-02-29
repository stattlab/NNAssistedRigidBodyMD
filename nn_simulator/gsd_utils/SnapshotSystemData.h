#ifndef __SNAPSHOT_SYSTEM_DATA_H__
#define __SNAPSHOT_SYSTEM_DATA_H__

#include <vector>
#include "VectorMath.h"
// using namespace linalg::aliases;

//! Structure for initializing system data
/*! A snapshot is used for multiple purposes:
 * 1. for initializing the system
 * 2. during the simulation, e.g. to dump the system state or to analyze it
 *
 * Snapshots are temporary data-structures, they are only used for passing around data.
 *
 * A SnapshotSystemData is just a super-structure that holds snapshots of other data, such
 * as particles, bonds, etc. It is used by the SystemDefinition class to initially
 * set up these data structures, and can also be obtained from an object of that class to
 * analyze the current system state.
 *
 * \ingroup data_structs
 */
template <class Real>
struct SnapshotSystemData {
    unsigned int dimensions;               //!< The dimensionality of the system
    linalg::aliases::double_3 global_box;                     //!< The dimensions of the simulation box
    std::vector< linalg::aliases::float3 > pos;             //!< positions
    std::vector< linalg::aliases::float4 > quat;             //!< quaternions
    std::vector< linalg::aliases::float4 > ang_mom;             //!< angular momentums
    std::vector< linalg::aliases::float3 > vel;             //!< velocities
    std::vector< linalg::aliases::float3 > moi;             //!< moment of inertia
    std::vector< float > mass;             //!< velocities
    std::vector<unsigned int> type;        //!< types
    std::vector<linalg::aliases::float3> image;             //!< images
    std::vector<int> bond_type;            //!< bond types
    std::vector<linalg::aliases::int2> bond_group;          //!< bond group
    std::vector<std::string> type_mapping;
    std::vector<std::string> bond_type_mapping;

    bool has_particle_data;                //!< True if snapshot contains particle data
    bool has_bond_data;                    //!< True if snapshot contains bond data
    bool has_angle_data;                   //!< True if snapshot contains angle data
    bool has_dihedral_data;                //!< True if snapshot contains dihedral data
    bool has_improper_data;                //!< True if snapshot contains improper data
    bool has_constraint_data;              //!< True if snapshot contains constraint data
    bool has_pair_data;                    //!< True if snapshot contains pair data
    bool has_integrator_data;              //!< True if snapshot contains integrator data

    //! Constructor
    SnapshotSystemData()
        {
        dimensions = 3;

        //! By default, all fields are used for initialization (even if they are empty)
        has_particle_data = true;
        has_bond_data = true;
        has_angle_data = true;
        has_dihedral_data = true;
        has_improper_data = true;
        has_constraint_data = true;
        has_pair_data = true;
        has_integrator_data = true;
        }


    };

#endif // #ifndef SNAPSHOT_SYSTEM_DATA_H__

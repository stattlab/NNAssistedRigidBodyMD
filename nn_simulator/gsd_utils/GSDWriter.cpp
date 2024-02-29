// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file GSDDumpWriter.cc
    \brief Defines the GSDDumpWriter class and related helper functions
*/

#include "GSDWriter.h"
#include "SnapshotSystemData.h"
#include <string>
#include <stdexcept>
#include <list>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

/*! Constructs the GSDDumpWriter. After construction, settings are set. No file operations are
    attempted until analyze() is called.

    \param sysdef SystemDefinition containing the ParticleData to dump
    \param fname File name to write data to
    \param group Group of particles to include in the output
    \param overwrite If false, existing files will be appended to. If true, existing files will be overwritten.
    \param truncate If true, truncate the file to 0 frames every time analyze() called, then write out one frame

    If the group does not include all particles, then topology information cannot be written to the file.
*/
GSDDumpWriter::GSDDumpWriter(const std::string &fname,
                             bool overwrite,
                             bool truncate)
    : m_fname(fname), m_overwrite(overwrite),m_truncate(truncate),    m_is_initialized(false)
    {
    //std::cout << "Constructing GSDDumpWriter: " << m_fname << " " << overwrite << " " << truncate << std::endl;
    }

void GSDDumpWriter::checkError(int retval)
    {
    // checkError prints errors and then throws exceptions for common gsd error codes
    if (retval == GSD_ERROR_IO)
        {
        std::cerr<< "dump.gsd: " << strerror(errno) << " - " << m_fname << std::endl;
        throw std::runtime_error("Error writing GSD file");
        }
    else if (retval == GSD_ERROR_INVALID_ARGUMENT)
        {
        std::cerr<< "dump.gsd: Invalid argument" " - " << m_fname << std::endl;
        throw std::runtime_error("Error writing GSD file");
        }
    else if (retval == GSD_ERROR_NOT_A_GSD_FILE)
        {
        std::cerr<< "dump.gsd: Not a GSD file" " - " << m_fname << std::endl;
        throw std::runtime_error("Error writing GSD file");
        }
    else if (retval == GSD_ERROR_INVALID_GSD_FILE_VERSION)
        {
        std::cerr<< "dump.gsd: Invalid GSD file version" " - " << m_fname << std::endl;
        throw std::runtime_error("Error writing GSD file");
        }
    else if (retval == GSD_ERROR_FILE_CORRUPT)
        {
        std::cerr<< "dump.gsd: File corrupt" " - " << m_fname << std::endl;
        throw std::runtime_error("Error writing GSD file");
        }
    else if (retval == GSD_ERROR_MEMORY_ALLOCATION_FAILED)
        {
        std::cerr<< "dump.gsd: Memory allocation failed" " - " << m_fname << std::endl;
        throw std::runtime_error("Error writing GSD file");
        }
    else if (retval == GSD_ERROR_NAMELIST_FULL)
        {
        std::cerr<< "dump.gsd: Namelist full" " - " << m_fname << std::endl;
        throw std::runtime_error("Error writing GSD file");
        }
    else if (retval == GSD_ERROR_FILE_MUST_BE_WRITABLE)
        {
        std::cerr<< "dump.gsd: File must be writeable" " - " << m_fname << std::endl;
        throw std::runtime_error("Error writing GSD file");
        }
    else if (retval == GSD_ERROR_FILE_MUST_BE_READABLE)
        {
        std::cerr<< "dump.gsd: File must be readable" " - " << m_fname << std::endl;
        throw std::runtime_error("Error writing GSD file");
        }
    else if (retval != GSD_SUCCESS)
        {
        std::cerr<< "dump.gsd: " << "Unknown error " << retval << " writing: "
                                  << m_fname << std::endl;
        throw std::runtime_error("Error writing GSD file");
        }
    }


bool checkExistence(std::string filename)
{
    std::ifstream f;
    f.open(filename);

    return f.is_open();
}

//! Initializes the output file for writing
void GSDDumpWriter::initFileIO()
    {
    int retval = 0;

    // create the file if it does not exist - mac specific
    if (m_overwrite || ! checkExistence(m_fname))
        {

        std::cout << "dump.gsd: create gsd file " << m_fname << std::endl;
        retval = gsd_create(m_fname.c_str(),
                            "HOOMD-blue 2.6.5",
                            "hoomd",
                            gsd_make_version(1,3));
        checkError(retval);
        }

    // populate the non-default map
    populateNonDefault();

    // open the file in append mode
    //std::cout << "dump.gsd: open gsd file " << m_fname << std::endl;
    retval = gsd_open(&m_handle, m_fname.c_str(), GSD_OPEN_APPEND);
    checkError(retval);

    // validate schema
    if (std::string(m_handle.header.schema) != std::string("hoomd"))
        {
        std::cerr<< "dump.gsd: " << "Invalid schema in " << m_fname << std::endl;
        throw std::runtime_error("Error opening GSD file");
        }
    if (m_handle.header.schema_version >= gsd_make_version(2,0))
        {
        std::cerr<< "dump.gsd: " << "Invalid schema version in " << m_fname << std::endl;
        throw std::runtime_error("Error opening GSD file");
        }

    m_is_initialized = true;
    }

GSDDumpWriter::~GSDDumpWriter()
    {
    //std::cout << "Destroying GSDDumpWriter" << std::endl;

    if (m_is_initialized)
        {
        //std::cout << "dump.gsd: close gsd file " << m_fname << std::endl;
        gsd_close(&m_handle);
        }
    }

/*! \param timestep Current time step of the simulation

    The first call to analyze() will create or overwrite the file and write out the current system configuration
    as frame 0. Subsequent calls will append frames to the file, or keep overwriting frame 0 if m_truncate is true.
*/
void GSDDumpWriter::analyze(unsigned int timestep, const SnapshotSystemData<float>& snapshot)
    {
    int retval;
    bool root=true;

    // take particle data snapshot
    //std::cout << "dump.gsd: taking particle data snapshot" << std::endl;
    //SnapshotSystemData<float> snapshot;
    //const std::map<unsigned int, unsigned int>& map = m_pdata->takeSnapshot<float>(snapshot);


    // open the file if it is not yet opened
    if (! m_is_initialized)
        initFileIO();

    // truncate the file if requested
    if (m_truncate)
        {
        //std::cout << "dump.gsd: truncating file" << std::endl;
        retval = gsd_truncate(&m_handle);
        checkError(retval);
        }

    uint64_t nframes = 0;
    nframes = gsd_get_nframes(&m_handle);
    //std::cout << "dump.gsd: " << m_fname << " has " << nframes << " frames" << std::endl;


    // write out the frame header on all frames
    writeFrameHeader(timestep,snapshot);

    // only write out data chunk categories if requested, or if on frame 0
    if (m_write_attribute || nframes == 0)
    {
      writeAttributes(snapshot);
    }
    if (m_write_property || nframes == 0)
    {
      writeProperties(snapshot);
    }
    if (m_write_momentum || nframes == 0)
    {
      writeMomenta(snapshot);
    }
    if (m_write_topology || nframes == 0)

        writeTopology(snapshot);



    //writeUser(timestep, root);


    //std::cout << "dump.gsd: ending frame" << std::endl;
    retval = gsd_end_frame(&m_handle);
    checkError(retval);

    }


void GSDDumpWriter::writeTypeMapping(std::string chunk, std::vector< std::string > type_mapping)
    {
    int max_len = 0;
    for (unsigned int i = 0; i < type_mapping.size(); i++)
        {
        max_len = std::max(max_len, (int)type_mapping[i].size());
        }
    max_len += 1;  // for null

        {
        //std::cout << "dump.gsd: writing " << chunk << std::endl;
        std::vector<char> types(max_len * type_mapping.size());
        for (unsigned int i = 0; i < type_mapping.size(); i++)
            strncpy(&types[max_len*i], type_mapping[i].c_str(), max_len);
        int retval = gsd_write_chunk(&m_handle, chunk.c_str(), GSD_TYPE_UINT8, type_mapping.size(), max_len, 0, (void *)&types[0]);
        checkError(retval);
        }

    }

/*! \param timestep

    Write the data chunks configuration/step, configuration/box, and particles/N. If this is frame 0, also write
    configuration/dimensions.

    N is not strictly necessary for constant N data, but is always written in case the user fails to select
    dynamic attributes with a variable N file.
*/
void GSDDumpWriter::writeFrameHeader(unsigned int timestep,const SnapshotSystemData<float>& snapshot)
    {
    int retval;
    //std::cout << "dump.gsd: writing configuration/step" << std::endl;
    uint64_t step = timestep;
    retval = gsd_write_chunk(&m_handle, "configuration/step", GSD_TYPE_UINT64, 1, 1, 0, (void *)&step);
    checkError(retval);

    if (gsd_get_nframes(&m_handle) == 0)
        {
        //std::cout << "dump.gsd: writing configuration/dimensions" << std::endl;
        uint8_t dimensions = snapshot.dimensions;
        retval = gsd_write_chunk(&m_handle, "configuration/dimensions", GSD_TYPE_UINT8, 1, 1, 0, (void *)&dimensions);
        checkError(retval);
        }

    //std::cout << "dump.gsd: writing configuration/box" << std::endl;
    linalg::aliases::double_3 box = snapshot.global_box;
    float box_a[6];
    box_a[0] = box[0];
    box_a[1] = box[1];
    box_a[2] = box[2];
    box_a[3] = 0;
    box_a[4] = 0;
    box_a[5] = 0;
    retval = gsd_write_chunk(&m_handle, "configuration/box", GSD_TYPE_FLOAT, 6, 1, 0, (void *)box_a);
    checkError(retval);

    //std::cout << "dump.gsd: writing particles/N" << std::endl;
    uint32_t N = snapshot.pos.size();
    retval = gsd_write_chunk(&m_handle, "particles/N", GSD_TYPE_UINT32, 1, 1, 0, (void *)&N);
    checkError(retval);
    }

/*! \param snapshot particle data snapshot to write out to the file

    Writes the data chunks types, typeid, mass, charge, diameter, body, moment_inertia in particles/.
*/
void GSDDumpWriter::writeAttributes(const SnapshotSystemData<float>& snapshot)
    {
    uint32_t N = snapshot.pos.size();
    int retval;
    uint64_t nframes = gsd_get_nframes(&m_handle);

    writeTypeMapping("particles/types", snapshot.type_mapping);

        {
        std::vector<uint32_t> type(N);
        type.reserve(1); //! make sure we allocate
        bool all_default = true;

        for (unsigned int group_idx = 0; group_idx < N; group_idx++)
            {

            if (snapshot.type[group_idx] != 0)
                all_default = false;

            type[group_idx] = uint32_t(snapshot.type[group_idx]);
            }

        if (!all_default || (nframes > 0 && m_nondefault["particles/typeid"]))
            {
            //std::cout << "dump.gsd: writing particles/typeid" << std::endl;
            retval = gsd_write_chunk(&m_handle, "particles/typeid", GSD_TYPE_UINT32, N, 1, 0, (void *)&type[0]);
            checkError(retval);
            if (nframes == 0)
                m_nondefault["particles/typeid"] = true;
            }
        }

        {
        std::vector<float> data(uint64_t(N)*1);
        data.reserve(1); //! make sure we allocate
        bool all_default = true;

        for (unsigned int group_idx = 0; group_idx < N; group_idx++)
            {

            if (snapshot.mass[group_idx] != 0)
                all_default = false;

            data[group_idx] = float(snapshot.mass[group_idx]);
            }

        if (!all_default || (nframes > 0 && m_nondefault["particles/mass"]))
            {
            //std::cout << "dump.gsd: writing particles/typeid" << std::endl;
            retval = gsd_write_chunk(&m_handle, "particles/mass", GSD_TYPE_FLOAT, N, 1, 0, (void *)&data[0]);
            checkError(retval);
            if (nframes == 0)
                m_nondefault["particles/mass"] = true;
            }
        }

        {
        std::vector<float> data(uint64_t(N)*3);
        data.reserve(1); //! make sure we allocate
        bool all_default = true;

        for (unsigned int group_idx = 0; group_idx < N; group_idx++)
            {

            if (snapshot.moi[group_idx][0] != float(0.0) ||
                snapshot.moi[group_idx][1] != float(0.0) ||
                snapshot.moi[group_idx][2] != float(0.0))
                {
                all_default = false;
                }

            data[group_idx*3+0] = float(snapshot.moi[group_idx][0]);
            data[group_idx*3+1] = float(snapshot.moi[group_idx][1]);
            data[group_idx*3+2] = float(snapshot.moi[group_idx][2]);
            }

        if (!all_default || (nframes > 0 && m_nondefault["particles/moment_inertia"]))
            {
            //std::cout << "dump.gsd: writing particles/velocity" << std::endl;
            retval = gsd_write_chunk(&m_handle, "particles/moment_inertia", GSD_TYPE_FLOAT, N, 3, 0, (void *)&data[0]);
            checkError(retval);
            if (nframes == 0)
                m_nondefault["particles/moment_inertia"] = true;
            }
        }


    }

/*! \param snapshot particle data snapshot to write out to the file

    Writes the data chunks position and orientation in particles/.
*/
void GSDDumpWriter::writeProperties(const SnapshotSystemData<float>& snapshot)
    {
    uint32_t N = snapshot.pos.size();
    int retval;
    uint64_t nframes = gsd_get_nframes(&m_handle);

        {
        std::vector<float> data(uint64_t(N)*3);
        data.reserve(1); //! make sure we allocate

        for (unsigned int group_idx = 0; group_idx < N; group_idx++)
            {
            data[group_idx*3+0] = float(snapshot.pos[group_idx][0]);
            data[group_idx*3+1] = float(snapshot.pos[group_idx][1]);
            data[group_idx*3+2] = float(snapshot.pos[group_idx][2]);
            }

        //std::cout << "dump.gsd: writing particles/position" << std::endl;
        retval = gsd_write_chunk(&m_handle, "particles/position", GSD_TYPE_FLOAT, N, 3, 0, (void *)&data[0]);
        checkError(retval);
        }

        // ORIENTATION //
        {
        std::vector<float> data(uint64_t(N)*4);
        data.reserve(1); //! make sure we allocate

        for (unsigned int group_idx = 0; group_idx < N; group_idx++)
            {
            data[group_idx*4+0] = float(snapshot.quat[group_idx][0]);
            data[group_idx*4+1] = float(snapshot.quat[group_idx][1]);
            data[group_idx*4+2] = float(snapshot.quat[group_idx][2]);
            data[group_idx*4+3] = float(snapshot.quat[group_idx][3]);
            }

        //std::cout << "dump.gsd: writing particles/position" << std::endl;
        retval = gsd_write_chunk(&m_handle, "particles/orientation", GSD_TYPE_FLOAT, N, 4, 0, (void *)&data[0]);
        checkError(retval);
        }


    }

/*! \param snapshot particle data snapshot to write out to the file

    Writes the data chunks velocity, angmom, and image in particles/.
*/
void GSDDumpWriter::writeMomenta(const SnapshotSystemData<float>& snapshot)
    {
    uint32_t N = snapshot.pos.size();
    int retval;
    uint64_t nframes = gsd_get_nframes(&m_handle);

        {
        std::vector<float> data(uint64_t(N)*3);
        data.reserve(1); //! make sure we allocate
        bool all_default = true;

        for (unsigned int group_idx = 0; group_idx < N; group_idx++)
            {

            if (snapshot.vel[group_idx][0] != float(0.0) ||
                snapshot.vel[group_idx][1] != float(0.0) ||
                snapshot.vel[group_idx][2] != float(0.0))
                {
                all_default = false;
                }

            data[group_idx*3+0] = float(snapshot.vel[group_idx][0]);
            data[group_idx*3+1] = float(snapshot.vel[group_idx][1]);
            data[group_idx*3+2] = float(snapshot.vel[group_idx][2]);
            }

        if (!all_default || (nframes > 0 && m_nondefault["particles/velocity"]))
            {
            //std::cout << "dump.gsd: writing particles/velocity" << std::endl;
            retval = gsd_write_chunk(&m_handle, "particles/velocity", GSD_TYPE_FLOAT, N, 3, 0, (void *)&data[0]);
            checkError(retval);
            if (nframes == 0)
                m_nondefault["particles/velocity"] = true;
            }
        }

        // ANGULAR MOMENTUM //
        {
        std::vector<float> data(uint64_t(N)*4);
        data.reserve(1); //! make sure we allocate
        bool all_default = true;

        for (unsigned int group_idx = 0; group_idx < N; group_idx++)
            {

            if (snapshot.ang_mom[group_idx][0] != float(0.0) ||
                snapshot.ang_mom[group_idx][1] != float(0.0) ||
                snapshot.ang_mom[group_idx][2] != float(0.0) ||
                snapshot.ang_mom[group_idx][3] != float(0.0))

                {
                all_default = false;
                }

            data[group_idx*3+0] = float(snapshot.ang_mom[group_idx][0]);
            data[group_idx*3+1] = float(snapshot.ang_mom[group_idx][1]);
            data[group_idx*3+2] = float(snapshot.ang_mom[group_idx][2]);
            data[group_idx*3+3] = float(snapshot.ang_mom[group_idx][3]);
            }

        if (!all_default || (nframes > 0 && m_nondefault["particles/angmom"]))
            {
            //std::cout << "dump.gsd: writing particles/velocity" << std::endl;
            retval = gsd_write_chunk(&m_handle, "particles/angmom", GSD_TYPE_FLOAT, N, 4, 0, (void *)&data[0]);
            checkError(retval);
            if (nframes == 0)
                m_nondefault["particles/angmom"] = true;
            }
        }


        // {
        // std::vector<int32_t> data(uint64_t(N)*3);
        // data.reserve(1); //! make sure we allocate
        // bool all_default = true;
        //
        // for (unsigned int group_idx = 0; group_idx < N; group_idx++)
        //     {
        //
        //     if (snapshot.image[group_idx][0] != 0 ||
        //         snapshot.image[group_idx][1] != 0 ||
        //         snapshot.image[group_idx][2] != 0)
        //         {
        //         all_default = false;
        //         }
        //
        //     data[group_idx*3+0] = float(snapshot.image[group_idx][0]);
        //     data[group_idx*3+1] = float(snapshot.image[group_idx][1]);
        //     data[group_idx*3+2] = float(snapshot.image[group_idx][2]);
        //     }
        //
        // if (!all_default || (nframes > 0 && m_nondefault["particles/image"]))
        //     {
        //     //std::cout << "dump.gsd: writing particles/image" << std::endl;
        //     retval = gsd_write_chunk(&m_handle, "particles/image", GSD_TYPE_INT32, N, 3, 0, (void *)&data[0]);
        //     checkError(retval);
        //     if (nframes == 0)
        //         m_nondefault["particles/image"] = true;
        //     }
        // }
    }

/*! \param bond Bond data snapshot
    \param angle Angle data snapshot
    \param dihedral Dihedral data snapshot
    \param improper Improper data snapshot
    \param constraint Constraint data snapshot
    \param pair Special pair data snapshot

    Write out all the snapshot data to the GSD file
*/
void GSDDumpWriter::writeTopology(const SnapshotSystemData<float>& snapshot)
    {
    if (snapshot.bond_group.size() > 0)
        {
        //std::cout << "dump.gsd: writing bonds/N" << std::endl;
        uint32_t N = snapshot.bond_type.size();
        int retval = gsd_write_chunk(&m_handle, "bonds/N", GSD_TYPE_UINT32, 1, 1, 0, (void *)&N);
        checkError(retval);

        writeTypeMapping("bonds/types", snapshot.bond_type_mapping);

        //std::cout << "dump.gsd: writing bonds/typeid" << std::endl;
        retval = gsd_write_chunk(&m_handle, "bonds/typeid", GSD_TYPE_UINT32, N, 1, 0, (void *)&snapshot.bond_type[0]);
        checkError(retval);

        //std::cout << "dump.gsd: writing bonds/group" << std::endl;
        retval = gsd_write_chunk(&m_handle, "bonds/group", GSD_TYPE_UINT32, N, 2, 0, (void *)&snapshot.bond_group[0]);
        checkError(retval);
        }

    }


/*! Populate the m_nondefault map.
    Set entries to true when they exist in frame 0 of the file, otherwise, set them to false.
*/
void GSDDumpWriter::populateNonDefault()
    {
    int retval;

    // open the file in read only mode
    //std::cout << "dump.gsd: check frame 0 in gsd file " << m_fname << std::endl;
    retval = gsd_open(&m_handle, m_fname.c_str(), GSD_OPEN_READONLY);
    if (retval == -1)
        {
        std::cerr<< "dump.gsd: " << strerror(errno) << " - " << m_fname << std::endl;
        throw std::runtime_error("Error opening GSD file");
        }
    else if (retval == -2)
        {
        std::cerr<< "dump.gsd: " << m_fname << " is not a valid GSD file" << std::endl;
        throw std::runtime_error("Error opening GSD file");
        }
    else if (retval == -3)
        {
        std::cerr<< "dump.gsd: " << "Invalid GSD file version in " << m_fname << std::endl;
        throw std::runtime_error("Error opening GSD file");
        }
    else if (retval == -4)
        {
        std::cerr<< "dump.gsd: " << "Corrupt GSD file: " << m_fname << std::endl;
        throw std::runtime_error("Error opening GSD file");
        }
    else if (retval == -5)
        {
        std::cerr<< "dump.gsd: " << "Out of memory opening: " << m_fname << std::endl;
        throw std::runtime_error("Error opening GSD file");
        }
    else if (retval != 0)
        {
        std::cerr<< "dump.gsd: " << "Unknown error opening: " << m_fname << std::endl;
        throw std::runtime_error("Error opening GSD file");
        }

    // validate schema
    if (std::string(m_handle.header.schema) != std::string("hoomd"))
        {
        std::cerr<< "dump.gsd: " << "Invalid schema in " << m_fname << std::endl;
        throw std::runtime_error("Error opening GSD file");
        }
    if (m_handle.header.schema_version >= gsd_make_version(2,0))
        {
        std::cerr<< "dump.gsd: " << "Invalid schema version in " << m_fname << std::endl;
        throw std::runtime_error("Error opening GSD file");
        }

    std::list<std::string> chunks {"particles/typeid",
                                   "particles/mass",
                                   "particles/charge",
                                   "particles/diameter",
                                   "particles/body",
                                   "particles/moment_inertia",
                                   "particles/orientation",
                                   "particles/velocity",
                                   "particles/angmom",
                                   "particles/image"};

    for (auto const& chunk : chunks)
        {
        const gsd_index_entry *entry = gsd_find_chunk(&m_handle, 0, chunk.c_str());
        m_nondefault[chunk] = (entry != nullptr);
        }

    // close the file
    gsd_close(&m_handle);
    }

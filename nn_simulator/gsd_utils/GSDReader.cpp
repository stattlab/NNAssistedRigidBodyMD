// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file GSDReader.ccp
    \brief Defines the GSDReader class
*/

#include "GSDReader.h"
#include "SnapshotSystemData.h"
#include "gsd.h"
#include <string>
#include <vector>
#include <memory>
#include <string.h>
#include <iostream>
#include <stdexcept>

/*! \param exec_conf The execution configuration
    \param name File name to read
    \param frame Frame index to read from the file
    \param from_end Count frames back from the end of the file

    The GSDReader constructor opens the GSD file, initializes an empty snapshot, and reads the file into
    memory (on the root rank).
*/
GSDReader::GSDReader(const std::string &name,
                     const uint64_t frame,
                     bool from_end)
    :  m_timestep(0), m_name(name), m_frame(frame), m_nframes(0)
    {
    m_snapshot = std::shared_ptr< SnapshotSystemData<float> >(new SnapshotSystemData<float>);

    // open the GSD file in read mode
    int retval = gsd_open(&m_handle, name.c_str(), GSD_OPEN_READONLY);
    checkError(retval);

    // validate schema
    if (std::string(m_handle.header.schema) != std::string("hoomd"))
        {
        std::cerr << "data.gsd_snapshot: " << "Invalid schema in " << name << std::endl;
        throw std::runtime_error("Error opening GSD file");
        }
    if (m_handle.header.schema_version >= gsd_make_version(2,0))
        {
        std::cerr << "data.gsd_snapshot: " << "Invalid schema version in " << name << std::endl;
        throw std::runtime_error("Error opening GSD file");
        }

    // set frame from the end of the file if requested
    uint64_t nframes = gsd_get_nframes(&m_handle);
    m_nframes = nframes;

    if (from_end && frame <= nframes)
        m_frame = nframes - frame;

    // validate number of frames
    if (m_frame >= nframes)
        {
        std::cerr << "data.gsd_snapshot: " << "Cannot read frame " << m_frame << " " << name << " only has " << gsd_get_nframes(&m_handle) << " frames" << std::endl;
        throw std::runtime_error("Error opening GSD file");
        }

    readHeader();
    readParticles();
    readTopology();
    }

GSDReader::~GSDReader()
    {

    gsd_close(&m_handle);
    }

/*! \param data Pointer to data to read into
    \param frame Frame index to read from
    \param name Name of the data chunk
    \param expected_size Expected size of the data chunk in bytes.
    \param cur_n N in the current frame.

    Attempts to read the data chunk of the given name at the given frame. If it is not present at this
    frame, attempt to read from frame 0. If it is also not present at frame 0, return false.
    If the found data chunk is not the expected size, throw an exception.

    Per the GSD spec, keep the default when the frame 0 N does not match the current N.

    Return true if data is actually read from the file.
*/
bool GSDReader::readChunk(void *data, uint64_t frame, const char *name, size_t expected_size, unsigned int cur_n)
    {
    const struct gsd_index_entry* entry = gsd_find_chunk(&m_handle, frame, name);
    if (entry == NULL && frame != 0)
        entry = gsd_find_chunk(&m_handle, 0, name);

    if (entry == NULL || (cur_n != 0 && entry->N != cur_n))
        {
        std::cerr << "WARNING: data.gsd_snapshot: chunk not found " << name << std::endl;
        return false;
        }
    else
        {
        size_t actual_size = entry->N * entry->M * gsd_sizeof_type((enum gsd_type)entry->type);
        if (actual_size != expected_size)
            {
            std::cerr<< "data.gsd_snapshot: " << "Expecting " << expected_size << " bytes in " << name << " but found " << actual_size << std::endl;
            throw std::runtime_error("Error reading GSD file");
            }
        int retval = gsd_read_chunk(&m_handle, data, entry);
        checkError(retval);

        return true;
        }
    }

/*! Read the same data chunks written by GSDDumpWriter::writeFrameHeader
*/
void GSDReader::readHeader()
    {
    readChunk(&m_timestep, m_frame, "configuration/step", 8);
    uint8_t dim = 3;
    readChunk(&dim, m_frame, "configuration/dimensions", 1);
    m_snapshot->dimensions = dim;

    float box[6] = {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    readChunk(&box, m_frame, "configuration/box", 6*4);
    m_snapshot->global_box = {box[0],box[1],box[2]};
    //m_snapshot->global_box.setTiltFactors(box[3], box[4], box[5]);

    unsigned int N = 0;
    readChunk(&N, m_frame, "particles/N", 4);
    if (N == 0)
        {
        std::cerr<< "data.gsd_snapshot: " << "cannot read a file with 0 particles" << std::endl;
        throw std::runtime_error("Error reading GSD file");
        }
    m_snapshot->vel.resize(N);
    m_snapshot->pos.resize(N);
    m_snapshot->quat.resize(N);
    m_snapshot->ang_mom.resize(N);
    m_snapshot->mass.resize(N);
    m_snapshot->moi.resize(N);
    m_snapshot->type.resize(N);
    m_snapshot->image.resize(N);
    }

/*! \param frame Frame index to read from
\param name Name of the data chunk

Attempts to read the data chunk of the given name at the given frame. If it is not present at this
frame, attempt to read from frame 0. If it is also not present at frame 0, return an empty list.

If the data chunk is found in the file, return a vector of string type names.
*/
std::vector<std::string> GSDReader::readTypes(uint64_t frame, const char *name)
{

std::vector<std::string> type_mapping;

// set the default particle type mapping per the GSD HOOMD Schema
if (std::string(name) == "particles/types")
    type_mapping.push_back("A");

const struct gsd_index_entry* entry = gsd_find_chunk(&m_handle, frame, name);
if (entry == NULL && frame != 0)
    entry = gsd_find_chunk(&m_handle, 0, name);

if (entry == NULL)
    return type_mapping;
else
    {
    size_t actual_size = entry->N * entry->M * gsd_sizeof_type((enum gsd_type)entry->type);
    std::vector<char> data(actual_size);
    int retval = gsd_read_chunk(&m_handle, &data[0], entry);
    checkError(retval);

    type_mapping.clear();
    for (unsigned int i = 0; i < entry->N; i++)
        {
        size_t l = strnlen(&data[i*entry->M], entry->M);
        type_mapping.push_back(std::string(&data[i*entry->M], l));
        }

    return type_mapping;
    }
}

/*! Read the same data chunks for particles
*/
void GSDReader::readParticles()
    {
      // the strings for each particle data type : https://gsd.readthedocs.io/en/v3.0.1/schema-hoomd.html

    uint64_t N = m_snapshot->pos.size();

    m_snapshot->type_mapping = readTypes(m_frame, "particles/types");
    readChunk(&m_snapshot->pos[0], m_frame, "particles/position", N*12, N);
    readChunk(&m_snapshot->quat[0], m_frame, "particles/orientation", N*16, N);
    readChunk(&m_snapshot->vel[0], m_frame, "particles/velocity", N*12, N);
    readChunk(&m_snapshot->ang_mom[0], m_frame, "particles/angmom", N*16, N);
    readChunk(&m_snapshot->type[0], m_frame, "particles/typeid", N*4, N);
    readChunk(&m_snapshot->image[0], m_frame, "particles/image", N*12, N);
    readChunk(&m_snapshot->mass[0], m_frame, "particles/mass", N*4, N);
    readChunk(&m_snapshot->moi[0], m_frame, "particles/moment_inertia", N*12, N);

    }

/*! Read the same data chunks for topology
*/
void GSDReader::readTopology()
    {

    uint64_t N = 0;
    readChunk(&N, m_frame, "bonds/N", 4);
    if (N > 0)
        {
        m_snapshot->bond_type.resize(N);
        m_snapshot->bond_group.resize(N);
        m_snapshot->bond_type_mapping = readTypes(m_frame, "bonds/types");
        readChunk(&m_snapshot->bond_type[0], m_frame, "bonds/typeid", N*4, N);
        readChunk(&m_snapshot->bond_group[0], m_frame, "bonds/group", N*8, N);
        }

    }


void GSDReader::checkError(int retval)
    {
    // checkError prints errors and then throws exceptions for common gsd error codes
    if (retval == GSD_ERROR_IO)
        {
        std::cerr << "dump.gsd: " << strerror(errno) << " - " << m_name << std::endl;
        throw std::runtime_error("Error reading GSD file");
        }
    else if (retval == GSD_ERROR_INVALID_ARGUMENT)
        {
        std::cerr << "dump.gsd: Invalid argument" " - " << m_name << std::endl;
        //throw runtime_error("Error reading GSD file");
        }
    else if (retval == GSD_ERROR_NOT_A_GSD_FILE)
        {
        std::cerr << "dump.gsd: Not a GSD file" " - " << m_name << std::endl;
        throw std::runtime_error("Error reading GSD file");
        }
    else if (retval == GSD_ERROR_INVALID_GSD_FILE_VERSION)
        {
        std::cerr<< "dump.gsd: Invalid GSD file version" " - " << m_name << std::endl;
        throw std::runtime_error("Error reading GSD file");
        }
    else if (retval == GSD_ERROR_FILE_CORRUPT)
        {
        std::cerr << "dump.gsd: File corrupt" " - " << m_name << std::endl;
        throw std::runtime_error("Error reading GSD file");
        }
    else if (retval == GSD_ERROR_MEMORY_ALLOCATION_FAILED)
        {
        std::cerr << "dump.gsd: Memory allocation failed" " - " << m_name << std::endl;
        throw std::runtime_error("Error reading GSD file");
        }
    else if (retval == GSD_ERROR_NAMELIST_FULL)
        {
        std::cerr << "dump.gsd: Namelist full" " - " << m_name << std::endl;
        throw std::runtime_error("Error reading GSD file");
        }
    else if (retval == GSD_ERROR_FILE_MUST_BE_WRITABLE)
        {
        std::cerr << "dump.gsd: File must be writeable" " - " << m_name << std::endl;
        throw std::runtime_error("Error reading GSD file");
        }
    else if (retval == GSD_ERROR_FILE_MUST_BE_READABLE)
        {
        std::cerr << "dump.gsd: File must be readable" " - " << m_name << std::endl;
        throw std::runtime_error("Error reading GSD file");
        }
    else if (retval != GSD_SUCCESS)
        {
        std::cerr<< "dump.gsd: " << "Unknown error " << retval << " reading: "
                                  << m_name << std::endl;
        throw std::runtime_error("Error reading GSD file");
        }
    }

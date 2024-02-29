#ifndef __GSD_INITIALIZER_H__
#define __GSD_INITIALIZER_H__

#include <string>
#include <vector>
#include <memory>
#include <string.h>
#include <iostream>
#include <stdexcept>

#include "gsd.h"
// #include "VectorMath.h"
// using namespace linalg::aliases;

//! Forward declarations
template <class Real> struct SnapshotSystemData;

//! Reads a GSD input file
/*! Read an input GSD file and generate a system snapshot. GSDReader can read any frame from a GSD
    file into the snapshot. For information on the GSD specification, see http://gsd.readthedocs.io/

    \ingroup data_structs
*/
class GSDReader
    {
    public:
        //! Loads in the file and parses the data
        GSDReader(const std::string &name,
                  const uint64_t frame,
                  bool from_end);

        //! Destructor
        ~GSDReader();

        //! Returns the timestep of the simulation
        uint64_t getTimeStep() const
            {
            uint64_t timestep = m_timestep;

            return timestep;
            }

        //! initializes a snapshot with the particle data
        std::shared_ptr< SnapshotSystemData<float> > getSnapshot() const
            {
            return m_snapshot;
            }

        //! initializes a snapshot with the particle data
        uint64_t getFrame() const
            {
            return m_frame;
            }
        //! initializes a snapshot with the particle data
        uint64_t getNFrame() const
            {
            return m_nframes;
            }
        //! Helper function to read a quantity from the file
        bool readChunk(void *data, uint64_t frame, const char *name, size_t expected_size, unsigned int cur_n=0);

        //! clears the snapshot object
        void clearSnapshot()
            {
            m_snapshot.reset();
            }

        //! get handle
        gsd_handle getHandle(void) const
            {
            return m_handle;
            }


    private:
        uint64_t m_timestep;                                         //!< Timestep at the selected frame
        std::string m_name;                                          //!< Cached file name
        uint64_t m_frame;                                            //!< Cached frame
        uint64_t m_nframes;                                          //!< total number of frames
        std::shared_ptr< SnapshotSystemData<float> > m_snapshot;     //!< The snapshot to read
        gsd_handle m_handle;                                         //!< Handle to the file

        //! Helper function to read a type list from the file
        std::vector<std::string> readTypes(uint64_t frame, const char *name);

        // helper functions to read sections of the file
        void readHeader();
        void readParticles();
        void readTopology();

        /// Check and raise an exception if an error occurs
        void checkError(int retval);
    };

#endif

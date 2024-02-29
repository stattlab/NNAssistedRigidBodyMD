#ifndef MPIERRORHANDLING_H_
#define MPIERRORHANDLING_H_

#define OMPI_SKIP_MPICXX    1
// #include <mpi.h>
#include <string>

#define TAG_TERMINATE                99
#define TIMECHECKTERMINATEREQUEST    10
#define TIMEMAXWAITBEFOREEXIT        11

/**
 * @brief Error and Warning handler for MPI programs
 */

/** This static class manages the behavior of the program if an error occurs.
 * The main idea is to inform all other ranks so that the whole MPI process group can end properly.
 * The error() and warning() from MonteCarloUtils have been adapted to this class.
 *
 * To use this class, each rank has to call listenForTerminateRequest() once at the beginning
 * and later on a regularly basis checkTerminateRequest() to see whether another rank has sent
 * a request. To send a request, a rank can call sendTerminateRequest().
 * Currently, error() calls sendTerminateRequest() automatically.
 *
 * Important remark on \#include <mpi.h>
 * As mpi.h is already included here, all classes including "MPIErrorHandling.h" automatically
 * get mpi.h. This is good because of the naming conflict of mpi.h and stdio.h resulting in one
 * of the following errors when compiling:
 * - \#error "SEEK_SET is #defined but must not be for the C++ binding of MPI"
 * - \#error "SEEK_CUR is #defined but must not be for the C++ binding of MPI"
 * - \#error "SEEK_END is #defined but must not be for the C++ binding of MPI"
 * The solution is to include mpi.h before stdio.h and iostream.
 * Therefore all classes should include their own header file first, where the header file
 * should first include this header file.
 *
 * More information:
 * There is a name conflict between stdio.h and the MPI C++ binding with
 * respect to the names SEEK_SET, SEEK_CUR, and SEEK_END. MPI wants
 * these in the MPI namespace, but stdio.h will \#define these to integer
 * values. \#undef'ing these can cause obscure problems with other include
 * files (such as iostream), so MPICH2 instead uses #error to indicate a
 * fatal error. Users can either \#undef the names before including
 * mpi.h or include mpi.h *before* stdio.h or iostream. Alternately,
 * passing the flag -DMPICH_IGNORE_CXX_SEEK causes MPICH2 to ignore this
 * naming conflict.
 *
 * @author Fabian Schmitz
 * @date 22/03/2013
 * @version 0.2
 * @brief Error and Warning handler for MPI programs
 */
class MPIErrorHandling {
public:

   /** Prints the given message to std out and calls sendTerminateRequest() with status 8.
    * @param Message the error message
    * @see MPIErrorHandling::error(const char* Message) */
   static void error(const char *Message);

   /** Prints the given message to std out and calls sendTerminateRequest() with status 8.
    * @param Message the error message
    * @see MPIErrorHandling::error(const string& Message)*/
   static void error(const std::string& Message);

   /** Prints the given message to std out and exits with exit code 8.
    *  Use only if necessary!
    * @param Message the error message
    * @see MPIErrorHandling::error(const char* Message) */
   static void errorFatal(const char *Message);

   /** Prints the given message to std out and exits with exit code 8.
    *  Use only if necessary!
    * @param Message the error message
    * @see MPIErrorHandling::error(const string& Message)*/
   static void errorFatal(const std::string& Message);

   /** turns listening to terminate requests on
    * */
   static void listenForTerminateRequest(void);

   /** sends a signal to all other ranks and sets their terminateProgram to 1, including self
    *  This can produce a deadlock when other ranks are waiting for a broadcast.
    *  @param __status exits with this status if nonzero. If zero, return but terminateProgram = 1
    * */
   static void sendTerminateRequest(const int __status);

   //JO

   /** communicates the last signal, that was caught by rank 0, if there was one
    *  This uses a MPI_Bcast to avoid a deadlock. This might perform a little worse, but works.
    *  @param signal The signal to communicate (0 if there was no signal)
    *  @return indication, whether there was a signal to react to*/
   static bool broadcastSignal(int&signal);

   //\JO

   /** tests if there is a terminate request. If there is one, the result is evaluated. The rank then calls listenForTerminateRequest().
    * @param signal If there is a terminate request, this variable contains the signal value. Otherwise -1
    * @return rank who has sent a terminate request, or -1 if no request
    * */
   static int checkTerminateRequest(int *signal);

   /** Prints the given message to std out as a warning.
    * @param Message the warning message
    * @see MPIErrorHandling::warning(const char* Message) */
   static void warning(const char *Message);

   static void warning(const std::string& Message)
   {
      warning(Message.c_str());
   }

   // static members --------------------------------------
   //
   // static MPI_Request requestTerminate;
   // static MPI_Status statusTerminate;

   /** This variable is 0 by default. It is set to 1 if checkTerminateRequest() gets a request. */
   static int flagReceivedTerminate;

   /** The signal received from rank receivedFrom */
   static int signalBuffer;
};
#endif // MPIERRORHANDLING_H_

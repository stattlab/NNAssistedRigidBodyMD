#include "MPIErrorHandling.h"
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <unistd.h>


// Define the static members
MPI_Request MPIErrorHandling::requestTerminate;
MPI_Status  MPIErrorHandling::statusTerminate;
int         MPIErrorHandling::flagReceivedTerminate;
int         MPIErrorHandling::signalBuffer;

void MPIErrorHandling::error(const char *text)
{
   int my_rank = -1;

   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   //std::cout << "# Rank["<<my_rank<<"]: ERROR: " << text << std::endl;
   //std::cout << "# ***************************************************************************\n";
   std::cerr << "# Rank[" << my_rank << "]: ERROR: " << text << std::endl;
   std::cerr << "# ***************************************************************************\n";
   sendTerminateRequest(8);
}

void MPIErrorHandling::errorFatal(const char *text)
{
   int my_rank = -1;

   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   //std::cout << "# Rank["<<my_rank<<"]: FATALERROR: " << text << std::endl;
   //std::cout << "# ***************************************************************************\n";
   std::cerr << "# Rank[" << my_rank << "]: FATALERROR: " << text << std::endl;
   std::cerr << "# ***************************************************************************\n";
   exit(8);
}

void MPIErrorHandling::error(const std::string& text)
{
   int my_rank = -1;

   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   //std::cout << "# Rank["<<my_rank<<"]: ERROR: "  << text << std::endl;
   //std::cout << "# ***************************************************************************\n";
   std::cerr << "# Rank[" << my_rank << "]: ERROR: " << text << std::endl;
   std::cerr << "# ***************************************************************************\n";
   sendTerminateRequest(8);
}

void MPIErrorHandling::errorFatal(const std::string& text)
{
   int my_rank = -1;

   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   //std::cout << "# Rank["<<my_rank<<"]: FATALERROR: " << text << std::endl;
   //std::cout << "# ***************************************************************************\n";
   std::cerr << "# Rank[" << my_rank << "]: FATALERROR: " << text << std::endl;
   std::cerr << "# ***************************************************************************\n";
   exit(8);
}

void MPIErrorHandling::listenForTerminateRequest(void)
{
   flagReceivedTerminate = 0;
   MPI_Irecv(&signalBuffer, 1, MPI_INT, MPI_ANY_SOURCE, TAG_TERMINATE, MPI_COMM_WORLD, &requestTerminate);
}

void MPIErrorHandling::sendTerminateRequest(const int __status)
{
   int my_rank = -1;
   int size    = -1;

   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   MPI_Request *requestSent = new MPI_Request[size];

   //LOG_MPI("Sending terminateRequest to all ranks.");
   std::cout << "# Rank[" << my_rank << "]: Sending terminateRequest to all ranks" << std::endl;
   for (int r = 0; r < size; r++)
   {
      int send = __status;
      MPI_Isend(&send, 1, MPI_INT, r, TAG_TERMINATE, MPI_COMM_WORLD, &requestSent[r]);
   }
   //LOGP_MPI("Waiting for other ranks to receive terminateRequest, size == ", size);
   //MPI_Status* statusSent = new MPI_Status[size];
   //MPI_Waitall(size, requestSent, statusSent);
   //int completed = size;
   //delete[] requestSent;
   //delete[] statusSent;

   // The following is an alternative to MPI_Waitall, but it is not sure how it behaves. Waitall however, can cause a deadlock.
   // if status==0, I do not care about whether the messages are received or not. I will maybe change it later.
   if (__status != 0)
   {
      //LOG_MPI("Waiting for other ranks to receive terminateRequest...");
      std::cout << "# Rank[" << my_rank << "]: Waiting for other ranks to receive terminateRequest..." << std::endl;
      int         completed        = 0;
      int *       indexOfCompleted = new int[size];
      MPI_Status *statusSent       = new MPI_Status[size];

      if (size > 1)
      {
         sleep(TIMEMAXWAITBEFOREEXIT);
      }
      //MPI_Testsome(size, requestSent, &completed, indexOfCompleted, statusSent);
      MPI_Waitsome(size, requestSent, &completed, indexOfCompleted, statusSent);

      if (completed == size)
      {
         //LOG_MPI("Communication successful for all ranks.");
         std::cout << "# Rank[" << my_rank << "]: Communication successful for all ranks." << std::endl;

         if (__status != 0)
         {
            //LOG_MPI("Reached MPI_Finalize() after sending terminateRequest.");
            std::cout << "# Rank[" << my_rank << "]: Reached MPI_Finalize() after sending terminateRequest." << std::endl;
            MPI_Finalize();
            exit(__status);
         }
      }
      else
      {
         //LOGP_MPI("Communication successful for N ranks: ", completed);
         std::cout << "# Rank[" << my_rank << "]: Communication successful for N ranks: " << completed << std::endl;
         warning("MPIErrorHandling::sendTerminateRequest(): Did not reach all other ranks.");
         for (int i = 0; i < size; i++)
         {
            //LOGP_MPI("indexOfCompleted: ", indexOfCompleted[i]);
            std::cout << "# Rank[" << my_rank << "]: indexOfCompleted: " << indexOfCompleted[i] << std::endl;
         }
         //LOG_MPI("Communication failed. Other ranks are possibly stuck in a broadcast. Exiting now!");
         std::cout << "# Rank[" << my_rank << "]: Communication failed. Other ranks are possibly stuck in a broadcast. Exiting now!" << std::endl;
         exit(__status);
      }
      delete[] statusSent;
      delete[] indexOfCompleted;
   }

   delete[] requestSent;
   //LOG_MPI("Returning to program.");
   std::cout << "# Rank[" << my_rank << "]: Returning to program." << std::endl;
   return;
}

//JO
bool MPIErrorHandling::broadcastSignal(int&signal)
{
   bool signalSent   = false;
   int  signalBuffer = -1;

   int my_rank = -1;
   int size    = -1;

   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   //rank 0 handles all signals and the walltime check
   if ((my_rank == 0) && (signal != 0))
   {
      signalBuffer = signal;
   }
   MPI_Bcast(&signalBuffer, 1, MPI_INT, 0, MPI_COMM_WORLD);
   if (signalBuffer != -1)
   {
      signal     = signalBuffer;
      signalSent = true;
   }

   return(signalSent);
}

//\JO

int MPIErrorHandling::checkTerminateRequest(int *signal)
{
   int receivedFrom = -1;
   int s            = -1;

   if (!flagReceivedTerminate)
   {
      //  MPI_Test(&requestTerminate, &flagReceivedTerminate, &statusTerminate);
      //   std::cout<< "in MPIErrorHandling::checkTerminateRequest(): "<< flagReceivedTerminate<< " "<< &requestTerminate<<std::endl;

      MPI_Request_get_status(requestTerminate, &flagReceivedTerminate, &statusTerminate);
      if (flagReceivedTerminate)
      {
         int my_rank = -1;
         int size    = -1;
         MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
         MPI_Comm_size(MPI_COMM_WORLD, &size);
         receivedFrom = statusTerminate.MPI_SOURCE;

         MPI_Status status;
         MPI_Wait(&requestTerminate, &status);
         s = signalBuffer;
         //  std::cout << "# Rank["<<my_rank<<"]: MPIErrorHandling::checkTerminateRequest(): Received signal "<<s<<" from rank "<<receivedFrom<<std::endl;
         // listen for more requests
         listenForTerminateRequest();
      }
   }
   *signal = s;
   return(receivedFrom);
}

void MPIErrorHandling::warning(const char *text)
{
   int my_rank = -1;

   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   std::cout << "# Rank[" << my_rank << "]: WARNING: " << text << std::endl;
   //std::cerr << "# Rank["<<my_rank<<"]: WARNING: " << text << std::endl;
}

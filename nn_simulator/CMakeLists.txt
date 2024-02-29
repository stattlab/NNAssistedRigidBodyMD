cmake_minimum_required(VERSION 3.0)
set (CMAKE_CXX_STANDARD 17)
project(nnsim)
find_package(Torch REQUIRED)
set( CMAKE_VERBOSE_MAKEFILE on )

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
#RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

message("system: ${CMAKE_SYSTEM_PREFIX_PATH}")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -ffast-math -march=native -funsafe-math-optimizations -pthread")

set(CMAKE_DEBUG_POSTFIX _debug)
include_directories(nnsim PRIVATE ./mc_sim_utils/)


set(sources ${sources}
    main.cpp
    gsd_utils/GSDReader.cpp
    gsd_utils/GSDWriter.cpp
    gsd_utils/gsd.c
    Sim.cpp
    mc_sim_utils/Box.cpp
    mc_sim_utils/BoxCuboid.cpp
    mc_sim_utils/NeighborListTree.cpp
    mc_sim_utils/Particle.cpp
    mc_sim_utils/Point.cpp
    CommandLineArgumentParser.cpp
)

#    mc_sim_utils/MonteCarloUtils.cpp
#    mc_sim_utils/Mersenne.cpp
# Specify build target
add_executable(nnsim ${sources})
target_link_libraries(nnsim "${TORCH_LIBRARIES}")
set_property(TARGET nnsim PROPERTY CXX_STANDARD 17)
set_target_properties(nnsim PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})

import numpy as np
import sys, os
# sys.path.insert(0,"/home/bargun2/Programs/hoomd-blue/build/hoomd-2.9.2/hoomd")
# sys.path.insert(0,"/home/bargun2/Programs/azplugins_fork/build/debug")
sys.path.insert(0,"/home/bargun2/Programs/azplugins_fork/build/release")
import hoomd
import hoomd.md
import azplugins
import gsd.hoomd
import argparse
from SimulatorHandler import SimulatorHandler

"""
Run the MD simulations and save the snapshots from which pairs are sampled. 
"""

parser = argparse.ArgumentParser(description="Basic run")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-g', '--gpu', metavar="<int>", type=int, dest="gpu_id", required=True, help="0 or 1, -1 for cpu",default=0)
non_opt.add_argument('--N', metavar="<int>", type=int, dest="N_mp", required=True, help=" - ")
non_opt.add_argument('--L', metavar="<float>", type=float, dest="box_size", required=True, help=" - ")
non_opt.add_argument('--output', metavar="<dat>", type=str, dest="output", required=True, help="p_shear L=20 ye NOM 250 per MP ",default=0 )
non_opt.add_argument('--ts', metavar="<int>", type=int, dest="timesteps", required=True, help="length of sim ",default=1 )
non_opt.add_argument('--dump', metavar="<int>", type=int, dest="dump_period", required=True, help="length of sim ",default=1 )
non_opt.add_argument('--kT', metavar="<float>", type=float, dest="kT", required=True, help="length of sim ",default=1 )
# non_opt.add_argument('--seed', metavar="<int>", type=int, dest="seed", required=True, help="length of sim ",default=1 )
non_opt.add_argument('--dt', metavar="<float>", type=float, dest="dt", required=True, help=" - ")

args = parser.parse_args()
gpu_id = args.gpu_id
N_mp = args.N_mp
output = args.output
box_size = args.box_size
timesteps = args.timesteps
dump_period = args.dump_period
dt = args.dt
# seed = args.seed

shape = '../shapes/cube_v2.gsd'
# shape = '../shapes/silindir_01.gsd'

log_period = dump_period
notice_level = 2
kT = args.kT


context_initialize_str = "--gpu=" + str(gpu_id)

if(gpu_id < -0.5):
    context_initialize_str = "--mode=cpu"

hoomd.context.initialize(context_initialize_str)
sim = hoomd.context.SimulationContext()
hoomd.option.set_notice_level(notice_level)

handler = SimulatorHandler(shape)
handler.setBox(box_size,N_mp)
snapshot = handler.get_snap(sim)
system = hoomd.init.read_snapshot(snapshot)
pos  = handler.getRigidConstituents()

rigid = hoomd.md.constrain.rigid()
rigid.set_param('Re', types = ['Os']*len(pos), positions = pos)
rigid.validate_bodies()

g_rigid_centers = hoomd.group.rigid_center()

nl = hoomd.md.nlist.tree()

hphc = azplugins.pair.halfpairhalfcosine(r_cut=2.396, nlist=nl)
hphc.pair_coeff.set('Os','Os', pwr=2.5, depth=0.00035, pos=1.46375, width=3.3833)
hphc.pair_coeff.set(['Re'],['Os','Re'], pwr=2.5, depth=0.00035, pos=1.46375, width=3.3833, r_cut=False)

if not os.path.exists('./out/'):
    os.makedirs('./out/')

n =  './out/' + output +'.gsd'
nlog = './out/' + output +'.log'

hoomd.md.integrate.mode_standard(dt=dt,aniso=True)
nvt = hoomd.md.integrate.nvt(group=g_rigid_centers, kT=kT, tau=0.5)
nvt.randomize_velocities(seed = np.random.randint(low=5,high=100))

d = hoomd.dump.gsd(n, period=dump_period,dynamic=['attribute', 'momentum'], group=hoomd.group.all(), overwrite=False)
analyzer = hoomd.analyze.log(filename=nlog, quantities=['potential_energy', 'kinetic_energy','translational_kinetic_energy','rotational_kinetic_energy','temperature'],
period=log_period, header_prefix='#', overwrite=False)
hoomd.run(timesteps)




exit(0)

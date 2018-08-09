"""
Evolve tracer under simple advection-diffusion.

Advection:
    dt(c) = - U.grad(c) + k*lap(c)

"""

import numpy as np
from mpi4py import MPI
import time
import h5py

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools import post

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Lz = (2., 1.)
Nx, Nz = 512, 256
D = 1e-4
initial_dt = 1e-3
safety = 1.0
stop_time = 10

# Create bases and domain
x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
z_basis = de.Fourier('z', Nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# Load date
with h5py.File("snapshots_pressure/snapshots_pressure_s1.h5", "r") as file:
    u_data = file['tasks']['u'][-1]
    w_data = file['tasks']['w'][-1]
scales = (512/Nx, 256/Nz)
slices = domain.distributor.grid_layout.slices(scales=scales)
u = domain.new_field(scales=scales)
w = domain.new_field(scales=scales)
u['g'] = u_data[slices]
w['g'] = w_data[slices]

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['c'])
problem.parameters['u'] = u
problem.parameters['w'] = w
problem.parameters['D'] = D
problem.substitutions['Lc'] = "dx(dx(c)) + dz(dz(c))"
problem.add_equation("dt(c) - D*Lc = - u*dx(c) - w*dz(c)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

# Initial conditions
x, z = domain.grids()
c = solver.state['c']
x0 = 0
sigma = 0.1
c['g'] = np.exp(-((x-x0)**2) / (2*sigma**2))

# Integration parameters
solver.stop_sim_time = stop_time
solver.stop_wall_time = 10*60
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots_tracer', sim_dt=0.1, max_writes=50, mode='overwrite')
snapshots.add_system(solver.state, scales=2)
snapshots.add_task("u", scales=2)
snapshots.add_task("w", scales=2)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=initial_dt, cadence=10, safety=safety,
                     max_change=1.5, min_change=0.5, threshold=0.05)
CFL.add_velocities(('u', 'w'))

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
    post.merge_process_files(snapshots.base_path, cleanup=True)

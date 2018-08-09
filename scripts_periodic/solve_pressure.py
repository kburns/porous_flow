"""
Find porous flow solution.

Darcy flow:
    div(U) = 0
    U = - K grad(p)
    div(K grad(p)) = 0
    grad(K).grad(p) + K*lap(p) = 0
    grad(log(K)).grad(p) + lap(p) = 0

Time-marching scheme to reach fixed point:
    -dt(lap(p)) = lap(p) + grad(log(K)).grad(p)

"""

import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools import post

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Lz = (2., 1.)
Nx, Nz = 512, 256
ln_K_std = 0.5
kmax = 100
tolerance = 1e-10
dt = 1
max_iter = 1000

# Create bases and domain
x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
z_basis = de.Fourier('z', Nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# Create permability field
def random_2d_fourier(kx, ky, kmax):
    k2 = kx*kx + ky*ky
    rand = np.random.randn(kx.size, ky.size) + 1j*np.random.randn(kx.size, ky.size)
    mask = (k2 < kmax**2)
    N = np.pi * np.abs(kmax/(kx[1,0]-kx[0,0])) * np.abs(kmax/(ky[0,1]-ky[0,0]))
    return rand * mask / np.sqrt(N)

kx = domain.elements(0)
kz = domain.elements(1)
ln_K = domain.new_field()
ln_K['c'] = ln_K_std * random_2d_fourier(kx, kz, kmax)
ln_K.require_grid_space()

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p1'])
problem.parameters['Lx'] = Lx
problem.parameters['Lz'] = Lz
problem.parameters['ln_K'] = ln_K
problem.parameters['K'] = np.exp(ln_K)
problem.substitutions['p0x'] = "-1"
problem.substitutions['p0z'] = "0"
problem.substitutions['p1x'] = "dx(p1)"
problem.substitutions['p1z'] = "dz(p1)"
problem.substitutions['px'] = "p0x + p1x"
problem.substitutions['pz'] = "p0z + p1z"
problem.substitutions['Lp0'] = "dx(p0x) + dz(p0z)"
problem.substitutions['Lp1'] = "dx(p1x) + dz(p1z)"
problem.add_equation("- dt(Lp1) - Lp1 = dx(ln_K)*px + dz(ln_K)*pz + Lp0", condition="(nx != 0) or (nz != 0)")
problem.add_equation("p1 = 0", condition="(nx == 0) and (nz == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK111)
logger.info('Solver built')

# Integration parameters
solver.stop_sim_time = np.inf
solver.stop_wall_time = np.inf
solver.stop_iteration = max_iter

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots_pressure', iter=1, max_writes=np.inf, mode='overwrite')
snapshots.add_system(solver.state)
snapshots.add_task("K")
snapshots.add_task("ln_K")
snapshots.add_task("-K*px", name="u")
snapshots.add_task("-K*pz", name="w")
snapshots.add_task("integ((dx(ln_K)*px + dz(ln_K)*pz + Lp0 + Lp1)**2)/Lx/Lz", name="residual")

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("integ((dx(ln_K)*px + dz(ln_K)*pz + Lp0 + Lp1)**2)/Lx/Lz", name="residual")

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    residual = np.inf
    while solver.ok and residual > tolerance:
        dt = solver.step(dt)
        residual = flow.max("residual")
        if (solver.iteration-1) % 1 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info("Residual  = %e" %residual)
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

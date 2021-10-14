#################
## Description ##
#################
# Code for implementing the EGM method when solving a simple 1d consumption saving problem.
# Income evolves stochastically as a 1st order Markov process.

#############
## Imports ##
#############

import numpy as np
import tools
import utility as util

# Compile with numba soon

######################
## Solver using EGM ##
######################

# Write this better!

def solve(sol, par, c_next, m_next):

    # Copy last iteration of the value function
    v_old = sol.v.copy()

    # Expand exogenous asset grid
    a = np.tile(par.grid_a, np.size(par.y)) # 2d end-of-period asset grid

    # Loop over exogneous states (post decision states)
    for a_i,a in enumerate(par.grid_a):

        # Compute expected assets in the next period 
        # m_plus = (1+par.r)*a + np.transpose(par.y) # Transpose for dimension to fit

        # E_m_plus = np.sum(par.w*par.eps)*a + np.transpose(par.y)

        # # Interpolate next periods consumption - can this be combined?
        # c_plus_1 = tools.interp_linear_1d(m_next[0,:], c_next[0,:], E_m_plus) # State 1
        # c_plus_2 = tools.interp_linear_1d(m_next[1,:], c_next[1,:], E_m_plus) # State 2

        # #Combine into a vector. Rows indicate income state, columns indicate asset state
        # c_plus = np.vstack((c_plus_1, c_plus_2))

        # Marginal utility
        # marg_u_plus = util.marg_u(c_plus,par) # This must be recomputed using the Guass Hermite weights!!!

        # Find expected marginal utility in the next period, given each realization of y_t+1
        marg_u_plus_1 = np.sum(par.w*par.eps * util.marg_u(tools.interp_linear_1d(m_next[0,:], c_next[0,:], par.eps*a + par.y1), par))
        marg_u_plus_2 = np.sum(par.w*par.eps * util.marg_u(tools.interp_linear_1d(m_next[1,:], c_next[1,:], par.eps*a + par.y2), par))

        # Combine
        marg_u_plus = np.vstack((marg_u_plus_1, marg_u_plus_2))

        # Compute the final expectation using the markov process for income
        av_marg_u_plus = np.array([par.P[0,0]*marg_u_plus_1 + par.P[0,1]*marg_u_plus_2, par.P[1,1]*marg_u_plus_2 + par.P[1,0]*marg_u_plus_1])

        # Add optimal consumption and endogenous state
        sol.c[:,a_i+1] = util.inv_marg_u(par.beta*av_marg_u_plus,par)
        sol.m[:,a_i+1] = a + sol.c[:,a_i+1]
        sol.v = util.u(sol.c,par)

    #Compute value function and update iteration parameters
    sol.delta = max( max(abs(sol.v[0] - v_old[0])), max(abs(sol.v[1] - v_old[1])))
    sol.it += 1

    # sol.delta = max( max(abs(sol.c[0] - c_next[0])), max(abs(sol.c[1] - c_next[1])))

    return sol
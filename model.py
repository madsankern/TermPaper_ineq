# Defines the simpel 1d model to be solved

# import packages 
import numpy as np
import tools # User written
from types import SimpleNamespace
import egm
import utility as util

class model_class():

    def __init__(self,name=None):
        """ defines default attributes """

        # Names
        self.par = SimpleNamespace()
        self.sol_egm = SimpleNamespace()

    ###########
    ## Setup ##
    ###########
    # Setup parameters used for all solvers
    # for the 1d model of consumption

    def setup(self):

        # Initialize
        par = self.par

        # Model
        par.beta = 0.96
        par.eta = 1.0
        
        par.r = 0.01
        par.y1 = 1.0
        par.y2 = 2.0
        par.y = np.array([par.y1, par.y2])
        
        par.P_11 = 0.6
        par.P_22 = 0.9
        par.P = np.array([[par.P_11, 1 - par.P_11], [1 - par.P_22, par.P_22]]) # Transition matrix

        # Grid settings
        par.Nm = 500
        par.m_max = 20.0
        par.m_min = 1e-6

        par.Na = par.Nm
        par.a_min = par.m_min
        par.a_max = par.m_max # Check this out later
        
        par.max_iter = 1000
        par.tol_egm = 1.0e-6

        par.N_bottom = 10

        ##############################
        ## Things for Gauss-Hermite ##
        ##############################
        # Stochastic process
        par.sigma2 = 0.02
        par.mu = np.log(1.0 + par.r) - par.sigma2/2
        
        par.num_shocks = 5 # Number of draws to approximate the integral
        x,w = tools.gauss_hermite(par.num_shocks) # Load values and associated weights

        # Reshape to fit the model
        par.eps = np.exp(np.sqrt(2)*np.sqrt(par.sigma2)*x + par.mu)
        par.w = w / np.sqrt(np.pi)

    # Asset grids
    def create_grids(self):

        par = self.par
        
        # Pre desicion
        # par.grid_m = np.linspace(par.m_min, par.m_max, par.Nm)
        
        # Post desicion
        par.grid_a = np.linspace(par.a_min, par.a_max, par.Na)
        
        # x grid
        # par.grid_x = np.linspace(par.x_min, par.x_max, par.Nx)

        # Convert these to nonlinspace later.

###########################################
##### SIMPLE CONUSMPTION SAVING MODEL #####
###########################################

    #############################
    ## Endogeneous grid method ##
    #############################

    def solve_egm(self):

        # Initialize
        par = self.par
        sol = self.sol_egm

        # Shape parameter for the solution vector
        shape = (np.size(par.y),1)
        
        # Initial guess is like a 'last period' choice - consume everything
        # This needs to be optimized
        sol.m = np.tile(np.linspace(par.a_min,par.a_max,par.Na+1), shape) # a is pre descision, so for any state consume everything.
        sol.c = sol.m.copy() # Consume everyting
        sol.v = util.u(sol.c,par) # Utility of consumption

        sol.it = 0 # Iteration counter
        sol.delta = 1000.0 # Difference between iterations

        # Iterate value function until convergence or break if no convergence
        while (sol.delta >= par.tol_egm and sol.it < par.max_iter):

            # Use last iteration to compute the continuation value
            # therefore, copy c and a grid from last iteration.
            c_next = sol.c.copy()
            m_next = sol.m.copy()           

            # Call EGM function
            sol = egm.solve(sol, par, c_next, m_next)

            # add zero consumption (not really necessary for current initial guess, where everything is consumed)
            sol.m[:,0] = 1e-6
            sol.c[:,0] = 1e-6
        
        print(sol.it)